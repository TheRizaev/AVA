"""Pedestrian rules: jaywalking and failure_to_yield."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .. import config
from ..scene import polygon, union_mask
from ..tracks import Track
from .base import CROSSWALKS, NOT_CARRIAGEWAY, Context, Event, fill_short_gaps, runs_of_true

CURB_MARGIN_BODY = 0.6    # a pedestrian must be this many body heights (~1 m) inside the carriageway
MIN_CURB_MARGIN = 12.0    # px
ZEBRA_MARGIN_BODY = 0.15  # a jaywalker is more than this many body heights (~0.25 m) from any zebra
MIN_ZEBRA_MARGIN = 10.0   # px
MIN_JAYWALK_SEC = 1.0
RIDER_OVERLAP = 0.35      # fraction of a person box covered by a two-wheeler/car box -> rider/occupant
MAX_PEDESTRIAN_BODY_SPEED = 2.5  # body heights per second; a runner is ~2, a scooter rider much more
U_NEAR = 0.25             # pedestrian within this fraction of the crossing's length from the vehicle's crossing point
U_CLEARED = 0.10          # ... and not already this far past the stretch it still has to drive over, walking away
WALKING_SPEED = 15.0      # px/s along the crossing
VEHICLE_FOOT_DEPTH = 0.5  # lower half of the box = where the vehicle meets the road
ZEBRA_TOLERANCE = 6       # px around the zebra that still counts as 'on it' (feet at the edge)
KERB_END = 0.13           # fraction of the zebra length at each end where people wait on the kerb
MIN_THROUGH_SPEED = 50.0  # px/s: a vehicle this fast on the crossing is driving through
MIN_THROUGH_TRAVEL = 30.0  # px the vehicle moves while on the crossing (a parked car's box jitter does not)
WALKED_TWO_WHEELER = 2.2  # box heights per second: a moped slower than this on a crossing is pushed on foot
CROSSING_WINDOW = 3.0     # s either side of a sample over which a pedestrian's progress along the zebra is measured
MIN_CROSSING_TRAVEL = 55.0  # px along the zebra in that window: less = standing (waiting at a refuge / kerb), not crossing


@lru_cache(maxsize=None)
def _dilated_crosswalk(name: str, px: int) -> np.ndarray:
    m = polygon(name).mask().astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k).astype(bool)


def _vehicle_boxes_by_frame(ctx: Context) -> dict[int, np.ndarray]:
    boxes: dict[int, list[np.ndarray]] = {}
    for tr in ctx.kind("vehicle") + ctx.kind("bicycle"):
        for f, b in zip(tr.frame, tr.box):
            boxes.setdefault(int(f), []).append(b)
    return {f: np.array(b) for f, b in boxes.items()}


def _rider_fraction(person: Track, veh_boxes: dict[int, np.ndarray]) -> float:
    """Share of the person's samples in which a vehicle/bicycle box covers the person box."""
    hits = 0
    for f, b in zip(person.frame, person.box):
        vb = veh_boxes.get(int(f))
        if vb is None:
            continue
        ix = np.clip(np.minimum(vb[:, 2], b[2]) - np.maximum(vb[:, 0], b[0]), 0, None)
        iy = np.clip(np.minimum(vb[:, 3], b[3]) - np.maximum(vb[:, 1], b[1]), 0, None)
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
        if (ix * iy / area).max() > RIDER_OVERLAP:
            hits += 1
    return hits / max(len(person), 1)


def _pedestrians(ctx: Context) -> list[Track]:
    """Person tracks that are real pedestrians (not riders or vehicle occupants).

    A person whose box rides on a two-wheeler/car box is dropped, and so is one
    moving faster than a run (in body heights per second): a rider whose
    vehicle the detector missed.
    """
    veh_boxes = _vehicle_boxes_by_frame(ctx)
    out = []
    for p in ctx.kind("person"):
        body = np.maximum(p.box[:, 3] - p.box[:, 1], 10.0)
        if np.median(p.speed(0.5) / body) > MAX_PEDESTRIAN_BODY_SPEED:
            continue
        if _rider_fraction(p, veh_boxes) < 0.5:
            out.append(p)
    return out


STITCH_GAP = 1.5          # s: longest occlusion bridged between two fragments of one person
STITCH_DIST_BODY = 1.0    # body heights between where one fragment ends and the next starts
HIDDEN_FEET_RATIO = 0.6   # box shorter than this share of a standing person at that row: feet hidden
HOP_SEC = 2.0             # s: longest stretch on an island tip that is walked over, not stood on
HOP_MIN_SPEED = 0.1       # body heights per second: moving across the island, not standing at its edge


def _height_model(peds: list[Track]) -> np.ndarray | None:
    """Median person box height as a linear function of the box bottom row (the perspective)."""
    y = np.concatenate([p.box[~p.at_border, 3] for p in peds]) if peds else np.zeros(0)
    if len(y) < 200:
        return None
    h = np.concatenate([(p.box[:, 3] - p.box[:, 1])[~p.at_border] for p in peds])
    lo, hi = np.percentile(y, [1, 99])
    cy, ch = [], []
    for b in np.arange(lo, hi, 40.0):
        m = (y >= b) & (y < b + 40.0)
        if m.sum() >= 50:
            cy.append(np.median(y[m]))
            ch.append(np.median(h[m]))
    if len(cy) < 3:
        return None
    a, b = np.polyfit(cy, ch, 1)
    return np.array([a, b]) if 0 < a < 0.5 else None      # people grow towards the camera, never degenerate


def _restore_hidden_feet(p: Track, model) -> tuple[Track | None, np.ndarray]:
    """A box much shorter than a standing person at its row has its lower body hidden behind
    someone in front, so its bottom edge is not the ground point: interpolate the ground point
    and body height of those samples from the visible ones (None: never seen whole)."""
    body = (p.box[:, 3] - p.box[:, 1]).astype(float)
    if model is None:
        return p, body
    a, b = model
    hidden = body < HIDDEN_FEET_RATIO * (a * p.box[:, 3] + b)
    if not hidden.any():
        return p, body
    if hidden.all():
        return None, body
    seen = ~hidden
    xy = p.xy.copy()
    for d in (0, 1):
        xy[hidden, d] = np.interp(p.t[hidden], p.t[seen], p.xy[seen, d])
    body[hidden] = np.interp(p.t[hidden], p.t[seen], body[seen])
    # before the first / after the last visible sample interpolation would pin a moving person to
    # one spot: place the feet where a standing person with the head at the box top would have them
    first, last = np.flatnonzero(seen)[[0, -1]]
    ends = np.flatnonzero(hidden & ((np.arange(len(p)) < first) | (np.arange(len(p)) > last)))
    if len(ends):
        y_top = p.box[ends, 1]
        height = (y_top + b) / (1 - a) - y_top
        top = (p.corners[ends, 0] + p.corners[ends, 1]) / 2
        bot = (p.corners[ends, 2] + p.corners[ends, 3]) / 2
        xy[ends] = top + (bot - top) * (height / np.maximum(p.box[ends, 3] - y_top, 1.0))[:, None]
        body[ends] = height
    return Track(p.tid, p.cls, p.kind, p.frame, p.t, p.box, xy, p.score, p.corners), body


def _stitch(items: list[tuple[Track, np.ndarray]]) -> list[tuple[Track, np.ndarray]]:
    """Join the fragments of one person that the tracker split at an occlusion (new id after it):
    a track starting <= STITCH_GAP s after another ended, within STITCH_DIST_BODY body heights of
    where it ended; one-to-one, nearest first."""
    items = sorted(items, key=lambda it: it[0].t[0])
    cands = []
    for i, (a, body_a) in enumerate(items):
        reach = STITCH_DIST_BODY * float(np.median(body_a[-5:]))
        for j, (b, _) in enumerate(items):
            gap = b.t[0] - a.t[-1]
            if i != j and 0 < gap <= STITCH_GAP:
                d = float(np.hypot(*(b.xy[0] - a.xy[-1])))
                if d <= reach:
                    cands.append((d, i, j))
    nxt, prv = {}, {}
    for _, i, j in sorted(cands):
        if i not in nxt and j not in prv:
            nxt[i], prv[j] = j, i
    out = []
    for i in range(len(items)):
        if i in prv:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) == 1:
            out.append(items[i])
            continue
        trs = [items[k][0] for k in chain]
        cat = lambda f: np.concatenate([getattr(tr, f) for tr in trs])  # noqa: E731
        head = trs[0]
        out.append((Track(head.tid, head.cls, head.kind, cat("frame"), cat("t"), cat("box"), cat("xy"),
                          cat("score"), cat("corners")),
                    np.concatenate([items[k][1] for k in chain])))
    return out


def _bridge_island_hops(t, road, passable, xy, body) -> np.ndarray:
    """Join road runs separated by a short walk over an island tip / the median (not a stand on it)."""
    out = road.copy()
    runs = runs_of_true(road)
    for (_, e1), (s2, _) in zip(runs, runs[1:]):
        dt = t[s2] - t[e1]
        if dt > HOP_SEC or not passable[e1 + 1:s2].all():
            continue
        if np.hypot(*(xy[s2] - xy[e1])) < HOP_MIN_SPEED * np.median(body[e1:s2 + 1]) * dt:
            continue      # position flickering across an island edge while standing on it
        out[e1 + 1:s2] = True
    return out


def jaywalking(ctx: Context) -> list[Event]:
    """A pedestrian on the carriageway away from the kerb and outside the crossings for >= 1 s.

    Tolerances scale with the person's size (so they are roughly metric): more
    than ~1 m inside the carriageway, more than ~0.25 m from any zebra. Samples
    whose box is cut by the frame edge are ignored and samples whose feet are
    hidden behind someone in front are interpolated (the box bottom is not a
    ground point there). Fragments of one person split by an occlusion are
    re-linked, and the event is extended to the moment the feet crossed the
    kerb, also across a short walk over an island tip.
    """
    stop_zone = polygon("stop_zone").mask()
    refuges = union_mask(list(NOT_CARRIAGEWAY))
    peds = _pedestrians(ctx)
    model = _height_model(peds)
    items = [_restore_hidden_feet(p, model) for p in peds]
    items = _stitch([(p, body) for p, body in items if p is not None])
    events = []
    for p, body in items:
        xy = p.smoothed_xy(0.4)
        # people between the stop line and cw1 are walking round cars stopped on the crossing
        beside_crossing = ctx.sample(stop_zone, xy)
        dcw = ctx.sample(ctx.dist_to_crosswalk, xy)
        margin = np.maximum(CURB_MARGIN_BODY * body, MIN_CURB_MARGIN)
        off_zebra = dcw > np.maximum(ZEBRA_MARGIN_BODY * body, MIN_ZEBRA_MARGIN)
        on_road = (ctx.sample(ctx.dist_to_sidewalk, xy) >= margin) & off_zebra & ~p.at_border & ~beside_crossing
        on_road = fill_short_gaps(p.t, on_road, 1.0)
        on_carriageway = ctx.sample(ctx.carriageway, xy)
        road = on_carriageway & (dcw > MIN_ZEBRA_MARGIN) & ~p.at_border & ~beside_crossing
        passable = (dcw > MIN_ZEBRA_MARGIN) & ~beside_crossing & (ctx.sample(refuges, xy) | on_carriageway)
        road = _bridge_island_hops(p.t, road, passable, xy, body)
        for s, e in runs_of_true(on_road):
            if p.t[e] - p.t[s] < MIN_JAYWALK_SEC:
                continue
            # extend to the moment the feet crossed the kerb (without the margin)
            while s > 0 and road[s - 1]:
                s -= 1
            while e + 1 < len(p) and road[e + 1]:
                e += 1
            events.append(Event(float(p.t[s]), float(p.t[e]), "jaywalking", 1.0, {"track": p.tid}))
    return events


def _zebra_axis(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Origin, unit direction and length of a crossing's long axis (kerb to kerb)."""
    pts = polygon(name).pts
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre)
    d = vt[0]
    proj = (pts - centre) @ d
    return centre + d * proj.min(), d, float(proj.max() - proj.min())


def _window_travel(t: np.ndarray, x: np.ndarray, half: float) -> np.ndarray:
    """Range of x within +-half seconds of every sample."""
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    return np.array([np.ptp(x[a:b]) for a, b in zip(lo, hi)])


def failure_to_yield(ctx: Context) -> list[Event]:
    """A vehicle drives through a crossing while a pedestrian is on it, close to the vehicle's path.

    Positions are compared along the crossing's own axis (u = 0..1 from kerb to
    kerb), which is roughly metric and free of perspective: a pedestrian on the
    zebra's travel part (not waiting at its kerb ends), actually crossing (not
    standing still on it), within U_NEAR of the vehicle's crossing point
    counts, unless they have already cleared the vehicle's path - are more
    than U_CLEARED outside the stretch of the crossing the vehicle still has
    to drive over (from its current point to where it leaves the crossing) -
    and are walking away from it. Start/end: the vehicle's footprint enters /
    leaves the crossing.
    """
    peds = _pedestrians(ctx)
    events = []
    for name in CROSSWALKS:
        on_zebra = _dilated_crosswalk(name, ZEBRA_TOLERANCE)
        zone_tight = polygon(name).mask()
        origin, axis, length = _zebra_axis(name)
        ped_t, ped_u, ped_du = [], [], []
        for p in peds:
            u = (p.xy - origin) @ axis / length
            du = (p.velocity(0.6) @ axis) / length
            crossing = _window_travel(p.t, u * length, CROSSING_WINDOW) >= MIN_CROSSING_TRAVEL
            on = ctx.sample(on_zebra, p.xy) & (u > KERB_END) & (u < 1 - KERB_END) & crossing
            ped_t.append(p.t[on])
            ped_u.append(u[on])
            ped_du.append(du[on])
        if not any(len(x) for x in ped_t):
            continue
        ped_t_all, ped_u_all, ped_du_all = (np.concatenate(x) for x in (ped_t, ped_u, ped_du))
        for v in ctx.kind("vehicle"):
            if v.cls == config.BUS:
                continue     # the box of a long bus covers the crossing far beyond the bus's real path
            fp = v.footprint(depth=VEHICLE_FOOT_DEPTH)
            inside = [zone_tight[np.clip(pts[:, 1].astype(int), 0, zone_tight.shape[0] - 1),
                                 np.clip(pts[:, 0].astype(int), 0, zone_tight.shape[1] - 1)] for pts in fp]
            on = fill_short_gaps(v.t, np.array([m.any() for m in inside]), 0.5)
            speed = v.speed(0.5)
            body = np.maximum(v.box[:, 3] - v.box[:, 1], 10.0)
            for s, e in runs_of_true(on):
                if v.t[e] - v.t[s] < 0.2 or np.median(speed[s:e + 1]) < MIN_THROUGH_SPEED:
                    continue
                if v.cls == config.MOTORCYCLE and np.median(speed[s:e + 1] / body[s:e + 1]) < WALKED_TWO_WHEELER:
                    continue     # a moped pushed along the crossing by a pedestrian
                if np.linalg.norm(v.xy[e] - v.xy[s]) < MIN_THROUGH_TRAVEL:
                    continue
                # the vehicle's crossing point at every sample it is on the crossing, and where it leaves it
                on_k = [k for k in range(s, e + 1) if inside[k].any()]
                if not on_k:
                    continue
                u_at = {k: float(np.median((fp[k][inside[k]] - origin) @ axis / length)) for k in on_k}
                u_exit = u_at[on_k[-1]]
                conflict = False
                for k in on_k:
                    u_v = u_at[k]
                    now = np.abs(ped_t_all - v.t[k]) <= 0.06
                    u_p, du_p = ped_u_all[now], ped_du_all[now]
                    d = u_p - u_v
                    # cleared = more than U_CLEARED outside the stretch the vehicle still has to drive over (its
                    # current point to its exit point) and walking away from it: a turner crossing at a shallow
                    # angle (cw2) sweeps along the zebra and passes people walking the same way ahead of it
                    lo, hi = min(u_v, u_exit), max(u_v, u_exit)
                    beyond = np.maximum(u_p - hi, lo - u_p)
                    walking_away = (beyond > U_CLEARED) & (np.sign(du_p) == np.sign(d)) & \
                        (np.abs(du_p) * length > WALKING_SPEED)
                    if ((np.abs(d) < U_NEAR) & ~walking_away).any():
                        conflict = True
                        break
                if conflict:
                    events.append(Event(float(v.t[s]), float(v.t[e]), "failure_to_yield", 1.0,
                                        {"track": v.tid, "crossing": name}))
    return events
