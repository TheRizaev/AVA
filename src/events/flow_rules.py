"""Traffic-state rules: stopped_vehicle (and congestion, see below)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..config import BUS
from ..scene import polygon, union_mask
from ..signals import GREEN
from .base import Context, Event, fill_short_gaps, runs_of_true

STOP_SPEED = 6.0         # px/s (1 s smoothing) below which a vehicle is stationary
MIN_STOPPED_SEC = 10.0   # class definition
LINK_DIST = 25.0         # px: stationary pieces of different track ids at the same spot are one vehicle
LINK_GAP = 4.0           # s
QUEUE_GREEN_SEC = 12.0   # stationary on the approach through this much green -> not a signal queue
PASSING_RADIUS = 220.0   # px around the stopped vehicle
PASSING_SPEED = 60.0     # px/s
PASSING_SHARE = 0.3      # traffic must pass it in at least this share of its stopped time
KERBSIDE_PX = 100.0      # a stopped vehicle stands next to the kerb (reference px from the roadside kerb)
SEAM_CLOSE_PX = 9        # closes the few-px seams between adjacent layout polygons (a seam is not a kerb)
ISLANDS_AND_MEDIAN = ("median", "island_round", "island_tri1", "island_tri2", "island_3")


@dataclass
class _Stay:
    t0: float
    t1: float
    xy: np.ndarray
    tids: list[int]


def _stationary_pieces(ctx: Context) -> list[_Stay]:
    pieces = []
    for tr in ctx.kind("vehicle"):
        # a bus at the kerb is serving the outbound stop: a scheduled dwell, not a stopped vehicle
        if tr.duration < 2.0 or tr.cls == BUS:
            continue
        speed = tr.speed(1.0)
        still = fill_short_gaps(tr.t, speed < STOP_SPEED, 1.0)
        still &= ~tr.at_border         # a box cut by the frame edge does not show whether it moves
        for s, e in runs_of_true(still):
            if tr.t[e] - tr.t[s] >= 2.0:
                pieces.append(_Stay(float(tr.t[s]), float(tr.t[e]), np.median(tr.xy[s:e + 1], axis=0), [tr.tid]))
    return pieces


def _link(pieces: list[_Stay]) -> list[_Stay]:
    pieces = sorted(pieces, key=lambda p: p.t0)
    out: list[_Stay] = []
    for p in pieces:
        for q in out:
            if np.linalg.norm(q.xy - p.xy) < LINK_DIST and p.t0 - q.t1 < LINK_GAP and p.t1 > q.t1:
                q.t1 = p.t1
                q.tids += p.tids
                break
        else:
            out.append(_Stay(p.t0, p.t1, p.xy, list(p.tids)))
    return out


def _green_seconds(ctx: Context, t0: float, t1: float) -> float:
    """Longest continuous green inside [t0, t1]."""
    sig = ctx.signal
    m = (sig.t >= t0) & (sig.t <= t1)
    if not m.any():
        return 0.0
    g = sig.state[m] == GREEN
    tt = sig.t[m]
    return max((tt[e] - tt[s] for s, e in runs_of_true(g)), default=0.0)


def _traffic_passing(ctx: Context, stay: _Stay, radius: float) -> float:
    """Share of 1-s steps of the stay in which another vehicle drives past within `radius` px."""
    steps = np.arange(stay.t0, stay.t1, 1.0)
    passing = np.zeros(len(steps), bool)
    for tr in ctx.kind("vehicle"):
        if tr.tid in stay.tids or tr.t[-1] < stay.t0 or tr.t[0] > stay.t1:
            continue
        m = (tr.t >= stay.t0) & (tr.t <= stay.t1)
        if not m.any():
            continue
        near = (np.linalg.norm(tr.xy[m] - stay.xy, axis=1) < radius) & (tr.speed()[m] > PASSING_SPEED)
        if near.any():
            k = np.clip(np.searchsorted(steps, tr.t[m][near]) - 1, 0, len(steps) - 1)
            passing[k] = True
    return float(passing.mean()) if len(passing) else 0.0


@lru_cache(maxsize=1)
def _kerbside_maps() -> tuple[np.ndarray, np.ndarray]:
    """(where a stop can count, distance to the ROADSIDE kerb) in reference px.

    The median and the junction islands are not kerbs for this test: a car waiting next to them is
    a turner in the middle of the junction, not a kerbside stop. The car-park apron at the east end
    of cw2 (layout polygon car_park_apron_east) is off the road: cars wait there at the barrier.
    """
    lanes = union_mask(["approach", "outbound", "junction"])
    islands = union_mask(list(ISLANDS_AND_MEDIAN))
    car_park = polygon("car_park_entrance").mask() | polygon("car_park_apron_east").mask()
    carriageway = lanes & ~islands & ~car_park
    road = (lanes | islands) & ~car_park
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEAM_CLOSE_PX, SEAM_CLOSE_PX))
    road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, k)
    dist = cv2.distanceTransform(road, cv2.DIST_L2, 5)
    # on the outbound road away from the junction nobody waits to turn: its median-side lane is a
    # kerb lane too (a car stopped there with traffic passing is a stopped vehicle)
    median = polygon("median").mask()
    dist_median = cv2.distanceTransform((~median).astype(np.uint8), cv2.DIST_L2, 5)
    outbound_only = polygon("outbound").mask() & ~polygon("junction").mask() & ~union_mask(["cw1", "cw2", "cw3"])
    return carriageway, np.where(outbound_only, np.minimum(dist, dist_median), dist)


def stopped_vehicle(ctx: Context) -> list[Event]:
    """A vehicle standing >= 10 s at the roadside kerb while the traffic next to it keeps moving.

    That separates a kerbside stop (dropping off, loading, broken down) from any
    queue - at the signal, inside the junction waiting to turn, or behind a
    downstream jam - where the neighbours stand still too.
    """
    carriageway, dist_to_kerb = _kerbside_maps()
    queue_area = polygon("approach").mask() | polygon("stop_zone").mask() | polygon("cw1").mask()
    events = []
    for stay in _link(_stationary_pieces(ctx)):
        if stay.t1 - stay.t0 < MIN_STOPPED_SEC:
            continue
        if not ctx.sample(carriageway, stay.xy)[0]:
            continue
        if ctx.sample(dist_to_kerb, stay.xy)[0] > KERBSIDE_PX:
            continue     # mid-junction: a turner waiting for a gap
        if ctx.sample(queue_area, stay.xy)[0] and _green_seconds(ctx, stay.t0, stay.t1) < QUEUE_GREEN_SEC:
            continue
        if _traffic_passing(ctx, stay, PASSING_RADIUS) < PASSING_SHARE:
            continue
        end = stay.t1 if stay.t1 < ctx.duration - 0.5 else ctx.duration
        events.append(Event(stay.t0, end, "stopped_vehicle", 1.0,
                            {"tracks": stay.tids, "xy": stay.xy.round().tolist()}))
    return events


CONGESTION_MIN_VEHICLES = 8
CONGESTION_STILL_FRAC = 0.8
CONGESTION_MIN_LANES = 4
CONGESTION_MIN_SEC = 10.0
CONGESTION_BODY_SPEED = 0.35   # crawling: below this many box heights per second
CONGESTION_AFTER_GREEN = 10.0  # approach only: still jammed this long into green (not a signal queue)


def congestion(ctx: Context) -> list[Event]:
    """All lanes of the approach at a standstill / crawling, beyond a normal signal queue.

    Per 0.5 s: approach vehicles present, the share of them crawling (speed in
    box heights per second, so perspective does not matter) and how many lanes
    they occupy (lane-angle coordinate). A jam that is still there
    CONGESTION_AFTER_GREEN s into the green phase is congestion; a queue that
    discharges on green is the signal doing its job.
    """
    from ..scene import lane_angle
    lanes = load_lanes()
    appr = polygon("approach")
    grid = np.arange(0.0, ctx.duration, 0.5)
    n = np.zeros(len(grid))
    still = np.zeros(len(grid))
    lane_sets: list[set] = [set() for _ in grid]
    for tr in ctx.kind("vehicle"):
        inside = appr.contains(tr.xy)
        if not inside.any():
            continue
        body = np.maximum(tr.box[:, 3] - tr.box[:, 1], 10.0)
        crawl = tr.speed(1.0) / body < CONGESTION_BODY_SPEED
        lane = np.digitize(lane_angle(tr.xy), lanes)
        gi = np.clip(np.searchsorted(grid, tr.t), 0, len(grid) - 1)
        for k in np.unique(gi[inside]):
            m = inside & (gi == k)
            n[k] += 1
            still[k] += crawl[m].mean() >= 0.5
            lane_sets[k].add(int(np.bincount(lane[m]).argmax()))
    jam = (n >= CONGESTION_MIN_VEHICLES) & (still >= CONGESTION_STILL_FRAC * np.maximum(n, 1)) & \
        (np.array([len(s) for s in lane_sets]) >= CONGESTION_MIN_LANES)
    # a queue that discharges on green is a signal queue, not congestion
    green_for = np.zeros(len(grid))
    state = ctx.signal.at(grid)
    for k in range(1, len(grid)):
        green_for[k] = green_for[k - 1] + 0.5 if state[k] == GREEN else 0.0
    jam &= green_for >= CONGESTION_AFTER_GREEN
    jam |= _junction_jam(ctx, grid)
    jam = fill_short_gaps(grid, jam, JUNCTION_JAM_GAP)
    return [Event(float(grid[s]), float(grid[e]) + 0.5, "congestion", 1.0, {})
            for s, e in runs_of_true(jam) if grid[e] - grid[s] >= CONGESTION_MIN_SEC]


JUNCTION_JAM_MIN_VEHICLES = 5
JUNCTION_JAM_STILL_FRAC = 0.6
JUNCTION_JAM_GAP = 12.0
# The class is a standstill of A direction. Cross-street cars queued in front of cw2 while the
# pedestrians have their phase (= main-road red) are a yield queue of their own approach; alone they
# are not a jam, however many there are. Most of the minimum jam must be main-road traffic.
JUNCTION_JAM_MIN_MAIN = 3


def _with_main_road(tr) -> bool:
    """The track moves with the main road (left -> right across the frame).

    Approach traffic enters top-left and leaves bottom-right / into the side street; the cross street
    and the outbound carriageway run right -> left. The net displacement of the whole track decides.
    """
    return bool(tr.xy[-1, 0] - tr.xy[0, 0] > 0)


def _junction_jam(ctx: Context, grid: np.ndarray) -> np.ndarray:
    """Per grid step: the junction interior is packed with crawling vehicles (spillback of the main road)."""
    interior = polygon("junction").mask() & ~ctx.crosswalk_mask & ~polygon("stop_zone").mask()
    n = np.zeros(len(grid))
    still = np.zeros(len(grid))
    still_main = np.zeros(len(grid))
    for tr in ctx.kind("vehicle"):
        inside = ctx.sample(interior, tr.xy) & ~tr.at_border
        if not inside.any():
            continue
        body = np.maximum(tr.box[:, 3] - tr.box[:, 1], 10.0)
        crawl = tr.speed(1.0) / body < CONGESTION_BODY_SPEED
        main = _with_main_road(tr)
        gi = np.clip(np.searchsorted(grid, tr.t), 0, len(grid) - 1)
        for k in np.unique(gi[inside]):
            m = inside & (gi == k)
            n[k] += 1
            c = crawl[m].mean() >= 0.5
            still[k] += c
            still_main[k] += c and main
    return (still >= JUNCTION_JAM_MIN_VEHICLES) & (still >= JUNCTION_JAM_STILL_FRAC * np.maximum(n, 1)) & \
        (still_main >= JUNCTION_JAM_MIN_MAIN)


def load_lanes() -> np.ndarray:
    """Lane-divider angles of the approach (sorted), for lane indexing."""
    from ..scene import load_layout
    return np.sort([ln["angle"] for ln in load_layout()["solid_lines"]["lines"]])
