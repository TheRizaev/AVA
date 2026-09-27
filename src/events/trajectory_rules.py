"""Trajectory-shape rules of vehicles.

* wrong_way - on the approach or outbound carriageway, moving against a strongly one-way cell of the
  scene's normal flow field (assets/flow.npz, learned from the sample tracks by
  scripts/build_scene_maps.py) for MIN_WRONG_SEC and MIN_WRONG_DIST px.
* illegal_u_turn - from the approach, around the median tip, into the outbound carriageway.
* solid_line_crossing - a clean lane change across a solid divider in front of the approach stop line.
* illegal_turn - a right turn into the lower-left side street from a lane other than the right-turn lane.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import config
from ..scene import lane_angle, load_layout, polygon
from .base import Context, Event, fill_short_gaps, runs_of_true

FLOW_FILE = config.ROOT / "assets" / "flow.npz"
MIN_SPEED = 40.0          # px/s; slower motion has no reliable heading
MIN_CELL_COUNT = 8.0      # flow samples needed before a cell's direction is trusted
MIN_DOMINANCE = 0.55      # share of the dominant direction bin in a one-way cell
OPPOSED_PROB = 0.03       # this direction is (almost) never seen in the cell
MIN_WRONG_SEC = 2.0       # real wrong-way driving lasts seconds; 1 s is reached by a noisy track (5 fps ablation)
MIN_WRONG_DIST = 100.0    # px travelled against the flow
U_TURN_MIN_X = 1250.0    # the U-turn loop passes beyond the median tip (reference px)
FLOW_APPROACH_DEG = 27.0  # travel direction on the approach / outbound (reference frame)
FLOW_OUTBOUND_DEG = -160.0
HALF_CAR_DEG = 0.75      # half a car width in lane-angle units on the approach
SETTLE_SEC = 1.0
LANE_CENTRE_TOL = 0.7      # deg from a lane-centre peak = driving in that lane
MAX_LANE_CHANGE_SEC = 5.0
IN_NEW_LANE_DEG = 0.4        # box bottom-centre (the front) this far past the divider: front wheels over it
REAR_LAG_PX = 100.0          # the rear wheels follow the front's path: over the line once the vehicle drove this far on
STOPPED_SPEED = 15.0         # px/s; a vehicle that stops first has finished its manoeuvre where it stands
MAX_LANE_CHANGE_EVENT_SEC = 6.0
MIN_LANE_CHANGE_SPEED = 30.0


@lru_cache(maxsize=1)
def _flow():
    d = np.load(FLOW_FILE)
    hist = d["hist"]
    total = hist.sum(-1)
    return hist, total, int(d["cell"]), int(d["n_bins"])


def _bin(angle: np.ndarray, n_bins: int) -> np.ndarray:
    return (((angle + np.pi) / (2 * np.pi)) * n_bins).astype(int) % n_bins


def opposed_mask(xy: np.ndarray, v: np.ndarray) -> np.ndarray:
    """True where the motion v at xy goes against a trusted one-way flow cell."""
    hist, total, cell, n_bins = _flow()
    gx = np.clip((xy[:, 0] // cell).astype(int), 0, hist.shape[1] - 1)
    gy = np.clip((xy[:, 1] // cell).astype(int), 0, hist.shape[0] - 1)
    h = hist[gy, gx]
    tot = total[gy, gx]
    dom = h.max(-1) / np.maximum(tot, 1e-6)
    b = _bin(np.arctan2(v[:, 1], v[:, 0]), n_bins)
    # probability of this heading +-1 bin in the cell
    p = (h[np.arange(len(b)), b] + h[np.arange(len(b)), (b + 1) % n_bins]
         + h[np.arange(len(b)), (b - 1) % n_bins]) / np.maximum(tot, 1e-6)
    dom_angle = (h.argmax(-1) + 0.5) / n_bins * 2 * np.pi - np.pi
    diff = np.abs((np.arctan2(v[:, 1], v[:, 0]) - dom_angle + np.pi) % (2 * np.pi) - np.pi)
    return (tot >= MIN_CELL_COUNT) & (dom >= MIN_DOMINANCE) & (p < OPPOSED_PROB) & (diff > np.radians(120))


def wrong_way(ctx: Context) -> list[Event]:
    lanes = polygon("approach").mask() | polygon("outbound").mask()
    events = []
    for tr in ctx.kind("vehicle"):
        if len(tr) < 8:
            continue
        v = tr.velocity(0.6)
        speed = np.hypot(v[:, 0], v[:, 1])
        xy = tr.smoothed_xy(0.6)
        bad = (speed > MIN_SPEED) & ctx.sample(lanes, xy) & opposed_mask(xy, v)
        bad = fill_short_gaps(tr.t, bad, 0.6)
        for s, e in runs_of_true(bad):
            dist = np.linalg.norm(xy[e] - xy[s])
            if tr.t[e] - tr.t[s] >= MIN_WRONG_SEC and dist >= MIN_WRONG_DIST:
                events.append(Event(float(tr.t[s]), float(tr.t[e]), "wrong_way", 1.0, {"track": tr.tid}))
    return events


def illegal_u_turn(ctx: Context) -> list[Event]:
    """U-turn at the junction: from the approach, around the median tip, into the outbound.

    Found by origin/destination on the layout (robust to box and heading noise):
    the vehicle is on the approach, later on the outbound carriageway, and passes
    the turning area beyond the median tip in between. Start: the heading leaves
    the approach direction (turn begins); end: the heading has settled on the
    outbound direction (turn complete).
    """
    approach, outbound = polygon("approach"), polygon("outbound")
    events = []
    for tr in ctx.kind("vehicle"):
        if len(tr) < 15:
            continue
        xy = tr.smoothed_xy(0.8)
        in_a = np.flatnonzero(approach.contains(xy))
        in_o = np.flatnonzero(outbound.contains(xy))
        if not len(in_a) or not len(in_o):
            continue
        a_last = in_a[-1]
        later_o = in_o[in_o > a_last]
        if len(later_o) < 5:
            continue
        o_first = int(later_o[0])
        loop = xy[a_last:o_first + 1]
        if not len(loop) or loop[:, 0].max() < U_TURN_MIN_X:
            continue            # never went around the median tip: an id switch at the far end
        v = tr.velocity(0.8)
        moving = np.hypot(v[:, 0], v[:, 1]) > MIN_SPEED
        heading = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        h_in = FLOW_APPROACH_DEG
        h_out = FLOW_OUTBOUND_DEG
        turned = moving & (np.abs((heading - h_in + 180) % 360 - 180) > 25)
        settled = moving & (np.abs((heading - h_out + 180) % 360 - 180) < 25)
        # the turn: the last stretch of approach-direction driving before the loop, to the outbound heading
        loop_idx = int(a_last + np.argmax(loop[:, 0]))
        s_idx = loop_idx
        while s_idx > 0 and not (moving[s_idx - 1] and not turned[s_idx - 1]):
            s_idx -= 1
            if tr.t[loop_idx] - tr.t[s_idx] > 8.0:
                break
        e_idx = next((i for i in range(loop_idx, len(tr)) if settled[i]), o_first)
        events.append(Event(float(tr.t[s_idx]), float(tr.t[e_idx]), "illegal_u_turn", 1.0, {"track": tr.tid}))
    return events


def _rear_over(xy: np.ndarray, speed: np.ndarray, k: int) -> int:
    """Index at which the rear wheels are over the line too, the front being over it at sample k.

    The rear wheels run along the front's path, so they reach the front's
    lateral position once the vehicle has driven REAR_LAG_PX further; a
    vehicle that stops before that ends its manoeuvre where it stands.
    """
    driven = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy[k:], axis=0).T))])
    done = (driven >= REAR_LAG_PX) | (speed[k:] < STOPPED_SPEED)
    return k + int(np.argmax(done)) if done.any() else len(xy) - 1


def solid_line_crossing(ctx: Context) -> list[Event]:
    """A clean lane change across one of the solid dividers in front of the approach stop line.

    The lane-angle of the vehicle (scene.lane_angle) must sit near the centre of
    one lane (within LANE_CENTRE_TOL of a lane-centre peak of the sample
    histogram) for SETTLE_SEC, cross the divider between it and the adjacent
    lane inside the divider's solid part, and then sit near the centre of the
    adjacent lane - a slow drift inside a lane or box noise in a queue does not
    count. Start: the vehicle's side reaches the line (centre within
    HALF_CAR_DEG of it); end: fully in the new lane, i.e. the rear wheels are
    over too. The box bottom-centre is the vehicle's front, and a vehicle
    changing lane is angled across it, so its rear crosses later - by the time
    it takes to drive about its own length, which for a slow, steep move in a
    queue is seconds (see _rear_over).
    """
    layout = load_layout()["solid_lines"]
    lines = sorted(layout["lines"], key=lambda ln: ln["angle"])
    centres = np.sort(layout["lane_centres"])
    approach = polygon("approach")
    events = []
    for tr in ctx.kind("vehicle"):
        # a long bus's box bottom-centre is far from its real ground centreline: no lane reading
        if tr.duration < 2 * SETTLE_SEC or tr.cls == config.BUS:
            continue
        xy = tr.smoothed_xy(0.8)
        ang = lane_angle(xy)
        speed = tr.speed(0.8)
        inside = approach.contains(xy)
        lane = np.argmin(np.abs(ang[:, None] - centres[None, :]), axis=1)
        centred = np.abs(ang - centres[lane]) < LANE_CENTRE_TOL
        for ln in lines:
            below = int(np.searchsorted(centres, ln["angle"]) - 1)      # lane on the low-angle side
            if below < 0 or below + 1 >= len(centres):
                continue
            rel = ang - ln["angle"]
            on_segment = inside & (xy[:, 1] >= ln["y_from"]) & (xy[:, 1] <= ln["y_to"] + 5)
            flips = np.flatnonzero((np.sign(rel[:-1]) != np.sign(rel[1:])) & on_segment[:-1] & on_segment[1:])
            for i in flips:
                if speed[i] < MIN_LANE_CHANGE_SPEED:
                    continue
                a, b = (below, below + 1) if rel[i] < 0 else (below + 1, below)
                # half a car from the line, but never more than a car centred in that lane can be
                # (the median lane's centre is only ~0.8 deg from its divider)
                clr = min(HALF_CAR_DEG, abs(centres[a] - ln["angle"]) - LANE_CENTRE_TOL / 2)
                # in the old lane AND clear of the line (a box already straddling the line is no evidence)
                pre = np.flatnonzero(centred[:i + 1] & (lane[:i + 1] == a) & (np.abs(rel[:i + 1]) >= clr))
                post = np.flatnonzero(centred[i + 1:] & (lane[i + 1:] == b)) + i + 1
                if not len(pre) or not len(post):
                    continue
                p0, p1 = int(pre[-1]), int(post[0])
                if tr.t[p1] - tr.t[p0] > MAX_LANE_CHANGE_SEC:
                    continue
                settled_pre = (tr.t >= tr.t[p0] - SETTLE_SEC) & (tr.t <= tr.t[p0])
                settled_post = (tr.t >= tr.t[p1]) & (tr.t <= tr.t[p1] + SETTLE_SEC)
                if (lane[settled_pre] == a).mean() < 0.9 or (lane[settled_post] == b).mean() < 0.9:
                    continue
                s_i = next((k for k in range(p0, i + 1) if abs(rel[k]) < clr), i)   # the side reaches the line
                past = rel * (b - a)                         # distance past the line towards the new lane
                k_w = next((k for k in range(i + 1, len(tr)) if past[k] >= IN_NEW_LANE_DEG), None)
                e_i = p1 if k_w is None else _rear_over(xy, speed, k_w)
                e_i = min(e_i, int(np.searchsorted(tr.t, tr.t[s_i] + MAX_LANE_CHANGE_EVENT_SEC)))
                e_i = min(e_i, len(tr) - 1)
                events.append(Event(float(tr.t[s_i]), float(tr.t[e_i]), "solid_line_crossing", 1.0,
                                    {"track": tr.tid, "line": ln["angle"]}))
    return events


RIGHT_TURN_LANE_DEG = 32.3     # the rightmost approach lane (beyond this divider) is the right-turn lane
LANE_MARGIN_DEG = 0.5
SIDE_STREET_END = (300.0, 800.0)   # a right turn ends in the lower-left side street (x <, y >)
TURN_DONE_DEG = 125.0          # heading this far from the approach direction = heading down the side street


def illegal_turn(ctx: Context) -> list[Event]:
    """Right turn into the lower-left side street from a lane other than the right-turn lane.

    On the sample videos 19 of 23 right turns start from the rightmost approach
    lane (separated by a solid divider); a right turn from any other lane cuts
    across it. Start: the heading leaves the approach direction; end: the
    vehicle has driven through the side street's crossing (cw3) into the side
    street (turn complete). The exit heading differs by path (~153 deg through
    the slip lane, ~180 deg round island_3 along the frame bottom), so the
    heading alone ends the junction-centre turns seconds early; the heading
    criterion is only the fallback for a track lost before it leaves cw3.
    """
    approach = polygon("approach")
    events = []
    for tr in ctx.kind("vehicle"):
        if len(tr) < 15:
            continue
        xy = tr.smoothed_xy(0.8)
        reached_side_street = (xy[:, 0] < SIDE_STREET_END[0]) & (xy[:, 1] > SIDE_STREET_END[1])
        on_cw3 = polygon("cw3").contains(xy)
        heading_w = np.cos(np.arctan2(*tr.velocity(0.8)[:, ::-1].T)) < -0.5    # moving leftwards
        if not (reached_side_street.any() or (on_cw3 & heading_w).sum() >= 3):
            continue
        before_line = approach.contains(xy) & (xy[:, 1] > 420) & (xy[:, 1] < 515)
        if before_line.sum() < 3:
            continue
        if np.median(lane_angle(xy[before_line])) >= RIGHT_TURN_LANE_DEG - LANE_MARGIN_DEG:
            continue
        v = tr.velocity(0.8)
        moving = np.hypot(v[:, 0], v[:, 1]) > MIN_SPEED
        heading = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        dev = np.abs((heading - FLOW_APPROACH_DEG + 180) % 360 - 180)
        last_on_line = int(np.flatnonzero(before_line)[-1])
        start = next((i for i in range(last_on_line, len(tr)) if moving[i] and dev[i] > 20), None)
        if start is None:
            continue
        end = next((i for i in range(start, len(tr)) if moving[i] and dev[i] > TURN_DONE_DEG), len(tr) - 1)
        cw3_run = np.flatnonzero(on_cw3[start:]) + start
        if len(cw3_run):     # first sample past the crossing (or the track's last one if it is lost on it)
            end = next((i for i in range(int(cw3_run[0]), len(tr)) if not on_cw3[i]), len(tr) - 1)
        events.append(Event(float(tr.t[start]), float(tr.t[end]), "illegal_turn", 1.0, {"track": tr.tid}))
    return events
