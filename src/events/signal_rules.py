"""Rules tied to the main-road signal: red_light and stop_line (approach stop line)."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..scene import polygon, side_of_line
from ..signals import RED
from .base import Context, Event

RED_GRACE = 0.5        # s after red onset in which we do not call a violation
# s before green in which a crossing is an early start on red+yellow, not red-light running: the
# queue front moves off with the red+yellow (3 of 14 green phases on the samples, 0.26-0.40 s early)
# and the dev annotators rejected a rider who went 0.75 s early.
RED_GRACE_END = 1.0
MIN_CROSS_SPEED = 25.0  # px/s at the crossing: a creeping vehicle is a stop_line case, not red_light
STOP_SPEED = 12.0       # px/s below which a vehicle counts as stopped
PAST_LINE = 8.0         # px beyond the stop line before "past the line" counts
MIN_STOP_SEC = 1.5
STOP_AREA_DILATE = 12     # px of tolerance around the stop zone + cw1


def _approach_sign(ctx: Context) -> float:
    a, b = ctx.stop_line
    return float(np.sign(side_of_line(np.array([[500.0, 380.0]]), a, b)[0]))


def _signed_dist(ctx: Context, xy: np.ndarray) -> np.ndarray:
    """Distance to the stop line, positive on the junction side (past the line)."""
    a, b = ctx.stop_line
    n = np.linalg.norm(b - a)
    return -_approach_sign(ctx) * side_of_line(xy, a, b) / n


def _along(ctx: Context, xy: np.ndarray) -> np.ndarray:
    a, b = ctx.stop_line
    d = b - a
    return (np.asarray(xy).reshape(-1, 2) - a) @ d / (d @ d)


def _red_bounds(ctx: Context, t: float) -> tuple[float, float] | None:
    """(red_start, red_end) of the red phase containing t, or None."""
    sig = ctx.signal
    if sig.at(t)[0] != RED:
        return None
    i = int(np.searchsorted(sig.t, t, side="right") - 1)
    s = i
    while s > 0 and sig.state[s - 1] == RED:
        s -= 1
    e = i
    while e + 1 < len(sig.state) and sig.state[e + 1] == RED:
        e += 1
    # a red phase touching the clip boundary has an unknown true edge: treat it as far away
    start = sig.t[s] if s > 0 else -1e9
    end = sig.t[e] if e + 1 < len(sig.state) else 1e9
    return float(start), float(end)


@lru_cache(maxsize=1)
def _stop_area() -> np.ndarray:
    """Between the stop line and the far edge of cw1 (dilated): past the line but not yet in the junction."""
    m = (polygon("stop_zone").mask() | polygon("cw1").mask()).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * STOP_AREA_DILATE + 1, 2 * STOP_AREA_DILATE + 1))
    return cv2.dilate(m, k).astype(bool)


def _entered_junction(ctx: Context, xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Past the stop line and outside the stop area: the vehicle is inside the intersection."""
    return (d > PAST_LINE) & ~ctx.sample(_stop_area(), xy)


def _exit_time(ctx: Context, tr, i0: int) -> float:
    """First time after sample i0 at which the vehicle leaves the junction (or the track ends)."""
    junction = polygon("junction")
    inside = junction.contains(tr.xy[i0:])
    out = np.flatnonzero(~inside)
    if len(out):
        return float(tr.t[i0 + out[0]])
    return float(tr.t[-1])


def red_light(ctx: Context) -> list[Event]:
    """An approach vehicle's front crosses the stop line on red and it drives on into the junction.

    A vehicle that crossed on green and later clears the junction on red is not
    counted (the annotators rejected those as clearing a gridlocked junction).
    """
    events = []
    approach = polygon("approach")
    for tr in ctx.kind("vehicle"):
        d = _signed_dist(ctx, tr.xy)
        crossed = np.flatnonzero((d[:-1] <= 0) & (d[1:] > 0))
        if not len(crossed):
            continue
        i = int(crossed[0])
        # interpolate the crossing time
        f = -d[i] / (d[i + 1] - d[i])
        tc = float(tr.t[i] + f * (tr.t[i + 1] - tr.t[i]))
        u = _along(ctx, tr.xy[i + 1])[0]
        came_from_approach = approach.contains(tr.xy[max(0, i - 10):i + 1]).any()
        if not (-0.05 <= u <= 1.05 and came_from_approach):
            continue
        bounds = _red_bounds(ctx, tc)
        if bounds is None or tc - bounds[0] < RED_GRACE or bounds[1] - tc < RED_GRACE_END:
            continue
        speed = tr.speed()
        if speed[i:i + 2].mean() < MIN_CROSS_SPEED:
            continue
        # must actually proceed into the intersection (beyond cw1), not stop on the crossing
        if not _entered_junction(ctx, tr.xy[i + 1:], d[i + 1:]).any():
            continue
        events.append(Event(tc, _exit_time(ctx, tr, i + 1), "red_light", 1.0,
                            {"track": tr.tid, "red_since": round(tc - bounds[0], 1)}))
    return events


def stop_line(ctx: Context) -> list[Event]:
    events = []
    for tr in ctx.kind("vehicle"):
        d = _signed_dist(ctx, tr.xy)
        speed = tr.speed()
        on_red = ctx.signal.at(tr.t) == RED
        past = (d > PAST_LINE) & ctx.sample(_stop_area(), tr.xy)
        stopped = past & (speed < STOP_SPEED) & on_red & ~tr.at_border
        idx = np.flatnonzero(stopped)
        if not len(idx):
            continue
        t_stop = float(tr.t[idx[0]])
        bounds = _red_bounds(ctx, t_stop)
        if bounds is None:
            continue
        # it must have stayed there: still stopped past the line for a while
        if tr.t[idx[-1]] - t_stop < MIN_STOP_SEC:
            continue
        # it must not have driven on into the junction during the red phase
        later = (tr.t > t_stop) & (tr.t < bounds[1])
        if _entered_junction(ctx, tr.xy[later], d[later]).any():
            continue
        # a vehicle already standing past the line when the signal turns red counts from the red onset
        start = max(t_stop, bounds[0]) if bounds[0] > -1e8 else t_stop
        end = min(bounds[1], ctx.duration)
        if end - start > 0.5:
            events.append(Event(start, end, "stop_line", 1.0, {"track": tr.tid}))
    return events
