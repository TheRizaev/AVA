"""Conflict rules: accident and near_miss.

accident  - the open-vocabulary "crashed car" prompt of the hazard detector
            firing (>= CRASH_CONF) in at least MIN_CRASH_HITS samples at one
            place, consecutive hits at most CRASH_GAP s apart (on the
            carriageway when the scene is recognised). The event runs from
            CRASH_BEFORE s before to CRASH_AFTER s after the first hit.
            CRASH_CONF stays above the prompt's peak on the ordinary traffic of
            the samples (0.37); re-measure it when the prompt, the hazard
            weights or DET_IMGSZ change.
near_miss - hard braking (>= NEAR_DROP of the speed lost within NEAR_WINDOW s)
            while another road user is in the vehicle's path and close.
            Start = onset of braking, end = when the two are clear of each other.
"""
from __future__ import annotations

import numpy as np

from ..hazards import CRASH_CLASSES
from ..tracks import Track
from .base import Context, Event

NEAR_DROP = 0.5
NEAR_WINDOW = 1.0
NEAR_FROM = 150.0
PATH_AHEAD_SEC = 1.5       # the other road user is within this many seconds of travel ahead
PATH_HALF_WIDTH = 0.8      # ... and within this many box widths of the travel line
CRASH_CONF = 0.45
MIN_CRASH_HITS = 2
CRASH_GAP = 2.5            # s between consecutive hits of one crash
CRASH_BEFORE = 1.5         # s: the contact is typically ~0.5-1.5 s before the prompt first fires
CRASH_AFTER = 1.5          # s: until the involved road users have stopped


def _others_at(ctx: Context, t: float, exclude: int) -> list[tuple[Track, int]]:
    out = []
    for tr in ctx.tracks:
        if tr.tid == exclude or tr.kind not in ("vehicle", "person", "bicycle"):
            continue
        if tr.t[0] - 0.05 <= t <= tr.t[-1] + 0.05:
            out.append((tr, int(np.argmin(np.abs(tr.t - t)))))
    return out


def _speed_drops(tr: Track, v_from: float, frac: float, window: float):
    """Indices (i_start, i_end) where the speed falls by `frac` within `window` seconds."""
    sp = tr.speed(0.3)
    out = []
    i = 0
    while i < len(tr):
        if sp[i] >= v_from:
            j = i + 1
            while j < len(tr) and tr.t[j] - tr.t[i] <= window:
                if sp[j] <= sp[i] * (1 - frac):
                    out.append((i, j))
                    i = j
                    break
                j += 1
        i += 1
    return out


def accident(ctx: Context) -> list[Event]:
    hz = ctx.analysis.hazards
    hz = hz[np.isin(hz[:, 6].astype(int), CRASH_CLASSES) & (hz[:, 5] >= CRASH_CONF)] if len(hz) else hz
    if not len(hz):
        return []
    road = ctx.carriageway if ctx.analysis.scene_recognised else None
    groups: list[dict] = []
    for t, x1, y1, x2, y2, _conf, _cls in hz[np.argsort(hz[:, 0], kind="stable")]:
        c = np.array([(x1 + x2) / 2, y2])
        if road is not None and not ctx.sample(road, ctx.analysis.to_ref(c[None], np.array([t])))[0]:
            continue
        for g in groups:
            if t - g["t1"] <= CRASH_GAP and np.linalg.norm(g["c"] - c) < max(x2 - x1, 100.0):
                g["hits"] += t > g["t1"]      # hits = distinct hazard samples, not boxes
                g["t1"], g["c"] = t, c
                break
        else:
            groups.append({"t0": t, "t1": t, "c": c, "hits": 1})
    return [Event(max(0.0, g["t0"] - CRASH_BEFORE), g["t0"] + CRASH_AFTER, "accident", 1.0,
                  {"crash_prompt_hits": g["hits"], "last_hit": g["t1"]}) for g in groups if g["hits"] >= MIN_CRASH_HITS]


def near_miss(ctx: Context) -> list[Event]:
    events = []
    for tr in ctx.kind("vehicle"):
        if len(tr) < 10:
            continue
        v = tr.velocity(0.3)
        width = np.maximum(tr.box[:, 2] - tr.box[:, 0], 1.0)
        border = tr.at_border
        for i, j in _speed_drops(tr, NEAR_FROM, NEAR_DROP, NEAR_WINDOW):
            if border[i:j + 1].any():
                continue
            speed_i = np.linalg.norm(v[i])
            u = v[i] / max(speed_i, 1e-6)
            conflict = None
            for o, k in _others_at(ctx, tr.t[i], tr.tid):
                rel = o.xy[k] - tr.xy[i]
                ahead = rel @ u
                lateral = abs(rel @ np.array([-u[1], u[0]]))
                if 0 < ahead < PATH_AHEAD_SEC * speed_i and lateral < PATH_HALF_WIDTH * width[i]:
                    conflict = (o, k)
                    break
            if conflict is None:
                continue
            o, _ = conflict
            # clear of each other: the gap grows again after the braking
            later = np.flatnonzero(tr.t > tr.t[j])
            end = float(tr.t[j]) + 1.0
            for m in later:
                ko = int(np.argmin(np.abs(o.t - tr.t[m])))
                if abs(o.t[ko] - tr.t[m]) > 0.2:
                    break
                d0 = np.linalg.norm(o.xy[ko] - tr.xy[m])
                if d0 > 2.5 * width[m]:
                    end = float(tr.t[m])
                    break
            events.append(Event(float(tr.t[i]), end, "near_miss", 1.0, {"track": tr.tid, "with": o.tid}))
    return events
