"""Scene hazards: road_obstacle and fire_smoke (from the sparse open-vocabulary detector).

Both classes are rare, and predicting a class that is absent from the test set
costs a whole class of macro-F1, so the rules are strict: a hazard must be on
the carriageway, not explained by a tracked vehicle / person, and seen in
several consecutive hazard samples at the same place.
"""
from __future__ import annotations

import numpy as np

from ..hazards import FIRE_CLASSES, HAZARD_EVERY_SEC, OBSTACLE_CLASSES
from .base import Context, Event

PERSIST_SEC = 4.0            # an obstacle / smoke is seen at the same place for this long
MIN_OBSTACLE_HITS = 1 + round(PERSIST_SEC / HAZARD_EVERY_SEC)   # consecutive hazard samples
MIN_FIRE_HITS = MIN_OBSTACLE_HITS
FIRE_CONF = 0.35
OBSTACLE_CONF = 0.30
SAME_PLACE_PX = 40.0
COVERED_BY_TRACK = 0.3       # fraction of the hazard box inside a tracked road user's box


def _track_boxes_at(ctx: Context, t: float) -> np.ndarray:
    tab = ctx.analysis.tracks
    if not len(tab):
        return np.empty((0, 4))
    near = np.abs(tab[:, 1] - t) <= 0.06   # rows of the analysed frame nearest to t
    return tab[near, 3:7]


def _covered(box: np.ndarray, others: np.ndarray) -> bool:
    if not len(others):
        return False
    ix = np.clip(np.minimum(others[:, 2], box[2]) - np.maximum(others[:, 0], box[0]), 0, None)
    iy = np.clip(np.minimum(others[:, 3], box[3]) - np.maximum(others[:, 1], box[1]), 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    return bool((ix * iy / area).max() > COVERED_BY_TRACK)


def _candidates(ctx: Context, classes: tuple[int, ...], conf: float, on_road_only: bool) -> list[tuple[float, np.ndarray]]:
    """(t, ground point in reference px) of hazard detections passing the filters."""
    hz = ctx.analysis.hazards
    out = []
    for t, x1, y1, x2, y2, c, k in hz:
        if int(k) not in classes or c < conf:
            continue
        box = np.array([x1, y1, x2, y2])
        if _covered(box, _track_boxes_at(ctx, t)):
            continue
        ground = ctx.analysis.to_ref(np.array([[(x1 + x2) / 2, y2]]), np.array([t]))[0]
        if on_road_only and ctx.sample(ctx.dist_to_sidewalk, ground[None])[0] < 8:
            continue
        out.append((float(t), ground))
    return out


def _persistent(cands: list[tuple[float, np.ndarray]], min_hits: int, label: str) -> list[Event]:
    """Group detections at the same place in consecutive samples into events."""
    groups: list[dict] = []
    for t, xy in sorted(cands, key=lambda c: c[0]):
        for g in groups:
            if np.linalg.norm(g["xy"] - xy) < SAME_PLACE_PX and t - g["t1"] <= max(HAZARD_EVERY_SEC * 1.6, 2.2):
                g["hits"] += t > g["t1"]      # hits = distinct hazard samples, not boxes
                g["t1"] = t
                break
        else:
            groups.append({"t0": t, "t1": t, "xy": xy, "hits": 1})
    return [Event(g["t0"], g["t1"] + HAZARD_EVERY_SEC / 2, label, 1.0, {"hits": g["hits"], "xy": g["xy"].round().tolist()})
            for g in groups if g["hits"] >= min_hits]


def road_obstacle(ctx: Context) -> list[Event]:
    events = _persistent(_candidates(ctx, OBSTACLE_CLASSES, OBSTACLE_CONF, True), MIN_OBSTACLE_HITS, "road_obstacle")
    # animals tracked by the main detector on the carriageway
    for tr in ctx.kind("animal"):
        on = ctx.sample(ctx.dist_to_sidewalk, tr.xy) >= 8
        if on.sum() >= 20 and tr.duration >= 2.0:
            idx = np.flatnonzero(on)
            events.append(Event(float(tr.t[idx[0]]), float(tr.t[idx[-1]]), "road_obstacle", 1.0, {"track": tr.tid}))
    return events


def fire_smoke(ctx: Context) -> list[Event]:
    return _persistent(_candidates(ctx, FIRE_CLASSES, FIRE_CONF, False), MIN_FIRE_HITS, "fire_smoke")
