"""Track table -> per-object trajectories in reference coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import config
from .scene import warp_points

BORDER_PX = 3.0

KIND_BY_CLS = {config.PERSON: "person", config.BICYCLE: "bicycle",
               **{c: "vehicle" for c in config.VEHICLES},
               **{c: "animal" for c in config.ANIMALS}}


@dataclass
class Track:
    tid: int
    cls: int                 # majority COCO class
    kind: str                # vehicle | person | bicycle | animal
    frame: np.ndarray        # (N,) frame indices
    t: np.ndarray            # (N,) seconds
    box: np.ndarray          # (N, 4) x1 y1 x2 y2 in analysis-frame pixels
    xy: np.ndarray           # (N, 2) ground point (bottom-centre) in reference pixels
    score: np.ndarray        # (N,)
    corners: np.ndarray      # (N, 4, 2) box corners x1y1, x2y1, x2y2, x1y2 in reference pixels

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def at_border(self) -> np.ndarray:
        """(N,) the box touches the frame edge, so its bottom-centre is not the real ground point."""
        w, h = config.ANALYSIS_SIZE
        b = self.box
        return (b[:, 0] <= BORDER_PX) | (b[:, 1] <= BORDER_PX) | (b[:, 2] >= w - BORDER_PX) | (b[:, 3] >= h - BORDER_PX)

    @property
    def size(self) -> np.ndarray:
        """(N,) box diagonal in pixels, a proxy for object scale."""
        return np.hypot(self.box[:, 2] - self.box[:, 0], self.box[:, 3] - self.box[:, 1])

    def footprint(self, depth: float = 0.35, nx: int = 5, ny: int = 3) -> np.ndarray:
        """(N, nx*ny, 2) points covering the lower `depth` of the box: its road contact area."""
        tl, tr_, br, bl = (self.corners[:, k] for k in range(4))
        pts = []
        for fy in np.linspace(1.0 - depth, 1.0, ny):
            left = tl + (bl - tl) * fy
            right = tr_ + (br - tr_) * fy
            for fx in np.linspace(0.1, 0.9, nx):
                pts.append(left + (right - left) * fx)
        return np.stack(pts, axis=1)

    def smoothed_xy(self, window_sec: float = 0.5) -> np.ndarray:
        if len(self) < 3:
            return self.xy
        dt = np.median(np.diff(self.t)) if len(self.t) > 1 else 0.1
        k = max(1, int(round(window_sec / max(dt, 1e-3))))
        return uniform_filter1d(self.xy, size=k, axis=0, mode="nearest")

    def velocity(self, window_sec: float = 0.5) -> np.ndarray:
        """(N, 2) px/s in reference coordinates, from smoothed positions."""
        if len(self) < 2:
            return np.zeros_like(self.xy)
        xy = self.smoothed_xy(window_sec)
        return np.gradient(xy, self.t, axis=0)

    def speed(self, window_sec: float = 0.5) -> np.ndarray:
        v = self.velocity(window_sec)
        return np.hypot(v[:, 0], v[:, 1])


def load_track_table(path: str | Path) -> np.ndarray:
    return np.load(path)["tracks"]


MAX_JUMP_PX = 60.0       # a step longer than this and JUMP_FACTOR x the median step is an ID switch
JUMP_FACTOR = 4.0
SPLIT_ID_OFFSET = 1_000_000


def _split_at_jumps(rows: np.ndarray) -> list[np.ndarray]:
    """Cut a track where its box jumps (ByteTrack occasionally hands an id to another object)."""
    if len(rows) < 3:
        return [rows]
    cx = (rows[:, 3] + rows[:, 5]) / 2
    cy = rows[:, 6]
    step = np.hypot(np.diff(cx), np.diff(cy))
    typical = max(float(np.median(step)), 2.0)
    cuts = np.flatnonzero((step > MAX_JUMP_PX) & (step > JUMP_FACTOR * typical)) + 1
    return np.split(rows, cuts) if len(cuts) else [rows]


def per_track(table: np.ndarray, to_ref, min_len: int = 3) -> Iterator[Track]:
    """Split the flat (frame, t, id, x1, y1, x2, y2, score, cls) table into Tracks.

    ``to_ref(xy, t)`` maps frame pixels at time t into reference pixels; a 3x3
    homography is accepted too (a camera that did not move).
    """
    if isinstance(to_ref, np.ndarray):
        H = to_ref
        to_ref = lambda xy, _t: warp_points(H, xy)  # noqa: E731
    if len(table) == 0:
        return
    order = np.lexsort((table[:, 0], table[:, 2]))
    table = table[order]
    ids, starts = np.unique(table[:, 2], return_index=True)
    bounds = list(starts[1:]) + [len(table)]
    for tid, s, e in zip(ids, starts, bounds):
        for k, rows in enumerate(_split_at_jumps(table[s:e])):
            if len(rows) >= min_len:
                yield _make_track(int(tid) + k * SPLIT_ID_OFFSET, rows, to_ref)


def _make_track(tid: int, rows: np.ndarray, to_ref) -> Track:
    cls = int(np.bincount(rows[:, 8].astype(int)).argmax())
    box = rows[:, 3:7]
    t = rows[:, 1]
    ground = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], axis=1)
    corners = np.stack([box[:, [0, 1]], box[:, [2, 1]], box[:, [2, 3]], box[:, [0, 3]]], axis=1)
    corners_ref = to_ref(corners.reshape(-1, 2), np.repeat(t, 4)).reshape(-1, 4, 2)
    return Track(tid, cls, KIND_BY_CLS.get(cls, "other"), rows[:, 0].astype(int), t,
                 box, to_ref(ground, t), rows[:, 7], corners_ref)
