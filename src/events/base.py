"""Shared types and helpers for the event rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

from .. import config
from ..analysis import VideoAnalysis
from ..scene import load_layout, polygon, union_mask
from ..signals import SignalTimeline
from ..tracks import Track

CARRIAGEWAY = ("approach", "outbound", "junction")
NOT_CARRIAGEWAY = ("median", "island_round", "island_tri1", "island_tri2", "island_3", "car_park_entrance")
CROSSWALKS = ("cw1", "cw2", "cw3")


@dataclass
class Event:
    start: float
    end: float
    label: str
    score: float = 1.0
    info: dict = field(default_factory=dict)

    def as_list(self) -> list:
        return [round(float(self.start), 2), round(float(self.end), 2), self.label]


@dataclass
class Context:
    """Everything a rule may look at for one video (all geometry in reference pixels)."""
    analysis: VideoAnalysis
    tracks: list[Track]
    signal: SignalTimeline

    @property
    def duration(self) -> float:
        return self.analysis.meta.duration

    @cached_property
    def by_kind(self) -> dict[str, list[Track]]:
        out: dict[str, list[Track]] = {}
        for tr in self.tracks:
            out.setdefault(tr.kind, []).append(tr)
        return out

    def kind(self, name: str) -> list[Track]:
        return self.by_kind.get(name, [])

    @cached_property
    def carriageway(self) -> np.ndarray:
        """Drivable road surface: carriageways minus median and islands (bool HxW)."""
        return union_mask(list(CARRIAGEWAY)) & ~union_mask(list(NOT_CARRIAGEWAY))

    @cached_property
    def crosswalk_mask(self) -> np.ndarray:
        return union_mask(list(CROSSWALKS))

    @cached_property
    def dist_to_crosswalk(self) -> np.ndarray:
        """Pixel distance from every pixel to the nearest zebra-crossing pixel (0 on a crossing)."""
        return cv2.distanceTransform((~self.crosswalk_mask).astype(np.uint8), cv2.DIST_L2, 5)

    @cached_property
    def dist_to_sidewalk(self) -> np.ndarray:
        """Pixel distance from each carriageway pixel to the nearest non-carriageway pixel."""
        return cv2.distanceTransform(self.carriageway.astype(np.uint8), cv2.DIST_L2, 5)

    @cached_property
    def stop_line(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = (np.array(p, float) for p in load_layout()["lines"]["stop_line"])
        return a, b

    def sample(self, mask: np.ndarray, xy: np.ndarray):
        """Look up a HxW map at (N,2) reference points (out-of-frame -> 0/False)."""
        xy = np.asarray(xy).reshape(-1, 2)
        h, w = mask.shape[:2]
        x = np.round(xy[:, 0]).astype(int)
        y = np.round(xy[:, 1]).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        out = np.zeros(len(xy), dtype=mask.dtype)
        out[ok] = mask[y[ok], x[ok]]
        return out


def runs_of_true(flags: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [s, e] (inclusive) where flags is True."""
    flags = np.asarray(flags, bool)
    if not flags.any():
        return []
    d = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts, ends))


def fill_short_gaps(t: np.ndarray, flags: np.ndarray, max_gap: float) -> np.ndarray:
    """Set False runs shorter than max_gap seconds (between True runs) to True."""
    flags = np.asarray(flags, bool).copy()
    runs = runs_of_true(flags)
    for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
        if t[s2] - t[e1] <= max_gap:
            flags[e1:s2 + 1] = True
    return flags


def merge_same_class(events: list[Event], gap: float = 0.0) -> list[Event]:
    """Union overlapping (or closer than `gap`) segments of the same label.

    Mirrors the annotation convention: simultaneous events of one class are one segment.
    """
    out: list[Event] = []
    for label in sorted({e.label for e in events}):
        evs = sorted((e for e in events if e.label == label), key=lambda e: e.start)
        cur: Event | None = None
        for e in evs:
            if cur is not None and e.start <= cur.end + gap:
                cur.end = max(cur.end, e.end)
                cur.score = max(cur.score, e.score)
                cur.info.setdefault("merged", []).append(e.info)
            else:
                if cur is not None:
                    out.append(cur)
                cur = Event(e.start, e.end, e.label, e.score, dict(e.info))
        if cur is not None:
            out.append(cur)
    return sorted(out, key=lambda e: (e.start, e.label))


def clip_events(events: list[Event], duration: float, min_len: float = 0.2) -> list[Event]:
    out = []
    for e in events:
        s, en = max(0.0, e.start), min(duration, e.end)
        if en - s >= min_len:
            out.append(Event(s, en, e.label, e.score, e.info))
    return out
