"""Traffic-signal phase of the main road, read from the lamps visible in frame.

Two heads face the camera (the approach itself only shows us the backs of its
gantry signals):

* ``vehicle`` - 3-lamp head on the median tip. Cycle red -> red+yellow ->
  green -> flashing green -> yellow -> red, ~75 s.
* ``ped`` - pedestrian head for cw3, in phase with the main road (walk while
  the main road has green), used as a fallback when the vehicle head is
  unreadable (washed out by sun, occluded by a bus).

Each lamp's raw colour activation is sampled on a small disk at its registered
position and normalised by the lamp's own off/on levels in the video, so the
same code works at noon and at dusk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .scene import load_layout, warp_points

RED, YELLOW, GREEN, UNKNOWN = 0, 1, 2, -1
STATE_NAMES = {RED: "red", YELLOW: "yellow", GREEN: "green", UNKNOWN: "unknown"}
LAMPS = (("vehicle", "red"), ("vehicle", "yellow"), ("vehicle", "green"), ("ped", "red"), ("ped", "green"))
MIN_CONTRAST = 12.0   # on/off difference (in activation units) below which a lamp is unreadable


def _activation(pixels_bgr: np.ndarray, colour: str) -> float:
    b, g, r = (pixels_bgr[:, i].astype(np.float32) for i in range(3))
    if colour == "red":
        a = r - np.maximum(g, b)
    elif colour == "yellow":
        a = np.minimum(r, g) - b
    else:  # the green LEDs are cyan-green
        a = g - r
    return float(np.mean(a))


class LampSampler:
    """Collects raw lamp activations frame by frame (analysis-frame pixels)."""

    def __init__(self) -> None:
        sig = load_layout()["signals"]
        self.radius = int(sig["lamp_radius"])
        self.ref_points = np.array([sig[head][col] for head, col in LAMPS], dtype=np.float64)
        yy, xx = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        self.offsets = np.stack([xx[xx ** 2 + yy ** 2 <= self.radius ** 2],
                                 yy[xx ** 2 + yy ** 2 <= self.radius ** 2]], axis=1)

    def sample(self, frame_bgr: np.ndarray, H_ref_from_frame: np.ndarray) -> np.ndarray:
        """Raw activation of each lamp in LAMPS order for one frame."""
        pts = warp_points(np.linalg.inv(H_ref_from_frame), self.ref_points)
        h, w = frame_bgr.shape[:2]
        out = np.empty(len(LAMPS), dtype=np.float32)
        for i, ((_, colour), (x, y)) in enumerate(zip(LAMPS, pts)):
            px = np.round(x + self.offsets[:, 0]).astype(int).clip(0, w - 1)
            py = np.round(y + self.offsets[:, 1]).astype(int).clip(0, h - 1)
            out[i] = _activation(frame_bgr[py, px], colour)
        return out


@dataclass
class SignalTimeline:
    t: np.ndarray        # (N,) sample times
    state: np.ndarray    # (N,) RED / YELLOW / GREEN / UNKNOWN (main-road phase)
    activation: np.ndarray  # (N, len(LAMPS)) normalised 0..1

    def at(self, t: float | np.ndarray) -> np.ndarray:
        """State at arbitrary time(s): the latest sample at or before t."""
        idx = np.searchsorted(self.t, np.atleast_1d(t), side="right") - 1
        out = np.full(idx.shape, UNKNOWN)
        ok = idx >= 0
        out[ok] = self.state[idx[ok]]
        return out


MIN_CONTRAST_YELLOW = 5.0  # the yellow lamp is faint in daylight but still separable


def normalise(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-lamp min/max normalisation. Returns (activation, readable-mask per lamp)."""
    lo = np.percentile(raw, 5, axis=0)
    hi = np.percentile(raw, 95, axis=0)
    contrast = hi - lo
    act = np.clip((raw - lo) / np.maximum(contrast, 1e-6), 0, 1)
    need = np.full(len(LAMPS), MIN_CONTRAST)
    need[LAMPS.index(("vehicle", "yellow"))] = MIN_CONTRAST_YELLOW
    return act, contrast >= need


def classify(t: np.ndarray, raw: np.ndarray, fps: float) -> SignalTimeline:
    """Turn raw lamp activations into a cleaned main-road phase timeline.

    Per sample the lit lamp of the vehicle head gives RED (red, or red+yellow),
    YELLOW or GREEN; nothing lit is a gap. Gaps are then resolved with the
    phase sequence: a short dark gap inside green is the flashing green, other
    short gaps are occlusions (a bus passing in front of the head) and inherit
    the previous phase, and a YELLOW island between two GREEN runs is an
    occlusion artefact, not a phase. The pedestrian head, which only walks
    while the main road has green, confirms GREEN in long unreadable stretches.
    """
    if len(t) == 0:
        return SignalTimeline(t, np.array([], int), np.zeros((0, len(LAMPS))))
    act, readable = normalise(raw)
    k = max(1, int(round(0.3 * fps)) | 1)
    act = median_filter(act, size=(k, 1), mode="nearest")
    vr, vy, vg, _, pg = (act[:, i] for i in range(len(LAMPS)))
    state = np.full(len(t), UNKNOWN)
    if readable[0] and readable[2]:
        y_on = (vy > 0.5) if readable[1] else np.zeros(len(t), bool)
        state[vg > 0.5] = GREEN
        state[y_on & (vg <= 0.5)] = YELLOW
        state[vr > 0.5] = RED
    state = _resolve_gaps(t, state)
    state = _drop_yellow_islands(t, state)
    if readable[3] and readable[4]:
        state[(state == UNKNOWN) & (pg > 0.5)] = GREEN
    return SignalTimeline(t, state, act)


def _runs(state: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end_exclusive, value) runs of equal values."""
    out, s = [], 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[s]:
            out.append((s, i, int(state[s])))
            s = i
    return out


def _resolve_gaps(t: np.ndarray, state: np.ndarray, flash_gap: float = 1.2, occlusion_gap: float = 3.0) -> np.ndarray:
    state = state.copy()
    runs = _runs(state)
    for k, (s, e, v) in enumerate(runs):
        if v != UNKNOWN:
            continue
        span = t[e - 1] - t[s]
        prev = runs[k - 1][2] if k > 0 else UNKNOWN
        nxt = runs[k + 1][2] if k + 1 < len(runs) else UNKNOWN
        if prev == GREEN and nxt in (GREEN, YELLOW, UNKNOWN) and span <= flash_gap:
            state[s:e] = GREEN
        elif prev != UNKNOWN and span <= occlusion_gap:
            state[s:e] = prev
        elif prev == UNKNOWN and nxt != UNKNOWN and span <= occlusion_gap:
            state[s:e] = nxt
    return state


def _drop_yellow_islands(t: np.ndarray, state: np.ndarray, max_len: float = 5.0) -> np.ndarray:
    """YELLOW between two GREEN runs is an occlusion artefact; YELLOW after RED is red+yellow."""
    state = state.copy()
    runs = _runs(state)
    for k, (s, e, v) in enumerate(runs):
        if v != YELLOW:
            continue
        prev = runs[k - 1][2] if k > 0 else UNKNOWN
        nxt = runs[k + 1][2] if k + 1 < len(runs) else UNKNOWN
        if prev == GREEN and nxt == GREEN and t[e - 1] - t[s] <= max_len:
            state[s:e] = GREEN
        elif prev == RED:
            state[s:e] = RED
    return state
