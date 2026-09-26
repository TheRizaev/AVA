"""One decoding pass over a video that gathers everything the rules need.

While the detector/tracker runs, every analysed frame is also used to

* re-register the camera to the reference layout every REGISTER_EVERY_SEC
  (the tripod drifts ~10 px during the first half-minute of a clip),
* sample the traffic-signal lamps,
* keep a sparse set of low-resolution frames for scene-level checks
  (static obstacles, smoke) that do not need every frame.

The result is a VideoAnalysis that can be cached to disk (.npz) so rule
development never re-runs the detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detection import load_model, track_video
from .hazards import HAZARD_EVERY_SEC, HazardDetector
from .scene import Registrar, warp_points
from .signals import LampSampler, SignalTimeline, classify
from .tracks import Track, per_track
from .video import VideoMeta, probe

REGISTER_EVERY_SEC = 5.0
THUMB_EVERY_SEC = 1.0
THUMB_SIZE = (640, 360)


@dataclass
class VideoAnalysis:
    meta: VideoMeta
    tracks: np.ndarray          # (N, 9) frame-pixel track table, see detection.COLUMNS
    reg_t: np.ndarray           # (K,) registration times
    reg_H: np.ndarray           # (K, 3, 3) frame -> reference homographies
    reg_ok: np.ndarray          # (K,) whether that keyframe's own fit succeeded
    lamp_t: np.ndarray          # (M,) lamp sample times
    lamp_raw: np.ndarray        # (M, 5) raw lamp activations
    thumb_t: np.ndarray         # (P,) thumbnail times
    thumbs: np.ndarray          # (P, h, w, 3) uint8 thumbnails
    hazards: np.ndarray         # (K, 7) open-vocabulary hazard detections, see hazards.COLUMNS

    @property
    def scene_recognised(self) -> bool:
        """True when the video is from the calibrated camera (most keyframes registered)."""
        return bool(len(self.reg_ok)) and float(np.mean(self.reg_ok)) >= 0.5

    # -- geometry ---------------------------------------------------------------
    def H_index(self, t: np.ndarray) -> np.ndarray:
        """Index of the registration valid at each time (latest keyframe at or before t)."""
        return np.clip(np.searchsorted(self.reg_t, t, side="right") - 1, 0, len(self.reg_t) - 1)

    def to_ref(self, xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(xy)
        idx = self.H_index(np.asarray(t))
        for k in np.unique(idx):
            m = idx == k
            out[m] = warp_points(self.reg_H[k], xy[m])
        return out

    def track_list(self, min_len: int = 3) -> list[Track]:
        return list(per_track(self.tracks, self.to_ref, min_len=min_len))

    def signal_timeline(self) -> SignalTimeline:
        return classify(self.lamp_t, self.lamp_raw, fps=config.SAMPLE_FPS)

    # -- persistence -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        m = self.meta
        np.savez_compressed(path, tracks=self.tracks, reg_t=self.reg_t, reg_H=self.reg_H, reg_ok=self.reg_ok,
                            lamp_t=self.lamp_t, lamp_raw=self.lamp_raw, thumb_t=self.thumb_t,
                            thumbs=self.thumbs, hazards=self.hazards,
                            meta=np.array([m.path, m.width, m.height, m.fps, m.n_frames, m.duration], dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "VideoAnalysis":
        d = np.load(path, allow_pickle=True)
        p, w, h, fps, n, dur = d["meta"]
        hazards = d["hazards"] if "hazards" in d else np.empty((0, 7))
        reg_ok = d["reg_ok"] if "reg_ok" in d else np.ones(len(d["reg_t"]), bool)
        return cls(VideoMeta(str(p), int(w), int(h), float(fps), int(n), float(dur)), d["tracks"], d["reg_t"],
                   d["reg_H"], reg_ok, d["lamp_t"], d["lamp_raw"], d["thumb_t"], d["thumbs"], hazards)


class _FrameHooks:
    """Per-frame side work done during the detection pass."""

    def __init__(self, hazard_detector: HazardDetector | None = None) -> None:
        self.hazard_detector = hazard_detector
        self.hazards: list[np.ndarray] = []
        self._next_hazard = 0.0
        self.lamps = LampSampler()
        self.registrar = Registrar(REGISTER_EVERY_SEC)
        self.reg_t: list[float] = []
        self.reg_H: list[np.ndarray] = []
        self.reg_ok: list[bool] = []
        self.lamp_t: list[float] = []
        self.lamp_raw: list[np.ndarray] = []
        self.thumb_t: list[float] = []
        self.thumbs: list[np.ndarray] = []
        self._next_thumb = 0.0

    def __call__(self, idx: int, t: float, img: np.ndarray, _tracks: np.ndarray) -> None:
        if self.registrar.update(t, img):
            self.reg_t.append(t)
            self.reg_H.append(self.registrar.H.copy())
            self.reg_ok.append(self.registrar.last_ok)
        self.lamp_t.append(t)
        self.lamp_raw.append(self.lamps.sample(img, self.registrar.H))
        if self.hazard_detector is not None and t >= self._next_hazard:
            self.hazards.append(self.hazard_detector(t, img))
            self._next_hazard = t + HAZARD_EVERY_SEC
        if t >= self._next_thumb:
            self.thumb_t.append(t)
            self.thumbs.append(cv2.resize(img, THUMB_SIZE, interpolation=cv2.INTER_AREA))
            self._next_thumb = t + THUMB_EVERY_SEC


def analyze(path: str, model=None, hazard_detector: HazardDetector | None = None,
            sample_fps: float = config.SAMPLE_FPS, det_imgsz: int = config.DET_IMGSZ, progress=None) -> VideoAnalysis:
    """Run the analysis pass. ``progress(fraction)`` is called as frames are processed."""
    meta = probe(path)
    model = model or load_model()
    hooks = _FrameHooks(hazard_detector)

    def on_frame(idx, t, img, tracks):
        hooks(idx, t, img, tracks)
        if progress is not None and meta.duration > 0:
            progress(min(1.0, t / meta.duration))

    table = track_video(path, model, sample_fps=sample_fps, imgsz=det_imgsz, on_frame=on_frame)
    return VideoAnalysis(
        meta, table,
        np.array(hooks.reg_t), np.array(hooks.reg_H).reshape(-1, 3, 3), np.array(hooks.reg_ok, bool),
        np.array(hooks.lamp_t), np.array(hooks.lamp_raw).reshape(-1, 5),
        np.array(hooks.thumb_t), np.array(hooks.thumbs).reshape(-1, THUMB_SIZE[1], THUMB_SIZE[0], 3),
        np.concatenate(hooks.hazards) if hooks.hazards else np.empty((0, 7)),
    )
