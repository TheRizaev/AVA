"""
solution.py — the interface the organizers' harness (run_submission.py) imports.

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/: a YOLO + ByteTrack pass over the video,
per-video registration to a fixed scene layout, traffic-signal reading, and
rule-based event detectors on the resulting trajectories (see README.md).
"""
from __future__ import annotations

import numpy as np

from src import config
from src.pipeline import detect_events as _detect_events
from src.risk import RiskEstimator as _RiskEstimator

# Official class ids (14). Classes we never predict may be removed; never add.
CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection for one .mp4."""
    return _detect_events(video_path, config.ENABLED_CLASSES)


class RiskEstimator(_RiskEstimator):
    """Part B — causal accident anticipation: P(accident starts within 5 s)."""

    def reset(self, meta: dict) -> None:
        super().reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return super().step(frame, t_sec)
