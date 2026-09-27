"""Part A entry point: video path -> list of [start_sec, end_sec, label]."""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from . import config
from .analysis import analyze
from .detection import load_model
from .hazards import HazardDetector
from .events import build_context, detect

_MODEL = None
_HAZARDS = None


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shared_model():
    """The detector is loaded once per process and shared by Part A and Part B."""
    global _MODEL
    if _MODEL is None:
        seed_everything()
        _MODEL = load_model()
    return _MODEL


def shared_hazard_detector() -> HazardDetector:
    global _HAZARDS
    if _HAZARDS is None:
        _HAZARDS = HazardDetector()
    return _HAZARDS


def detect_events(video_path: str, enabled: tuple[str, ...] | None = config.ENABLED_CLASSES) -> list[list]:
    t0 = time.perf_counter()
    analysis = analyze(video_path, shared_model(), shared_hazard_detector())
    ctx = build_context(analysis)
    events = detect(ctx, enabled)
    if config.VERBOSE:
        print(f"  part A: {len(events)} events in {time.perf_counter() - t0:.0f}s")
    return [e.as_list() for e in events]
