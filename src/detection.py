"""Road-user detection (YOLO) and multi-object tracking (ByteTrack).

Decoding runs in a background thread so the CPU decode of the next frames
overlaps with GPU inference of the current batch. Detections are tracked with
Ultralytics' ByteTrack implementation, fed frame by frame in decode order.

Output is a flat table, one row per (sampled frame, tracked object), in the
analysis resolution (ANALYSIS_SIZE):

    frame, t, track_id, x1, y1, x2, y2, score, cls
"""
from __future__ import annotations

import queue
import threading
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from . import config
from .video import iter_frames

COLUMNS = ("frame", "t", "track_id", "x1", "y1", "x2", "y2", "score", "cls")


def gpu() -> bool:
    """A usable CUDA device (is_available() alone can be True with no device visible)."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def device() -> str:
    return "cuda:0" if gpu() else "cpu"


def load_model(weights: str | Path = config.DETECTOR_WEIGHTS) -> YOLO:
    model = YOLO(str(weights))
    model.to(device())
    return model


def _prefetch(it: Iterable, depth: int) -> Iterator:
    """Run an iterator in a daemon thread, buffering up to ``depth`` items."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()
    error: list[BaseException] = []

    def worker() -> None:
        try:
            for item in it:
                q.put(item)
        except BaseException as exc:  # surfaced in the consumer thread
            error.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not sentinel:
        yield item
    if error:
        raise error[0]


def detect_batch(model: YOLO, frames: list[np.ndarray], conf: float = config.DET_CONF,
                 imgsz: int = config.DET_IMGSZ) -> list[Boxes]:
    results = model.predict(frames, imgsz=imgsz, conf=conf, classes=list(config.DET_CLASSES), device=device(),
                            quantize=16 if gpu() else None, verbose=False)
    return [r.boxes.cpu() for r in results]


def new_tracker(fps: float) -> BYTETracker:
    args = Namespace(**config.TRACKER_ARGS)
    tracker = BYTETracker(args)
    tracker.max_frames_lost = int(config.TRACK_BUFFER_SEC * fps)
    return tracker


def track_video(path: str, model: YOLO, sample_fps: float = config.SAMPLE_FPS,
                imgsz: int = config.DET_IMGSZ, on_frame=None) -> np.ndarray:
    """Detect and track road users in one video; returns an (N, 9) float array.

    ``on_frame(frame_idx, t, bgr, tracks)`` is called for every processed frame
    (used by the visualiser to render without decoding twice).
    """
    frames_it = _prefetch(iter_frames(path, config.ANALYSIS_SIZE, skip="NONREF", min_step=1.0 / sample_fps),
                          depth=config.DET_BATCH * 3)
    tracker = new_tracker(sample_fps)
    rows: list[np.ndarray] = []
    batch: list[tuple[int, float, np.ndarray]] = []

    def flush() -> None:
        boxes = detect_batch(model, [b[2] for b in batch], imgsz=imgsz)
        for (idx, t, img), det in zip(batch, boxes):
            tracks = tracker.update(det, img)
            if len(tracks):
                # tracks: x1, y1, x2, y2, id, score, cls, det_idx
                out = np.empty((len(tracks), len(COLUMNS)), dtype=np.float64)
                out[:, 0] = idx
                out[:, 1] = t
                out[:, 2] = tracks[:, 4]
                out[:, 3:7] = tracks[:, :4]
                out[:, 7] = tracks[:, 5]
                out[:, 8] = tracks[:, 6]
                rows.append(out)
            if on_frame is not None:
                on_frame(idx, t, img, tracks)
        batch.clear()

    for item in frames_it:
        batch.append(item)
        if len(batch) == config.DET_BATCH:
            flush()
    if batch:
        flush()
    return np.concatenate(rows) if rows else np.empty((0, len(COLUMNS)))
