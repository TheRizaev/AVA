"""Open-vocabulary hazard detector (road obstacles, animals, fire, smoke, crashed cars).

YOLOE with the text prompts below baked into the weights (no text encoder is
needed at run time, see scripts/make_hazard_weights.py). It runs on a sparse
subset of the analysis frames - obstacles and smoke persist for many seconds.
"""
from __future__ import annotations

import numpy as np

from . import config
from .detection import device, gpu

HAZARD_WEIGHTS = config.WEIGHTS_DIR / "yoloe26l_hazards.pt"
HAZARD_CLASSES = ("cardboard box", "tire", "traffic cone", "fallen tree branch", "debris", "sack",
                  "dog", "cat", "cow", "horse", "fire", "smoke", "crashed car")
OBSTACLE_CLASSES = tuple(range(10))
FIRE_CLASSES = (10, 11)
CRASH_CLASSES = (12,)
HAZARD_EVERY_SEC = 1.0
HAZARD_CONF = 0.25
COLUMNS = ("t", "x1", "y1", "x2", "y2", "conf", "cls")


class HazardDetector:
    def __init__(self) -> None:
        from ultralytics import YOLOE
        self.model = YOLOE(str(HAZARD_WEIGHTS))
        self.model.to(device())

    def __call__(self, t: float, frame_bgr: np.ndarray) -> np.ndarray:
        """(K, 7) rows t, x1, y1, x2, y2, conf, cls for one analysis frame."""
        r = self.model.predict(frame_bgr, imgsz=config.DET_IMGSZ, conf=HAZARD_CONF, verbose=False,
                               device=device(), quantize=16 if gpu() else None)[0]
        b = r.boxes.cpu()
        if not len(b):
            return np.empty((0, len(COLUMNS)))
        out = np.empty((len(b), len(COLUMNS)))
        out[:, 0] = t
        out[:, 1:5] = b.xyxy.numpy()
        out[:, 5] = b.conf.numpy()
        out[:, 6] = b.cls.numpy()
        return out
