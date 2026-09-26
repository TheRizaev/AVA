"""Dev helper: add hazard detections to cached analyses from the 1080p proxies.

Production computes them inside the analysis pass (src/analysis.py); caches
made before that was added get the same detections here without re-running
the whole pass.

    python scripts/add_hazards_to_cache.py --cache work/cache --proxy work/proxy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import VideoAnalysis  # noqa: E402
from src.hazards import HAZARD_EVERY_SEC, HazardDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--proxy", default="work/proxy")
    args = ap.parse_args()
    det = HazardDetector()
    for f in sorted(Path(args.cache).glob("*.npz")):
        va = VideoAnalysis.load(f)
        cap = cv2.VideoCapture(str(Path(args.proxy) / f"{f.stem}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        rows = []
        for t in np.arange(0.0, va.meta.duration, HAZARD_EVERY_SEC):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if ok:
                rows.append(det(float(t), frame))
        va.hazards = np.concatenate(rows) if rows else np.empty((0, 7))
        va.save(f)
        print(f.stem, len(va.hazards), "hazard detections")


if __name__ == "__main__":
    main()
