"""Compare detector throughput on this GPU at the analysis resolution.

    python scripts/bench_detector.py samples/C3896.MP4
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video import iter_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", default=["yolo11m.pt", "yolo11l.pt", "yolo26m.pt", "yolo26l.pt"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    frames = [f for _, _, f in islice(iter_frames(args.video, (1920, 1080), min_step=0.1), 96)]
    for name in args.models:
        try:
            model = YOLO(str(Path("weights") / name) if (Path("weights") / name).exists() else name)
        except Exception as exc:  # model not available in this ultralytics version
            print(name, "unavailable:", exc)
            continue
        model.predict(frames[: args.batch], imgsz=args.imgsz, half=True, verbose=False)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_det = 0
        for i in range(0, len(frames), args.batch):
            res = model.predict(frames[i:i + args.batch], imgsz=args.imgsz, half=True, verbose=False,
                                classes=[0, 1, 2, 3, 5, 7], conf=0.25)
            n_det += sum(len(r.boxes) for r in res)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{name:<12} imgsz={args.imgsz} {len(frames) / dt:6.1f} img/s  "
              f"{dt / len(frames) * 1000:5.1f} ms/img  dets/img={n_det / len(frames):.1f}")


if __name__ == "__main__":
    main()
