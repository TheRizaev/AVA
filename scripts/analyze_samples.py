"""Run the analysis pass (detection, tracking, registration, lamps) and cache it.

    python scripts/analyze_samples.py samples --out work/cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import analyze  # noqa: E402
from src.hazards import HazardDetector  # noqa: E402
from src.detection import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--out", default="work/cache")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    hazards = HazardDetector()
    src = Path(args.videos)
    for p in [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4"):
        t0 = time.perf_counter()
        va = analyze(str(p), model, hazards)
        dt = time.perf_counter() - t0
        va.save(out / f"{p.stem}.npz")
        print(f"{p.name}: {va.meta.duration:.0f}s video, {dt:.0f}s ({dt / va.meta.duration:.2f}x), "
              f"{len(va.tracks)} rows, {len(va.reg_t)} registrations", flush=True)


if __name__ == "__main__":
    main()
