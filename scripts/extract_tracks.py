"""Run detection + tracking on the sample videos and cache the track tables.

Rule development iterates on these caches instead of re-running the detector.

    python scripts/extract_tracks.py samples --out work/tracks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection import COLUMNS, load_model, track_video  # noqa: E402
from src.video import probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--out", default="work/tracks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    src = Path(args.videos)
    for p in [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4"):
        meta = probe(str(p))
        t0 = time.perf_counter()
        tracks = track_video(str(p), model)
        dt = time.perf_counter() - t0
        np.savez_compressed(out / f"{p.stem}.npz", tracks=tracks, columns=np.array(COLUMNS),
                            fps=meta.fps, duration=meta.duration, n_frames=meta.n_frames)
        n_ids = len(np.unique(tracks[:, 2])) if len(tracks) else 0
        print(f"{p.name}: {meta.duration:.0f}s video, {dt:.0f}s wall ({dt / meta.duration:.2f}x), "
              f"{len(tracks)} rows, {n_ids} tracks")


if __name__ == "__main__":
    main()
