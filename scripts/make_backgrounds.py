"""EDA step: median background per sample video + registration to a reference.

The per-video median of ~60 frames spread over the clip removes moving traffic
and leaves the road markings, which is what we draw the scene layout on. It
also tells us how much the framing moves between clips (it does: the camera is
re-mounted between recordings), which is why the layout is registered per video.

    python scripts/make_backgrounds.py samples --out work/bg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video import iter_frames, probe  # noqa: E402

SIZE = (1920, 1080)


def median_background(path: str, n: int = 60) -> np.ndarray:
    meta = probe(path)
    step = meta.duration / n
    frames = [f for _, _, f in iter_frames(path, SIZE, skip="NONKEY", min_step=step)]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, int]:
    sift = cv2.SIFT_create(4000)
    g1, g2 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (src, dst))
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n2 in matches if m.distance < 0.7 * n2.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    return H, int(inl.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--out", default="work/bg")
    ap.add_argument("--ref", default="C3896")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bgs = {}
    for p in sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4"):
        bg = median_background(str(p))
        cv2.imwrite(str(out / f"{p.stem}.png"), bg)
        bgs[p.stem] = bg
        print("background", p.stem)
    report = {}
    ref = bgs[args.ref]
    for name, bg in bgs.items():
        H, n_inl = homography(bg, ref)
        # displacement of the image corners/centre when mapped into the reference
        pts = np.float32([[0, 0], [1919, 0], [0, 1079], [1919, 1079], [960, 540]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(pts, H).reshape(-1, 2) - pts.reshape(-1, 2)
        report[name] = {"inliers": n_inl, "shift_px": np.round(moved, 1).tolist(), "H": H.tolist()}
        print(name, "inliers", n_inl, "shift", np.round(moved, 1).tolist())
    (out / "registration.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
