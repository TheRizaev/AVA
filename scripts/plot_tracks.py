"""EDA: draw every track on the video's median background, coloured by heading.

    python scripts/plot_tracks.py C3905 --kind vehicle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def heading_color(dx: float, dy: float) -> tuple[int, int, int]:
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    hsv = np.uint8([[[int(ang / 2), 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--kind", choices=["vehicle", "person"], default="vehicle")
    ap.add_argument("--min-len", type=int, default=15, help="min samples per track")
    args = ap.parse_args()
    data = np.load(f"work/tracks/{args.name}.npz")
    tr = data["tracks"]
    classes = config.VEHICLES if args.kind == "vehicle" else (config.PERSON,)
    bg = cv2.imread(f"work/bg/{args.name}.png")
    canvas = (bg * 0.55).astype(np.uint8)
    n = 0
    for tid in np.unique(tr[:, 2]):
        t = tr[tr[:, 2] == tid]
        cls = np.bincount(t[:, 8].astype(int)).argmax()
        if cls not in classes or len(t) < args.min_len:
            continue
        pts = np.stack([(t[:, 3] + t[:, 5]) / 2, t[:, 6]], axis=1)
        if np.linalg.norm(pts[-1] - pts[0]) < 30:  # parked / static
            continue
        n += 1
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            if np.hypot(*d) < 0.5:
                continue
            cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), heading_color(*d), 2, cv2.LINE_AA)
    # legend: colour wheel
    cx, cy, r = 1820, 980, 70
    for a in range(0, 360, 3):
        rad = np.radians(a)
        cv2.line(canvas, (cx, cy), (int(cx + r * np.cos(rad)), int(cy + r * np.sin(rad))),
                 heading_color(np.cos(rad), np.sin(rad)), 3)
    out = Path("work/eda")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{args.name}_{args.kind}_tracks.jpg"), canvas)
    print(f"{n} moving {args.kind} tracks drawn")


if __name__ == "__main__":
    main()
