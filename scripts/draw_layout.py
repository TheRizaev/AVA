"""Overlay the scene layout on a video's background (registered) for visual checks.

    python scripts/draw_layout.py C3902
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene import load_layout, warp_points  # noqa: E402

COLORS = {"approach": (0, 165, 255), "outbound": (255, 128, 0), "junction": (180, 180, 180),
          "median": (255, 255, 255), "stop_zone": (0, 0, 255)}


def main() -> None:
    name = sys.argv[1]
    img = cv2.imread(f"work/bg/{name}.png")
    H = np.array(json.loads(Path("work/bg/registration.json").read_text())[name]["H"])
    Hinv = np.linalg.inv(H)                   # reference -> this video
    layout = load_layout()
    over = img.copy()
    for pname, pts in layout["polygons"].items():
        p = warp_points(Hinv, np.array(pts, float)).astype(np.int32)
        col = COLORS.get(pname, (0, 255, 0) if pname.startswith("cw") else (255, 0, 255))
        cv2.fillPoly(over, [p], col)
        cv2.polylines(img, [p], True, col, 2)
        c = p.mean(0).astype(int)
        cv2.putText(img, pname, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for lname, pts in layout["lines"].items():
        p = warp_points(Hinv, np.array(pts, float)).astype(np.int32)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 255), 3)
    for head in ("vehicle", "ped"):
        for colour, xy in layout["signals"][head].items():
            p = warp_points(Hinv, np.array([xy], float)).astype(int)[0]
            cv2.circle(img, tuple(p), 5, (0, 255, 255), 1)
    out = cv2.addWeighted(img, 0.7, over, 0.3, 0)
    Path("work/eda").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"work/eda/layout_{name}.jpg", out)


if __name__ == "__main__":
    main()
