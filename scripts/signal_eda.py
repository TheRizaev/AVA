"""EDA: how do the two visible signal heads behave over time?

Writes, per video, a montage of signal-head crops and a CSV of lamp colour
scores (bright red / green / yellow pixel fractions) at ~6 fps.

    python scripts/signal_eda.py C3896
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene import load_layout, warp_points  # noqa: E402
from src.signals import lamp_scores  # noqa: E402


def main() -> None:
    name = sys.argv[1]
    H = np.array(json.loads(Path("work/bg/registration.json").read_text())[name]["H"])
    Hinv = np.linalg.inv(H)
    boxes = {}
    for sname, (x1, y1, x2, y2) in load_layout()["signals"].items():
        p = warp_points(Hinv, np.array([[x1, y1], [x2, y2]], float))
        boxes[sname] = [int(p[0, 0]), int(p[0, 1]), int(p[1, 0]), int(p[1, 1])]
    cap = cv2.VideoCapture(f"work/proxy/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, tiles, idx = [], {k: [] for k in boxes}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 5 == 0:
            t = idx / fps
            row = [t]
            for sname, (x1, y1, x2, y2) in boxes.items():
                crop = frame[y1:y2, x1:x2]
                row += list(lamp_scores(crop))
                if idx % 150 == 0:
                    tiles[sname].append(cv2.resize(crop, ((x2 - x1) * 3, (y2 - y1) * 3), interpolation=cv2.INTER_NEAREST))
            rows.append(row)
        idx += 1
    out = Path("work/eda/signals")
    out.mkdir(parents=True, exist_ok=True)
    header = "t," + ",".join(f"{s}_{c}" for s in boxes for c in ("red", "yellow", "green"))
    np.savetxt(out / f"{name}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.4f")
    for sname, ts in tiles.items():
        h = max(x.shape[0] for x in ts)
        ts = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 2, cv2.BORDER_CONSTANT) for x in ts]
        cv2.imwrite(str(out / f"{name}_{sname}_montage.jpg"), np.hstack(ts))
    print(name, len(rows), "samples")


if __name__ == "__main__":
    main()
