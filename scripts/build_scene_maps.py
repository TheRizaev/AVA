"""Learn the scene's traffic flow field and occupancy from the sample tracks.

For every 16x16 px cell of the reference frame we accumulate the motion
direction of the vehicles that pass through it (a 16-bin histogram). The
normalised histogram is the "normal flow" used by the wrong-way rule, and the
vehicle / pedestrian occupancy maps are EDA material for the website.

    python scripts/build_scene_maps.py --tracks work/tracks --out assets/flow.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.scene import warp_points  # noqa: E402
from src.tracks import load_track_table, per_track  # noqa: E402

CELL = 16
N_BINS = 16
DRIVEN_MIN_DENSITY = 0.02   # blurred vehicle ground-point hits per pixel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="work/tracks")
    ap.add_argument("--registration", default="work/bg/registration.json")
    ap.add_argument("--out", default="assets/flow.npz")
    ap.add_argument("--eda", default="work/eda")
    args = ap.parse_args()
    reg = json.loads(Path(args.registration).read_text())
    w, h = config.ANALYSIS_SIZE
    gw, gh = -(-w // CELL), -(-h // CELL)
    hist = np.zeros((gh, gw, N_BINS))
    veh_occ = np.zeros((h, w), np.float32)
    ped_occ = np.zeros((h, w), np.float32)

    for f in sorted(Path(args.tracks).glob("*.npz")):
        H = np.array(reg[f.stem]["H"])
        table = load_track_table(f)
        for tr in per_track(table, H):
            if tr.kind == "person":
                for x, y in tr.xy.astype(int):
                    if 0 <= x < w and 0 <= y < h:
                        ped_occ[y, x] += 1
                continue
            if tr.kind != "vehicle":
                continue
            v = tr.velocity()               # px/s in reference frame, smoothed
            speed = np.hypot(v[:, 0], v[:, 1])
            for (x, y), (vx, vy), s in zip(tr.xy, v, speed):
                xi, yi = int(x), int(y)
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                veh_occ[yi, xi] += 1
                if s < 15:                   # direction is meaningless when (nearly) stopped
                    continue
                b = int(((np.arctan2(vy, vx) + np.pi) / (2 * np.pi)) * N_BINS) % N_BINS
                hist[yi // CELL, xi // CELL, b] += 1
        print("accumulated", f.stem)

    # where vehicles actually drive: dilated ground-point occupancy (the "driven" mask). Pedestrians
    # inside it are on a live lane; sidewalks, the bus-stop edge and corners where people wait are not.
    driven = cv2.GaussianBlur(veh_occ, (0, 0), 8) > DRIVEN_MIN_DENSITY
    driven = cv2.dilate(driven.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    cv2.imwrite(str(Path(args.out).parent / "driven.png"), driven * 255)

    # smooth spatially so sparse cells borrow from neighbours
    smooth = np.stack([cv2.GaussianBlur(hist[..., b], (0, 0), 1.0) for b in range(N_BINS)], axis=-1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, hist=smooth.astype(np.float32), cell=CELL, n_bins=N_BINS)
    print("saved", args.out)

    # ---- EDA images --------------------------------------------------------
    eda = Path(args.eda)
    eda.mkdir(parents=True, exist_ok=True)
    ref = cv2.imread(str(config.SCENE_REF_IMAGE))
    for name, occ in (("vehicle", veh_occ), ("person", ped_occ)):
        heat = cv2.GaussianBlur(occ, (0, 0), 6)
        heat = np.log1p(heat)
        heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        vis = np.where(heat[..., None] > 10, (0.35 * ref + 0.65 * color), 0.5 * ref).astype(np.uint8)
        cv2.imwrite(str(eda / f"occupancy_{name}.jpg"), vis)
    vis = (ref * 0.5).astype(np.uint8)
    total = smooth.sum(-1)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS * 2 * np.pi - np.pi
    for gy in range(0, gh, 2):
        for gx in range(0, gw, 2):
            if total[gy, gx] < 3:
                continue
            b = smooth[gy, gx].argmax()
            dom = smooth[gy, gx, b] / total[gy, gx]
            ang = centers[b]
            c = (gx * CELL + CELL // 2, gy * CELL + CELL // 2)
            e = (int(c[0] + 14 * np.cos(ang)), int(c[1] + 14 * np.sin(ang)))
            col = (0, 255, 0) if dom > 0.5 else (0, 200, 255)
            cv2.arrowedLine(vis, c, e, col, 1, cv2.LINE_AA, tipLength=0.4)
    cv2.imwrite(str(eda / "flow_field.jpg"), vis)


if __name__ == "__main__":
    main()
