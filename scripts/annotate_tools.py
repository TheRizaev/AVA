"""Frame-extraction helpers used to build the dev-set labels by looking at the video.

All times are seconds from the first frame; frames come from the 1080p proxies.

    python scripts/annotate_tools.py sheet C3896 0 9            # 3x3 grid, 1 s apart
    python scripts/annotate_tools.py sheet C3896 30 32 --step 0.25 --crop 200 350 1000 700
    python scripts/annotate_tools.py frame C3896 41.5 --crop 800 400 1400 800

Each tile shows its timestamp and an enlarged inset of the vehicle signal head
(top-right) so red/green can be judged without a second image. Output paths are
printed; open them with an image viewer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene import load_layout, warp_points  # noqa: E402

PROXY_DIR = ROOT / "work" / "proxy"
OUT_DIR = ROOT / "work" / "annot"


def _registration(name: str) -> np.ndarray:
    reg = json.loads((ROOT / "work" / "bg" / "registration.json").read_text())
    return np.array(reg[name]["H"])


def _signal_box(name: str) -> tuple[int, int, int, int]:
    sig = load_layout()["signals"]["vehicle"]
    pts = np.array([sig["red"], sig["green"]], float)
    p = warp_points(np.linalg.inv(_registration(name)), pts)
    cx, y1, y2 = int(p[:, 0].mean()), int(p[0, 1]) - 14, int(p[1, 1]) + 14
    return cx - 16, y1, cx + 16, y2


def _read(cap: cv2.VideoCapture, fps: float, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def _tile(frame: np.ndarray, t: float, crop, sig_box, width: int) -> np.ndarray:
    x1, y1, x2, y2 = sig_box
    inset = cv2.resize(frame[y1:y2, x1:x2], ((x2 - x1) * 3, (y2 - y1) * 3), interpolation=cv2.INTER_NEAREST)
    if crop:
        cx1, cy1, cx2, cy2 = crop
        frame = frame[cy1:cy2, cx1:cx2]
    scale = width / frame.shape[1]
    img = cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if not crop:
        ih, iw = inset.shape[:2]
        img[0:ih, width - iw:width] = inset
        cv2.rectangle(img, (width - iw, 0), (width - 1, ih), (255, 255, 255), 1)
    label = f"t={t:.2f}s"
    cv2.putText(img, label, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(img, label, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def sheet(name: str, t0: float, t1: float, step: float, cols: int, crop, width: int) -> Path:
    cap = cv2.VideoCapture(str(PROXY_DIR / f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sig_box = _signal_box(name)
    times = np.arange(t0, t1 + 1e-6, step)
    tiles = [_tile(f, t, crop, sig_box, width) for t in times if (f := _read(cap, fps, t)) is not None]
    if not tiles:
        raise SystemExit("no frames in range")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    tag = "_crop" + "-".join(map(str, crop)) if crop else ""
    out = OUT_DIR / name / f"sheet_{t0:07.2f}_{t1:07.2f}_s{step}_c{cols}_w{width}{tag}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("name")
    s.add_argument("t0", type=float)
    s.add_argument("t1", type=float)
    s.add_argument("--step", type=float, default=1.0)
    s.add_argument("--cols", type=int, default=3)
    s.add_argument("--crop", type=int, nargs=4)
    s.add_argument("--width", type=int, default=640)
    f = sub.add_parser("frame")
    f.add_argument("name")
    f.add_argument("t", type=float)
    f.add_argument("--crop", type=int, nargs=4)
    f.add_argument("--width", type=int, default=1600)
    args = ap.parse_args()
    if args.cmd == "sheet":
        print(sheet(args.name, args.t0, args.t1, args.step, args.cols, args.crop, args.width))
    else:
        print(sheet(args.name, args.t, args.t, 1.0, 1, args.crop, args.width))


if __name__ == "__main__":
    main()
