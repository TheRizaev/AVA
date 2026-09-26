"""Contact sheets for external clips (used to time accidents in third-party CCTV footage).

    python scripts/ext_sheet.py VIDEO.mp4 0 9                  # 3x3 grid, 1 s apart
    python scripts/ext_sheet.py VIDEO.mp4 4 6 --step 0.2 --crop 400 200 1200 700
    python scripts/ext_sheet.py VIDEO.mp4 --info                # duration, fps, size

Tiles carry their timestamp (seconds from the first frame). The output path is printed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "work" / "ext_annot" / "sheets"


def read_frames(path: str, times: list[float]) -> tuple[list[np.ndarray | None], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = sorted({int(round(t * fps)) for t in times})
    frames: dict[int, np.ndarray] = {}
    i, j = 0, 0
    while j < len(want):          # sequential read: exact frames, no seek inaccuracy
        ok = cap.grab()
        if not ok:
            break
        if i == want[j]:
            ok, fr = cap.retrieve()
            if ok:
                frames[i] = fr
            j += 1
        i += 1
    cap.release()
    return [frames.get(int(round(t * fps))) for t in times], fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("t0", type=float, nargs="?", default=0.0)
    ap.add_argument("t1", type=float, nargs="?")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--width", type=int, default=640, help="tile width")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if args.info:
        print(f"duration {n / fps:.2f} s, fps {fps:.2f}, frames {n}, size {w}x{h}")
        return
    t1 = args.t1 if args.t1 is not None else args.t0 + 8 * args.step
    times = [round(t, 3) for t in np.arange(args.t0, t1 + 1e-6, args.step) if 0 <= t < n / fps]
    if args.crop:
        x1, y1, x2, y2 = args.crop
        args.crop = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
        if args.crop[2] <= args.crop[0] or args.crop[3] <= args.crop[1]:
            raise SystemExit(f"crop {x1} {y1} {x2} {y2} is outside the {w}x{h} frame")
    frames, _ = read_frames(args.video, times)
    tiles = []
    for t, fr in zip(times, frames):
        if fr is None:
            continue
        if args.crop:
            x1, y1, x2, y2 = args.crop
            fr = fr[y1:y2, x1:x2]
        s = args.width / fr.shape[1]
        img = cv2.resize(fr, (args.width, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (118, 26), (0, 0, 0), -1)
        cv2.putText(img, f"{t:7.2f}s", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        raise SystemExit("no frames in that range")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = np.zeros((rows * th, args.cols * tw, 3), np.uint8)
    for k, img in enumerate(tiles):
        r, c = divmod(k, args.cols)
        sheet[r * th:r * th + img.shape[0], c * tw:c * tw + img.shape[1]] = img
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crop = "_crop{}_{}_{}_{}".format(*args.crop) if args.crop else ""
    out = OUT_DIR / f"{Path(args.video).stem}_{args.t0:.2f}_{t1:.2f}_{args.step:g}{crop}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
