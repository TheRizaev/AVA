"""Cache detector/tracker output for third-party clips, so Part B cues and the accident rule
can be evaluated and tuned without re-running the detector.

    python scripts/ext_cache.py ../external/tad/eval_acc ../external/tad/eval_norm --out work/ext_cache

Per clip it writes
  <name>.npz        Part A analysis (src.analysis.analyze, exactly what detect_events uses)
  <name>.risk.npz   Part B online tracker rows: every frame the RiskEstimator analyses
                    (same stride, detector size and tracker), with the registration state.
Frames for Part B are decoded with OpenCV like the official harness.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.analysis import analyze  # noqa: E402
from src.detection import detect_batch, new_tracker  # noqa: E402
from src.pipeline import shared_hazard_detector, shared_model  # noqa: E402
from src.risk import REGISTER_EVERY_SEC, RISK_FPS, RISK_IMGSZ  # noqa: E402
from src.scene import Registrar  # noqa: E402


def risk_rows(path: Path, model) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / RISK_FPS)))
    tracker, reg = new_tracker(RISK_FPS), Registrar(REGISTER_EVERY_SEC)
    rows, times, Hs, fit = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if i % stride == 0:
            if frame.shape[1] != config.ANALYSIS_SIZE[0]:
                frame = cv2.resize(frame, config.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            reg.update(t, frame)
            det = detect_batch(model, [frame], imgsz=RISK_IMGSZ)[0]
            tr = tracker.update(det, frame)
            tr = np.asarray(tr, float).reshape(-1, tr.shape[1] if len(tr) else 8)[:, :7]
            rows.append(np.column_stack([np.full(len(tr), len(times)), tr]) if len(tr) else np.zeros((0, 8)))
            times.append(t)
            Hs.append(reg.H.copy())
            fit.append(reg.have_fit)
        i += 1
    cap.release()
    return {"t": np.array(times), "rows": np.concatenate(rows) if rows else np.zeros((0, 8)),
            "H": np.array(Hs).reshape(-1, 3, 3), "have_fit": np.array(fit, bool), "fps": fps, "n_frames": i}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="work/ext_cache")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: only every n-th clip starting at k (run n processes)")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = shared_model()
    for d in args.dirs:
        for p in sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".mp4")[k::n]:
            t0 = time.perf_counter()
            if not args.skip_a and not (out / f"{p.stem}.npz").exists():
                analyze(str(p), model, shared_hazard_detector()).save(out / f"{p.stem}.npz")
            if not (out / f"{p.stem}.risk.npz").exists():
                tmp = out / f"{p.stem}.risk.tmp.npz"      # written whole, then renamed: an interrupted
                np.savez_compressed(tmp, **risk_rows(p, model))   # run never leaves a partial cache
                tmp.replace(out / f"{p.stem}.risk.npz")
            print(f"{p.name}: {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
