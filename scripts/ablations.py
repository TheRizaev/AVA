"""Ablations on the sample videos: detector, input size and sampling rate vs dev Score A and runtime.

    python scripts/ablations.py                      # runs every variant not cached yet, then scores
    python scripts/ablations.py --score-only

Each variant re-runs the Part A analysis pass (detection, tracking, registration, lamps, hazards)
on the four samples with one setting changed, then the same rules and the official evaluate.py
against dev/labels.json. Writes dev/ablations.json and website/data/ablations.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
from src import config  # noqa: E402
from src.analysis import VideoAnalysis, analyze  # noqa: E402
from src.detection import load_model  # noqa: E402
from src.events import build_context, detect  # noqa: E402
from src.hazards import HazardDetector  # noqa: E402

VARIANTS = [
    {"key": "baseline", "label": "YOLO26-L · 1280 px · 10 fps (submitted)", "weights": "yolo26l.pt", "imgsz": 1280, "fps": 10.0},
    {"key": "yolo26m", "label": "YOLO26-M · 1280 px · 10 fps", "weights": "yolo26m.pt", "imgsz": 1280, "fps": 10.0},
    {"key": "yolo26s", "label": "YOLO26-S · 1280 px · 10 fps", "weights": "yolo26s.pt", "imgsz": 1280, "fps": 10.0},
    {"key": "yolo11l", "label": "YOLO11-L · 1280 px · 10 fps", "weights": "yolo11l.pt", "imgsz": 1280, "fps": 10.0},
    {"key": "imgsz960", "label": "YOLO26-L · 960 px · 10 fps", "weights": "yolo26l.pt", "imgsz": 960, "fps": 10.0},
    {"key": "fps5", "label": "YOLO26-L · 1280 px · 5 fps", "weights": "yolo26l.pt", "imgsz": 1280, "fps": 5.0},
]
OUT = ROOT / "work" / "ablations"


def run(variant: dict, samples: Path) -> None:
    folder = OUT / variant["key"]
    folder.mkdir(parents=True, exist_ok=True)
    model = load_model(config.WEIGHTS_DIR / variant["weights"])
    hazards = HazardDetector()
    for p in sorted(p for p in samples.iterdir() if p.suffix.lower() == ".mp4"):
        dst = folder / f"{p.stem}.npz"
        if dst.exists():
            continue
        t0 = time.perf_counter()
        va = analyze(str(p), model, hazards, sample_fps=variant["fps"], det_imgsz=variant["imgsz"])
        dt = time.perf_counter() - t0
        va.thumbs, va.thumb_t = va.thumbs[:0], va.thumb_t[:0]
        va.save(dst)
        (folder / f"{p.stem}.time").write_text(f"{dt:.1f} {va.meta.duration:.2f}")
        print(f"{variant['key']} {p.name}: {dt:.0f}s ({dt / va.meta.duration:.2f}x)", flush=True)


def score(variant: dict, gt: dict) -> dict:
    folder = OUT / variant["key"]
    pred, secs, dur = {}, 0.0, 0.0
    for f in sorted(folder.glob("*.npz")):
        ctx = build_context(VideoAnalysis.load(f))
        pred[f"{f.stem}.MP4"] = {"events": [e.as_list() for e in detect(ctx, config.ENABLED_CLASSES)], "risk": []}
        a, b = (float(x) for x in (folder / f"{f.stem}.time").read_text().split())
        secs, dur = secs + a, dur + b
    rep = evaluate.evaluate_part_a({v: gt[v] for v in pred}, pred)
    return {"key": variant["key"], "label": variant["label"], "score_a": round(rep["score_a"], 4),
            "micro_f1_05": round(rep["micro"]["0.5"]["f1"], 3),
            "per_class": {c: round(pc["f1_mean"], 3) for c, pc in rep["per_class"].items()},
            "analysis_x_realtime": round(secs / dur, 3) if dur else None,
            "n_events": sum(len(p["events"]) for p in pred.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "samples"))
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    variants = [v for v in VARIANTS if not args.only or v["key"] in args.only]
    if not args.score_only:
        for v in variants:
            run(v, Path(args.samples))
    gt = json.loads((ROOT / "dev" / "labels.json").read_text())
    rows = [score(v, gt) for v in VARIANTS if (OUT / v["key"]).exists() and any((OUT / v["key"]).glob("*.npz"))]
    out = {"note": ("Part A analysis pass re-run on the four sample videos with one setting changed; same rules, "
                    "official evaluate.py against our dev labels (76 events). Runtime is the analysis pass only "
                    "(decode + detection + tracking + hazards), RTX 2060 Super + i5-12400F."),
           "variants": rows}
    for path in (ROOT / "dev" / "ablations.json", ROOT / "website" / "data" / "ablations.json"):
        path.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['label']:42s} Score A {r['score_a']:.3f}  micro {r['micro_f1_05']:.3f}  "
              f"{r['analysis_x_realtime']}x  events {r['n_events']}")


if __name__ == "__main__":
    main()
