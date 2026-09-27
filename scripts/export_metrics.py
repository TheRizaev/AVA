"""Score the detector on the dev labels and write the website's metrics.json.

    python scripts/export_metrics.py --gt dev/labels.json --pred predictions_samples.json

Uses the official evaluate.py on the full sample videos (Part B is only scored
when the labels contain an accident) and also stores the report in dev/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="dev/labels.json")
    ap.add_argument("--pred", default="predictions_samples.json")
    ap.add_argument("--site", default="website/data/metrics.json")
    ap.add_argument("--report", default="dev/eval_report.json")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="unlabelled windows to leave out of the score, e.g. C3902:160-240")
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text())
    pred = json.loads(Path(args.pred).read_text())
    for w in args.exclude:   # drop events starting in windows that were not annotated
        vid, rng = w.split(":")
        a, b = map(float, rng.split("-"))
        key = f"{vid}.MP4"
        drop = lambda e: a <= e[0] < b  # noqa: E731
        gt[key]["events"] = [e for e in gt[key]["events"] if not drop(e)]
        pred["videos"][key]["events"] = [e for e in pred["videos"][key]["events"] if not drop(e)]
    rep = evaluate.evaluate(gt, pred, per_video=True)
    Path(args.report).write_text(json.dumps(rep, indent=1))
    a, b = rep["part_a"], rep["part_b"]
    n_events = sum(len(v["events"]) for v in gt.values())
    excluded = sum(float(w.split(":")[1].split("-")[1]) - float(w.split(":")[1].split("-")[0]) for w in args.exclude)
    minutes = (sum(v["duration"] for v in gt.values()) - excluded) / 60
    metrics = {
        "score_a": round(a["score_a"], 4),
        "score_b": round(b["score_b"], 4) if b else None,
        "model_score": round(rep["model_score"], 4),
        "per_class": {c: {"0.3": round(pc["0.3"]["f1"], 3), "0.5": round(pc["0.5"]["f1"], 3),
                          "0.7": round(pc["0.7"]["f1"], 3), "f1_mean": round(pc["f1_mean"], 3),
                          "tp_fp_fn_05": [pc["0.5"]["tp"], pc["0.5"]["fp"], pc["0.5"]["fn"]]}
                      for c, pc in a["per_class"].items()},
        "micro": {t: round(v["f1"], 3) for t, v in a["micro"].items()},
        "class_agnostic": {t: round(v["f1"], 3) for t, v in a["class_agnostic"].items()},
        "note": ("Part B is not scored: the sample videos contain no accident."
                 if b is None else ""),
        "dataset": (f"Our own labels of the 4 sample videos ({minutes:.1f} min, {n_events} events): "
                    "a blind sweep in 64-80 s windows plus a per-video pass for long events, every event "
                    "re-checked in an independent pass, detector candidates the first pass missed checked in "
                    "a third pass, and remaining disagreements re-checked blind (see dev/README.md)."),
    }
    Path(args.site).write_text(json.dumps(metrics, indent=1))
    evaluate.print_report(rep)
    print("wrote", args.site, args.report)


if __name__ == "__main__":
    main()
