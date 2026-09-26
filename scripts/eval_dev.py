"""Evaluate the rules on the dev labels and list every miss / false alarm.

    python scripts/eval_dev.py --gt dev/labels.json [--only red_light jaywalking] [--iou 0.3]

Runs the rules on the cached analyses, scores them with the official
evaluate.py, then prints per class the ground-truth segments nobody matched
(FN) and the predictions that matched nothing (FP) at the given tIoU, with the
best-overlapping segment on the other side, for error analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
from src import config  # noqa: E402
from src.analysis import VideoAnalysis  # noqa: E402
from src.events import build_context, detect  # noqa: E402


def predictions(cache: Path, only: tuple[str, ...] | None) -> dict:
    videos = {}
    for f in sorted(cache.glob("*.npz")):
        ctx = build_context(VideoAnalysis.load(f))
        videos[f"{f.stem}.MP4"] = {"events": [e.as_list() for e in detect(ctx, only)], "risk": []}
    return {"team": "dev", "videos": videos}


def unmatched(gt_events, pred_events, label, thr):
    g = [(s, e) for s, e, lab in gt_events if lab == label]
    p = [(s, e) for s, e, lab in pred_events if lab == label]
    pairs = sorted(((evaluate.tiou(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
    ug, up = set(range(len(g))), set(range(len(p)))
    for iou, i, j in pairs:
        if iou >= thr and i in ug and j in up:
            ug.discard(i)
            up.discard(j)
    return [g[i] for i in sorted(ug)], [p[j] for j in sorted(up)], g, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="dev/labels.json")
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--save", default="work/pred_dev.json")
    ap.add_argument("--windows", nargs="*", help="score only events starting in these windows, e.g. C3896:0-70 C3896:140-340")
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text())
    only = tuple(args.only) if args.only else config.ENABLED_CLASSES
    pred = predictions(Path(args.cache), only)
    Path(args.save).write_text(json.dumps(pred, indent=1))
    if args.only:   # explicit --only: score just those classes; otherwise every labelled class counts
        gt = {v: {**d, "events": [e for e in d["events"] if e[2] in only]} for v, d in gt.items()}
    if args.windows:
        win: dict[str, list[tuple[float, float]]] = {}
        for w in args.windows:
            vid, rng = w.split(":")
            a, b = map(float, rng.split("-"))
            win.setdefault(f"{vid}.MP4", []).append((a, b))
        keep = lambda vid, e: any(a <= e[0] < b for a, b in win.get(vid, []))  # noqa: E731
        gt = {v: {**d, "events": [e for e in d["events"] if keep(v, e)]} for v, d in gt.items() if v in win}
        pred["videos"] = {v: {**d, "events": [e for e in d["events"] if keep(v, e)]}
                          for v, d in pred["videos"].items() if v in win}
    rep = evaluate.evaluate(gt, pred, per_video=True)
    evaluate.print_report(rep)
    labels = sorted({e[2] for d in gt.values() for e in d["events"]} | {e[2] for d in pred["videos"].values() for e in d["events"]})
    print(f"\n=== unmatched at tIoU {args.iou} ===")
    for label in labels:
        lines = []
        for vid, d in gt.items():
            fn, fp, g, p = unmatched(d["events"], pred["videos"].get(vid, {}).get("events", []), label, args.iou)
            for s, e in fn:
                best = max(p, key=lambda b: evaluate.tiou((s, e), b), default=None)
                lines.append(f"  FN {vid} {s:7.2f}-{e:7.2f}  best pred {best} iou {evaluate.tiou((s, e), best) if best else 0:.2f}")
            for s, e in fp:
                best = max(g, key=lambda b: evaluate.tiou((s, e), b), default=None)
                lines.append(f"  FP {vid} {s:7.2f}-{e:7.2f}  best gt {best} iou {evaluate.tiou((s, e), best) if best else 0:.2f}")
        if lines:
            print(f"{label}:")
            print("\n".join(lines))


if __name__ == "__main__":
    main()
