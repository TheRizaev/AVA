"""Class confusion on the dev labels: which label our segments get when they overlap a labelled event.

    python scripts/confusion.py --pred predictions_samples.json

Segments are matched one-to-one regardless of class (greedy by temporal IoU, >= --iou), then
the labelled class is compared with the predicted one. Unmatched labelled events are "missed",
unmatched predictions are "false alarm". Writes dev/confusion.json and website/data/confusion.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402


def match(gt: list, pred: list, thr: float) -> tuple[list, list, list]:
    pairs = sorted(((evaluate.tiou((g[0], g[1]), (p[0], p[1])), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
                   reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((gt[i][2], pred[j][2]))
    missed = [gt[i][2] for i in range(len(gt)) if i not in used_g]
    extra = [pred[j][2] for j in range(len(pred)) if j not in used_p]
    return matched, missed, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ROOT / "dev" / "labels.json"))
    ap.add_argument("--pred", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text())
    pred = json.loads(Path(args.pred).read_text())["videos"]
    cells, missed, extra = Counter(), Counter(), Counter()
    for vid, g in gt.items():
        m, mi, ex = match(g["events"], pred.get(vid, {}).get("events", []), args.iou)
        cells.update(m)
        missed.update(mi)
        extra.update(ex)
    classes = sorted({c for pair in cells for c in pair} | set(missed) | set(extra))
    matrix = [[cells.get((a, b), 0) for b in classes] for a in classes]
    out = {"iou": args.iou, "classes": classes, "matrix": matrix,
           "missed": [missed.get(c, 0) for c in classes], "false_alarm": [extra.get(c, 0) for c in classes],
           "note": ("Rows: labelled class; columns: predicted class, for segments matched one-to-one regardless of "
                    f"class at tIoU >= {args.iou}. Off-diagonal cells are class confusions.")}
    for path in (ROOT / "dev" / "confusion.json", ROOT / "website" / "data" / "confusion.json"):
        path.write_text(json.dumps(out, indent=1))
    w = max(len(c) for c in classes)
    print(" " * (w + 2) + " ".join(f"{c[:6]:>6s}" for c in classes) + "  missed")
    for c, row, mi in zip(classes, matrix, out["missed"]):
        print(f"{c:>{w}s}  " + " ".join(f"{x:6d}" for x in row) + f"  {mi:6d}")
    print(f"{'false alarm':>{w}s}  " + " ".join(f"{x:6d}" for x in out["false_alarm"]))
    off = sum(matrix[i][j] for i in range(len(classes)) for j in range(len(classes)) if i != j)
    print(f"matched {sum(map(sum, matrix))}, of which cross-class {off}")


if __name__ == "__main__":
    main()
