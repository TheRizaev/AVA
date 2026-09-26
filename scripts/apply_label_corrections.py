"""Apply the blind re-check corrections (dev/label_corrections.json) to dev/labels.json.

Idempotent: removals of absent segments do nothing, and added segments merge with themselves
(same-class overlaps are one segment, as in the official convention).

    python scripts/apply_label_corrections.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.base import Event, merge_same_class  # noqa: E402

LABELS = Path("dev/labels.json")
CORRECTIONS = Path("dev/label_corrections.json")


def same(a: list, b: list) -> bool:
    return a[2] == b[2] and abs(a[0] - b[0]) < 0.01 and abs(a[1] - b[1]) < 0.01


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    applied = 0
    for c in json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]:
        if not c["apply"]:
            continue
        events = labels[c["video"]]["events"]
        if c["op"] in ("remove", "replace"):
            events[:] = [e for e in events if not same(e, c["segment"])]
        events.extend([c["segment"]] if c["op"] == "add" else c.get("with", []))
        applied += 1
    for video in labels.values():
        merged = merge_same_class([Event(s, e, lab) for s, e, lab in video["events"]])
        video["events"] = sorted((ev.as_list() for ev in merged), key=lambda x: (x[0], x[2]))
    LABELS.write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")
    n = sum(len(v["events"]) for v in labels.values())
    print(f"applied {applied} corrections -> {LABELS} ({n} events)")


if __name__ == "__main__":
    main()
