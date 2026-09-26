"""Collect the annotation workflow's output into dev/labels.json.

Every first-pass (sweep) event is keyed by (video, label, start) - the same key
the verification agent's label carries (``verify:<video>:<label>@<start>``).
Until its verdict arrives a sweep event is used as it is (if its confidence is
at least --min-conf); a verdict then replaces it: rejected -> dropped,
otherwise the verified label and times. Same-class overlaps are merged like
the official convention.

    python scripts/collect_labels.py --journal <workflow journal.jsonl> --out dev/labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.base import Event, merge_same_class  # noqa: E402

DURATION = {"C3896": 340.34, "C3897": 317.82, "C3902": 317.82, "C3905": 127.63}
FPS = 29.97
CONF_RANK = {"low": 0, "medium": 1, "high": 2}
POLICY_DROP = "BUS AT BUS STOP"


def load(journal: Path) -> tuple[list[tuple[str, dict]], dict[tuple, dict]]:
    labels, sweeps, verdicts = {}, [], {}
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    for d in rows:
        if d.get("type") == "started":
            labels[d["key"]] = d.get("label", "")
    for d in rows:
        if d.get("type") != "result" or not isinstance(d.get("result"), dict):
            continue
        lab, r = labels.get(d["key"], ""), d["result"]
        if lab.startswith(("sweep:", "long:")):
            video = lab.split(":")[1].split("@")[0]
            sweeps += [(video, ev) for ev in r.get("events", [])]
        elif lab.startswith("verify:"):
            _, video, rest = lab.split(":", 2)
            label, start = rest.rsplit("@", 1)
            verdicts[(video, label, float(start))] = r
    return sweeps, verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", default="dev/labels.json")
    ap.add_argument("--details", default="dev/labels_details.json")
    ap.add_argument("--min-conf", default="medium", choices=list(CONF_RANK))
    ap.add_argument("--extra", nargs="*", default=sorted(str(p) for p in Path("dev").glob("fp_check_*.json")),
                    help="verified detector candidates (scripts/verify workflow output) to add when real")
    args = ap.parse_args()
    sweeps, verdicts = load(Path(args.journal))
    per_video: dict[str, list[Event]] = {v: [] for v in DURATION}
    details = []
    for video, ev in sweeps:
        if POLICY_DROP in ev.get("description", ""):
            continue   # policy: a bus dwelling at its stop is normal operation, not a stopped vehicle
        v = verdicts.get((video, ev["label"], float(ev["start"])))
        details.append({"video": video, "first": ev, "verdict": v})
        if v is not None:
            if v["is_real"] and CONF_RANK.get(v["confidence"], 0) >= CONF_RANK[args.min_conf]:
                per_video[video].append(Event(float(v["start"]), float(v["end"]), v["label"]))
        elif CONF_RANK.get(ev["confidence"], 0) >= CONF_RANK[args.min_conf]:
            per_video[video].append(Event(float(ev["start"]), float(ev["end"]), ev["label"]))
    for f in args.extra:
        for c in json.loads(Path(f).read_text()):
            video = c["key"].split(":")[1]
            if c["is_real"] and CONF_RANK.get(c["confidence"], 0) >= CONF_RANK[args.min_conf]:
                per_video[video].append(Event(float(c["start"]), float(c["end"]), c["label"]))
                details.append({"video": video, "first": None, "verdict": c, "source": f})
    gt = {}
    for video, evs in per_video.items():
        evs = [e for e in evs if e.end > e.start]
        gt[f"{video}.MP4"] = {"duration": DURATION[video], "fps": FPS,
                              "events": [[round(e.start, 2), round(min(e.end, DURATION[video]), 2), e.label]
                                         for e in merge_same_class(evs)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gt, indent=1))
    Path(args.details).write_text(json.dumps(details, indent=1))
    n_ver = sum(d["verdict"] is not None for d in details)
    n_rej = sum(d["verdict"] is not None and not d["verdict"]["is_real"] for d in details)
    print(f"{len(details)} first-pass events, {n_ver} verified ({n_rej} rejected) -> {args.out}")
    for v, d in gt.items():
        print(f"  {v}: {len(d['events'])} events", sorted({e[2] for e in d["events"]}))


if __name__ == "__main__":
    main()
