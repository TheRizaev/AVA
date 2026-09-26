"""Score Part B (and the accident rule) on third-party CCTV clips with our accident timings.

    python scripts/eval_external.py --labels work/ext_annot/labels.json --cache work/ext_cache

labels.json: {"<clip>.mp4": {"case": "collision"|"aftermath"|"unclear"|"normal", "start": s, "end": e,
                             "extra": [[s2, e2], ...]}}   (extra: replays of the crash in edited clips)
Clips whose collision is not visible (aftermath / unclear) are left out. Normal clips are
all-negative. The risk curve is a causal replay of the cached online tracker rows through
the current src/risk.py (see scripts/ext_cache.py), scored with the official evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
from src import risk  # noqa: E402


def replay(path: Path) -> list[list[float]]:
    d = np.load(path)
    online = risk.OnlineRisk(float(d["fps"]))
    rows, t, H, fit = d["rows"], d["t"], d["H"], d["have_fit"]
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    starts = np.searchsorted(rows[:, 0], np.arange(len(t)))
    ends = np.searchsorted(rows[:, 0], np.arange(len(t)), side="right")
    curve = []
    for k in range(len(t)):
        tracks = rows[starts[k]:ends[k], 1:8]
        online.observe(float(t[k]), tracks, H[k], bool(fit[k]))
        curve.append([round(float(t[k]), 3), round(online.report(), 4)])
    return curve


def per_frame(curve: list[list[float]], fps: float, n_frames: int) -> list[list[float]]:
    """Hold each analysed sample over the stride, one sample per decoded frame, as the harness records it."""
    stride = max(1, int(round(fps / risk.RISK_FPS)))
    out, last = [], 0.0
    for idx in range(n_frames):
        if idx % stride == 0 and idx // stride < len(curve):
            last = curve[idx // stride][1]
        out.append([round(idx / fps, 4), last])
    return out


def load_labels(path: Path, cache: Path) -> dict:
    labels = json.loads(path.read_text()) if path.exists() else {}
    for f in cache.glob("*.risk.npz"):
        name = f.name.replace(".risk.npz", ".mp4")
        if "_norm_" in name and name not in labels:
            labels[name] = {"case": "normal"}
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="work/ext_annot/labels.json")
    ap.add_argument("--cache", default="work/ext_cache")
    ap.add_argument("--out", default="work/ext_eval.json")
    ap.add_argument("--accident-rule", action="store_true", help="also score the Part A accident rule")
    args = ap.parse_args()
    cache = Path(args.cache)
    labels = load_labels(Path(args.labels), cache)
    gt, pred, info = {}, {"team": "ext", "videos": {}}, {}
    for name, lab in sorted(labels.items()):
        f = cache / name.replace(".mp4", ".risk.npz")
        if lab["case"] not in ("collision", "normal") or not f.exists():
            continue
        d = np.load(f)
        dur = float(d["n_frames"]) / float(d["fps"])
        # extra: the same crash replayed in an edited clip. For Part B it is marked near_miss, so
        # evaluate.py ignores [a - H, b] (not a new crash to anticipate, alarms there are not false);
        # for the Part A accident rule it is an accident like the original.
        ev = ([[lab["start"], lab["end"], "accident"]] + [[a, b, "near_miss"] for a, b in lab.get("extra", [])]
              if lab["case"] == "collision" else [])
        gt[name] = {"duration": dur, "fps": float(d["fps"]), "events": ev}
        curve = per_frame(replay(f), float(d["fps"]), int(d["n_frames"]))
        pred["videos"][name] = {"events": [], "risk": curve}
        sc = np.array([c[1] for c in curve]) if curve else np.zeros(1)
        info[name] = {"case": lab["case"], "max": float(sc.max()),
                      "alarms": evaluate.alarm_starts(curve), "start": lab.get("start")}
    b = evaluate.evaluate_part_b(gt, pred["videos"])
    n_col = sum(1 for v in info.values() if v["case"] == "collision")
    n_norm = sum(1 for v in info.values() if v["case"] == "normal")
    minutes = sum(g["duration"] for g in gt.values()) / 60
    print(f"clips: {n_col} with a visible collision, {n_norm} normal ({minutes:.1f} min)")
    if b:
        print(f"Score_B {b['score_b']:.3f}  AP {b['ap']:.3f}  alarm P/R/F1 {b['alarm_precision']:.3f}/"
              f"{b['alarm_recall']:.3f}/{b['f1_alarm']:.3f}  mTTA {b['mtta_sec']:.2f}s  alarms {b['n_alarms']}")
    fa_norm = sum(len(v["alarms"]) for v in info.values() if v["case"] == "normal")
    norm_min = sum(gt[n]["duration"] for n, v in info.items() if v["case"] == "normal") / 60
    if norm_min:
        print(f"false alarms on normal clips: {fa_norm} in {norm_min:.1f} min ({fa_norm / norm_min:.2f}/min)")
    report = {"part_b": b, "clips": info}
    if args.accident_rule:
        from src.analysis import VideoAnalysis
        from src.events import build_context, detect
        for name in gt:
            f = cache / name.replace(".mp4", ".npz")
            if f.exists():
                ctx = build_context(VideoAnalysis.load(f))
                pred["videos"][name]["events"] = [e.as_list() for e in detect(ctx, ("accident",))]
        gt_a = {n: {**g, "events": [[s, e, "accident"] for s, e, _ in g["events"]]} for n, g in gt.items()}
        a = evaluate.evaluate_part_a(gt_a, pred["videos"])
        pc = a["per_class"].get("accident")
        if pc:
            print("accident rule: F1@0.3/0.5/0.7 = " + "/".join(f"{pc[k]['f1']:.2f}" for k in ("0.3", "0.5", "0.7")) +
                  f"  TP/FP/FN@0.3 = {pc['0.3']['tp']}/{pc['0.3']['fp']}/{pc['0.3']['fn']}")
        report["part_a_accident"] = pc
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, v in sorted(info.items()):
        if v["case"] == "collision":
            print(f"  {name:40s} start {v['start']:6.2f}  max {v['max']:.2f}  alarms {v['alarms']}")


if __name__ == "__main__":
    main()
