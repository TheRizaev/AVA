"""Replay cached track tables through the Part B risk logic (no detector needed).

Prints the distribution of the risk and its cues on the sample videos and the
top moments, so thresholds can be set so ordinary traffic stays below 0.5.

    python scripts/tune_risk.py --cache work/cache [--top 15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import VideoAnalysis  # noqa: E402
from src.risk import OnlineRisk  # noqa: E402


def replay(va: VideoAnalysis) -> dict[str, np.ndarray]:
    online = OnlineRisk()
    table = va.tracks
    frames = np.unique(table[:, 0])
    order = np.argsort(table[:, 0], kind="stable")
    table = table[order]
    starts = np.searchsorted(table[:, 0], frames)
    ends = np.append(starts[1:], len(table))
    out = {"t": [], "score": [], "raw": [], "cross": [], "rear": [], "ped": [], "brake": [], "pair": []}
    for f, s, e in zip(frames, starts, ends):
        rows = table[s:e]
        t = float(rows[0, 1])
        H = va.reg_H[va.H_index(np.array([t]))[0]]
        # cached rows: frame, t, id, x1, y1, x2, y2, score, cls -> tracker layout
        tracks = np.column_stack([rows[:, 3:7], rows[:, 2], rows[:, 7], rows[:, 8]])
        score = online.observe(t, tracks, H)
        out["t"].append(t)
        out["score"].append(score)
        out["raw"].append(online.last_raw)
        for k in ("cross", "rear", "ped", "brake", "pair"):
            out[k].append(online.last_cues.get(k, 0.0 if k != "pair" else None))
    return {k: (np.array(v) if k != "pair" else v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    for f in sorted(Path(args.cache).glob("*.npz")):
        r = replay(VideoAnalysis.load(f))
        s = r["score"]
        pct = {p: round(float(np.percentile(s, p)), 3) for p in (50, 90, 99, 99.9)}
        cue_pct = {k: round(float(np.percentile(r[k], 99)), 3) for k in ("cross", "rear", "ped", "brake")}
        print(f"{f.stem}: score pct {pct}  frac>=0.5 {np.mean(s >= 0.5):.4f}  cue p99 {cue_pct}")
        order = np.argsort(-r["raw"])
        shown, last = 0, []
        for i in order:
            if shown >= args.top:
                break
            if any(abs(r["t"][i] - x) < 3 for x in last):
                continue
            last.append(r["t"][i])
            shown += 1
            print(f"   t={r['t'][i]:7.2f} raw={r['raw'][i]:.2f} cross={r['cross'][i]:.2f} rear={r['rear'][i]:.2f} "
                  f"ped={r['ped'][i]:.2f} brake={r['brake'][i]:.2f} pair={r['pair'][i]}")


if __name__ == "__main__":
    main()
