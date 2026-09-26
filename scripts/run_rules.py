"""Run the event rules on cached analyses and write a predictions.json.

    python scripts/run_rules.py --cache work/cache --out work/pred_dev.json [--verbose]
    python evaluate.py --pred work/pred_dev.json --gt dev/labels.json --per-video
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import VideoAnalysis  # noqa: E402
from src.events import build_context, detect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--out", default="work/pred_dev.json")
    ap.add_argument("--only", nargs="*", help="restrict to these labels")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    videos = {}
    for f in sorted(Path(args.cache).glob("*.npz")):
        ctx = build_context(VideoAnalysis.load(f))
        events = detect(ctx, tuple(args.only) if args.only else None)
        videos[f"{f.stem}.MP4"] = {"events": [e.as_list() for e in events], "risk": []}
        print(f"{f.stem}: {len(events)} events")
        if args.verbose:
            for e in events:
                print(f"   {e.start:7.2f} {e.end:7.2f} {e.label:<18} {e.info}")
    Path(args.out).write_text(json.dumps({"team": "dev", "videos": videos}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
