"""Render annotated versions of the sample videos (website 'Results' section).

    python scripts/render_samples.py --cache work/cache --proxy work/proxy --pred predictions_samples.json --out website/media
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.analysis import VideoAnalysis  # noqa: E402
from src.events import build_context, detect  # noqa: E402
from src.render import Renderer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--proxy", default="work/proxy")
    ap.add_argument("--pred", default=None, help="predictions.json to take the risk curve from")
    ap.add_argument("--out", default="work/render")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text())["videos"] if args.pred else {}
    for f in sorted(Path(args.cache).glob("*.npz")):
        if args.only and f.stem not in args.only:
            continue
        va = VideoAnalysis.load(f)
        events = detect(build_context(va), config.ENABLED_CLASSES)
        risk = pred.get(f"{f.stem}.MP4", {}).get("risk")
        out = Renderer(va, events, risk).render(Path(args.proxy) / f"{f.stem}.mp4", Path(args.out) / f"{f.stem}_annotated.mp4",
                                                fps_out=args.fps)
        print("rendered", out)


if __name__ == "__main__":
    main()
