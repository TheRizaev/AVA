"""Run Part A twice in separate processes on one video and compare the events.

    python scripts/check_determinism.py samples/C3905.MP4
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = "import json, sys; sys.path.insert(0, r'{root}'); import solution; print('EVENTS=' + json.dumps(solution.detect_events(r'{video}')))"


def run(video: str) -> list:
    out = subprocess.run([sys.executable, "-c", CODE.format(root=ROOT, video=video)], capture_output=True, text=True, cwd=ROOT)
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("EVENTS="))
    return json.loads(line[len("EVENTS="):])


def main() -> None:
    video = sys.argv[1]
    a, b = run(video), run(video)
    same = len(a) == len(b) and all(x[2] == y[2] and abs(x[0] - y[0]) < 1e-3 and abs(x[1] - y[1]) < 1e-3 for x, y in zip(a, b))
    print(f"run 1: {len(a)} events, run 2: {len(b)} events -> {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        print("only in run 1:", [x for x in a if x not in b][:10])
        print("only in run 2:", [x for x in b if x not in a][:10])


if __name__ == "__main__":
    main()
