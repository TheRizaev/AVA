"""Run Part A twice in separate processes on one video and compare the events exactly.

    python scripts/check_determinism.py samples/C3905.MP4

Exits with code 1 when the two runs differ or a run fails.
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
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("EVENTS=")), None)
    if out.returncode != 0 or line is None:
        sys.stderr.write(out.stderr)
        sys.exit(f"Part A run failed (exit code {out.returncode})")
    return json.loads(line[len("EVENTS="):])


def main() -> None:
    video = sys.argv[1]
    a, b = run(video), run(video)
    same = a == b
    print(f"run 1: {len(a)} events, run 2: {len(b)} events -> {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        print("only in run 1:", [x for x in a if x not in b][:10])
        print("only in run 2:", [x for x in b if x not in a][:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
