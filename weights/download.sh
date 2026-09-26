#!/usr/bin/env bash
# Fetch / rebuild the model weights (run once, with internet, before the offline evaluation).
# The repository already ships them; this script only restores missing files.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
for f in yolo26l.pt yolo26s.pt; do
  [ -f "$f" ] || curl -L --fail -o "$f" "$BASE/$f"
done
if [ ! -f yoloe26l_hazards.pt ]; then
  # YOLOE-26L with our hazard prompts baked in (needs the MobileCLIP text encoder once)
  cd .. && python scripts/make_hazard_weights.py
fi
echo "weights ready"
