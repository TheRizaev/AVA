"""Build weights/yoloe26l_hazards.pt: YOLOE-26L with the hazard text prompts baked in.

Run once with internet (downloads yoloe-26l-seg.pt and the MobileCLIP text
encoder); the result needs no text encoder at inference time.

    python scripts/make_hazard_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hazards import HAZARD_CLASSES, HAZARD_WEIGHTS  # noqa: E402


def main() -> None:
    import os
    from ultralytics import YOLOE
    os.chdir(HAZARD_WEIGHTS.parent)     # yoloe-26l-seg.pt and the MobileCLIP encoder are fetched into weights/
    model = YOLOE("yoloe-26l-seg.pt")
    names = list(HAZARD_CLASSES)
    model.set_classes(names, model.get_text_pe(names))
    model.save(str(HAZARD_WEIGHTS))
    print("saved", HAZARD_WEIGHTS, YOLOE(str(HAZARD_WEIGHTS)).names)


if __name__ == "__main__":
    main()
