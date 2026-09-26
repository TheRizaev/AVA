"""Global settings shared by Part A, Part B and the tooling."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
SCENE_FILE = ROOT / "assets" / "layout.json"
SCENE_REF_IMAGE = ROOT / "assets" / "reference.jpg"

SEED = 0
VERBOSE = True

# Classes Part A reports. A class we predict that never occurs in the test set
# costs a full class of macro-F1, so only rules validated on the dev labels are on.
# Decisions from the dev labels (see dev/decisions.md):
#  near_miss      - hard braking is everywhere at this junction (queues, yielding); every detection was
#                   rejected by the annotators -> off
#  illegal_u_turn - U-turns around the median tip are frequent and nothing in view prohibits them;
#                   the annotators did not accept them as illegal -> off
#  illegal_turn   - on: after independent re-checks, right turns into the side street from a lane
#                   other than the right-turn lane remain in the dev labels (dev F1 ~0.5, and
#                   switching it off costs the whole class)
ENABLED_CLASSES: tuple[str, ...] | None = (
    "red_light", "stop_line", "jaywalking", "failure_to_yield", "wrong_way", "stopped_vehicle",
    "solid_line_crossing", "illegal_turn", "congestion", "road_obstacle", "fire_smoke", "accident",
)

# ---- decoding / analysis resolution ----------------------------------------
ANALYSIS_SIZE = (1920, 1080)   # all geometry (layout, tracks) lives in this frame size
SAMPLE_FPS = 10.0              # reference frames of an IBBP GOP at 29.97 fps

# ---- detector ----------------------------------------------------------------
DETECTOR_WEIGHTS = WEIGHTS_DIR / "yolo26l.pt"
DET_IMGSZ = 1280
DET_BATCH = 8
DET_CONF = 0.10                # low threshold: ByteTrack uses the 0.1-0.25 band for its second pass
# COCO ids: person, bicycle, car, motorcycle, bus, truck, + animals for road_obstacle
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
ANIMALS = (15, 16, 17, 18, 19)  # cat, dog, horse, sheep, cow
VEHICLES = (CAR, MOTORCYCLE, BUS, TRUCK)
DET_CLASSES = (PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK) + ANIMALS

# ---- tracker (ByteTrack) -----------------------------------------------------
TRACKER_ARGS = dict(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.10,
    new_track_thresh=0.30,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)
TRACK_BUFFER_SEC = 3.0         # keep lost tracks this long (occlusion by buses, gantry)
