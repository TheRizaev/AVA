"""Traffic-event detection for the WIUT CV track (see README.md)."""
import os

# The evaluation machine has no internet: never let Ultralytics try to download,
# auto-install packages or ping hosts. Must be set before ultralytics is imported.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_VERBOSE", "False")
