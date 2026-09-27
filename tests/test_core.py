"""Unit tests for the pure-logic parts of the pipeline:  python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.base import Event, fill_short_gaps, merge_same_class, runs_of_true  # noqa: E402
from src.scene import lane_angle, load_layout, register  # noqa: E402
from src.signals import GREEN, RED, UNKNOWN, YELLOW, _drop_yellow_islands, _resolve_gaps  # noqa: E402


def test_runs_and_gap_filling():
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert runs_of_true(flags) == [(1, 2), (5, 5), (7, 9)]
    t = np.arange(10) * 0.1
    assert runs_of_true(fill_short_gaps(t, flags, max_gap=0.25)) == [(1, 2), (5, 9)]   # the 0.3 s gap stays
    assert runs_of_true(fill_short_gaps(t, flags, max_gap=0.3)) == [(1, 9)]


def test_merge_same_class_unions_overlaps_only_within_a_class():
    ev = [Event(0, 5, "jaywalking"), Event(4, 8, "jaywalking"), Event(10, 12, "jaywalking"), Event(3, 6, "red_light")]
    out = merge_same_class(ev)
    assert [(e.start, e.end, e.label) for e in out] == [(0, 8, "jaywalking"), (3, 6, "red_light"), (10, 12, "jaywalking")]


def test_flashing_green_and_occlusion_are_resolved():
    t = np.arange(0, 12, 0.1)
    state = np.full(len(t), GREEN)
    state[30:36] = UNKNOWN          # flashing-green off phase (0.6 s)
    state[60:75] = YELLOW           # yellow island inside green = occlusion artefact
    state[100:] = RED
    fixed = _drop_yellow_islands(t, _resolve_gaps(t, state))
    assert (fixed[:100] == GREEN).all() and (fixed[100:] == RED).all()


def test_lane_angle_separates_the_approach_lanes():
    lines = [ln["angle"] for ln in load_layout()["solid_lines"]["lines"]]
    centres = load_layout()["solid_lines"]["lane_centres"]
    assert sorted(lines) == sorted(set(lines))
    for c in centres[:-1]:
        assert all(abs(c - ln) > 0.5 for ln in lines)
    vp = np.array(load_layout()["road_vp"]["xy"])
    pt = vp + 1000 * np.array([np.cos(np.radians(27.0)), np.sin(np.radians(27.0))])
    assert abs(lane_angle(pt[None])[0] - 27.0) < 1e-6


def test_reference_registers_to_identity():
    ref = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets" / "reference.jpg"))
    H, n, ok = register(ref)
    assert ok and n > 300
    corners = np.float32([[0, 0], [1919, 0], [0, 1079], [1919, 1079]]).reshape(-1, 1, 2)
    assert np.abs(cv2.perspectiveTransform(corners, H) - corners).max() < 2.0


def test_risk_model_features_and_probability():
    """Base features are finite for a toy scene, the context has the documented size, and a
    one-hidden-layer model maps any input to a probability with a monotone calibration."""
    from src.risk import overlap_time
    from src.risk_model import BASE_FEATURES, FEATURE_NAMES, FeatureHistory, RiskModel, base_features
    xy = np.array([[100.0, 100.0], [300.0, 100.0], [200.0, 300.0]])
    v = np.array([[200.0, 0.0], [-200.0, 0.0], [0.0, 0.0]])
    half = np.array([[20.0, 10.0], [20.0, 10.0], [5.0, 4.0]])
    widths = np.array([57.0, 57.0, 10.0])
    kinds = np.array(["vehicle", "vehicle", "person"])
    base = base_features(xy, v, half, widths, kinds, {"cross": 0.5}, 0.3, v * 0.9, overlap_time)
    assert base.shape == (len(BASE_FEATURES),) and np.isfinite(base).all()
    assert base[BASE_FEATURES.index("any_ttc")] > 0          # the two cars close head-on
    hist = FeatureHistory()
    for t in (0.0, 0.1, 0.2):
        x = hist.push(t, base)
    assert x.shape == (len(FEATURE_NAMES),)
    rng = np.random.default_rng(0)
    n = len(FEATURE_NAMES)
    model = RiskModel({"mean": [0.0] * n, "std": [1.0] * n, "weights": rng.normal(size=4).tolist(), "bias": 0.1,
                       "alarm_p": 0.8, "hidden": {"w": rng.normal(size=(n, 4)).tolist(), "b": [0.0] * 4}})
    p = model.prob(x)
    assert 0.0 <= p <= 1.0
    assert model.calibrated(0.8) == 0.5 and model.calibrated(0.2) < model.calibrated(0.9)


def test_shipped_risk_model_loads_and_scores_stay_off_the_threshold():
    """The committed model file matches the features this code computes (a mismatch would silently
    fall back to the hand-made cues), and sub-threshold scores stay clear of 0.5 after the
    harness rounds them to 4 decimals."""
    from src.risk import OnlineRisk
    from src.risk_model import ALARM, RiskModel
    model = RiskModel.load()
    assert model is not None and model.combine == "model" and len(model.tcns) == 5
    online = OnlineRisk()
    rng = np.random.default_rng(1)
    for i in range(300):
        t = i / 10
        boxes = [[100 + 40 * t + 30 * k, 400 + 50 * k, 180 + 40 * t + 30 * k, 450 + 50 * k, k, 0.9, 2] for k in range(4)]  # cars
        tracks = np.array(boxes) + rng.normal(0, 1.0, (4, 7)) * [1, 1, 1, 1, 0, 0, 0]
        online.observe(t, tracks, np.eye(3), False)
        s = online.report()
        assert 0.0 <= s <= 1.0 and not (ALARM - 1e-4 <= s < ALARM)

