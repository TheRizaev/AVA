"""Event rules on synthetic scenes: identity registration, hand-made tracks and signal timelines.

No video, detector or GPU is needed:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import VideoAnalysis  # noqa: E402
from src.events.base import Context, Event, clip_events  # noqa: E402
from src.events.pedestrian_rules import jaywalking  # noqa: E402
from src.events.signal_rules import red_light, stop_line  # noqa: E402
from src.events.trajectory_rules import wrong_way  # noqa: E402
from src.signals import GREEN, RED, SignalTimeline  # noqa: E402
from src.tracks import per_track  # noqa: E402
from src.video import VideoMeta  # noqa: E402

FPS = 10.0
CAR, PERSON = 2.0, 0.0     # COCO class ids
T = np.arange(0, 30, 1 / FPS)


def scene(path: np.ndarray, state: np.ndarray, cls: float = CAR, size: tuple[float, float] = (80.0, 50.0)) -> Context:
    """One road user whose box bottom-centre follows `path`; the camera frame is the reference frame."""
    w, h = size
    rows = np.column_stack([np.round(T * FPS), T, np.ones(len(T)),
                            path[:, 0] - w / 2, path[:, 1] - h, path[:, 0] + w / 2, path[:, 1],
                            np.full(len(T), 0.9), np.full(len(T), cls)])
    meta = VideoMeta("synthetic.mp4", 1920, 1080, 30.0, 900, 30.0)
    va = VideoAnalysis(meta, rows, np.array([0.0]), np.eye(3)[None], np.array([True]), np.zeros(0), np.zeros((0, 5)),
                       np.zeros(0), np.zeros((0, 360, 640, 3), np.uint8), np.empty((0, 7)))
    return Context(va, list(per_track(rows, np.eye(3))), SignalTimeline(T, state, np.zeros((len(T), 5))))


def move(a: tuple, b: tuple, t0: float, t1: float) -> np.ndarray:
    """Stand at a until t0, move at constant speed to b by t1, stand there."""
    f = np.clip((T - t0) / (t1 - t0), 0, 1)[:, None]
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * f


def walker(path: np.ndarray) -> Context:
    return scene(path, np.full(len(T), RED), cls=PERSON, size=(30.0, 80.0))


def test_red_light_fires_when_crossing_on_red_and_not_on_green():
    path = move((500, 420), (700, 720), 8.0, 14.0)    # approach -> over the stop line -> through cw1
    events = red_light(scene(path, np.where(T < 20, RED, GREEN)))
    assert len(events) == 1 and 8.0 < events[0].start < 14.0
    assert red_light(scene(path, np.full(len(T), GREEN))) == []


def test_stop_line_fires_for_a_car_waiting_past_the_line_on_red():
    path = move((500, 440), (560, 520), 2.0, 5.0)     # stops between the stop line and cw1
    events = stop_line(scene(path, np.where(T < 25, RED, GREEN)))
    assert len(events) == 1 and events[0].end <= 25.1


def test_wrong_way_fires_against_the_approach_flow_only():
    against = move((780, 470), (380, 260), 5.0, 10.0)   # approach traffic runs top-left -> bottom-right
    events = wrong_way(scene(against, np.full(len(T), GREEN)))
    assert len(events) == 1 and 5.0 <= events[0].start < 7.0
    assert wrong_way(scene(against[::-1], np.full(len(T), GREEN))) == []


def test_jaywalking_on_the_carriageway_but_not_on_a_zebra_or_the_median():
    assert len(jaywalking(walker(move((560, 330), (700, 250), 5.0, 12.0)))) == 1   # across the approach
    assert jaywalking(walker(move((420, 638), (1070, 558), 5.0, 15.0))) == []      # along cw1
    assert jaywalking(walker(move((750, 380), (800, 380), 5.0, 12.0))) == []       # on the median


def test_clip_events_trims_to_the_video_and_drops_slivers():
    out = clip_events([Event(-1.0, 3.0, "congestion"), Event(29.9, 40.0, "congestion"), Event(5.0, 5.1, "jaywalking")],
                      duration=30.0)
    assert [(e.start, e.end, e.label) for e in out] == [(0.0, 3.0, "congestion")]


def test_interface_classes_match_the_starter_kit():
    import solution
    from src import config
    assert solution.CLASSES == ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
                                "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
                                "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]
    assert set(config.ENABLED_CLASSES) <= set(solution.CLASSES)
