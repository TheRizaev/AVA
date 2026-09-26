"""Event rules: each takes a Context and returns a list of Events of one class."""
from __future__ import annotations

from ..analysis import VideoAnalysis
from .base import Context, Event, clip_events, merge_same_class
from .collision_rules import accident, near_miss
from .flow_rules import congestion, stopped_vehicle
from .hazard_rules import fire_smoke, road_obstacle
from .pedestrian_rules import failure_to_yield, jaywalking
from .signal_rules import red_light, stop_line
from .trajectory_rules import illegal_turn, illegal_u_turn, solid_line_crossing, wrong_way

# label -> (rule, same-class merge gap in seconds)
RULES = {
    "red_light": (red_light, 0.0),
    "stop_line": (stop_line, 0.0),
    "jaywalking": (jaywalking, 6.0),
    "failure_to_yield": (failure_to_yield, 2.0),
    "wrong_way": (wrong_way, 1.0),
    "illegal_u_turn": (illegal_u_turn, 0.0),
    "stopped_vehicle": (stopped_vehicle, 2.0),
    "solid_line_crossing": (solid_line_crossing, 0.0),
    "road_obstacle": (road_obstacle, 2.0),
    "fire_smoke": (fire_smoke, 4.0),
    "accident": (accident, 0.0),
    "near_miss": (near_miss, 0.0),
    "congestion": (congestion, 3.0),
    "illegal_turn": (illegal_turn, 0.0),
}


# Rules that do not depend on the calibrated scene layout (usable on footage from other cameras).
LAYOUT_FREE = ("accident", "near_miss")


def build_context(analysis: VideoAnalysis) -> Context:
    return Context(analysis, analysis.track_list(), analysis.signal_timeline())


def detect(ctx: Context, enabled: tuple[str, ...] | None = None) -> list[Event]:
    events: list[Event] = []
    for label, (rule, gap) in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        if not ctx.analysis.scene_recognised and label not in LAYOUT_FREE:
            continue
        found = clip_events(rule(ctx), ctx.duration)
        events += merge_same_class(found, gap=gap)
    return sorted(events, key=lambda e: (e.start, e.label))
