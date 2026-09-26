"""Annotated-video renderer: boxes, trails, scene layout, signal state, events, risk.

Used for the website's sample-video results and by the live demo. Frames come
from any decodable video (the original 4K file or a 1080p proxy); tracks are
drawn from the nearest analysed frame at or before each output frame.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config
from .analysis import VideoAnalysis
from .scene import load_layout, warp_points
from .signals import GREEN, RED, YELLOW
from .tracks import KIND_BY_CLS

# BGR; the same palette as the website (website/assets/js/palette.js, light theme)
CLASS_COLORS = {
    "failure_to_yield": (214, 120, 42),
    "near_miss": (52, 104, 235),
    "jaywalking": (122, 175, 27),
    "illegal_u_turn": (0, 161, 237),
    "stopped_vehicle": (164, 123, 232),
    "solid_line_crossing": (0, 131, 0),
    "red_light": (167, 58, 74),
    "accident": (72, 73, 227),
    "stop_line": (196, 155, 14),
    "illegal_turn": (0, 123, 154),
    "wrong_way": (110, 48, 176),
    "congestion": (205, 90, 106),
    "road_obstacle": (31, 143, 111),
    "fire_smoke": (59, 59, 163),
}
KIND_COLORS = {"vehicle": (255, 190, 60), "person": (80, 255, 120), "bicycle": (255, 120, 255), "animal": (0, 160, 255)}
SIGNAL_COLORS = {RED: (40, 40, 240), YELLOW: (0, 200, 255), GREEN: (80, 220, 80)}


def _event_tracks(info: dict) -> set[int]:
    """Track ids mentioned in an Event.info (including merged sub-events)."""
    out: set[int] = set()
    for key in ("track", "with", "tracks"):
        v = info.get(key)
        if isinstance(v, (int, np.integer)):
            out.add(int(v))
        elif isinstance(v, list):
            out.update(int(x) for x in v if isinstance(x, (int, np.integer)))
    for sub in info.get("merged", []):
        out |= _event_tracks(sub)
    return out


def _ffmpeg_writer(path: Path, size: tuple[int, int], fps: float) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Renderer:
    def __init__(self, analysis: VideoAnalysis, events: list, risk: list[list] | None = None,
                 out_size: tuple[int, int] = (1280, 720)) -> None:
        """events: [start, end, label] lists, or Event objects (their track ids get highlighted)."""
        self.va = analysis
        norm = []
        for e in events:
            if isinstance(e, (list, tuple)):
                norm.append((float(e[0]), float(e[1]), str(e[2]), set()))
            else:
                norm.append((e.start, e.end, e.label, _event_tracks(e.info)))
        self.events = sorted(norm, key=lambda e: e[0])
        self.risk = np.array(risk) if risk else np.zeros((0, 2))
        self.out_size = out_size
        self.signal = analysis.signal_timeline()
        tab = analysis.tracks
        self.frames = np.unique(tab[:, 0]) if len(tab) else np.array([])
        order = np.argsort(tab[:, 0], kind="stable")
        self.tab = tab[order]
        self.starts = np.searchsorted(self.tab[:, 0], self.frames)
        self.ends = np.append(self.starts[1:], len(self.tab))
        self.trails: dict[int, list[tuple[float, float]]] = {}
        self.layout = load_layout()

    # -- drawing helpers -----------------------------------------------------------
    def _rows_at(self, frame_idx: int) -> np.ndarray:
        k = int(np.searchsorted(self.frames, frame_idx, side="right") - 1)
        if k < 0:
            return np.empty((0, 9))
        return self.tab[self.starts[k]:self.ends[k]]

    def _draw_layout(self, img: np.ndarray, t: float, sx: float, sy: float) -> None:
        Hinv = np.linalg.inv(self.va.reg_H[self.va.H_index(np.array([t]))[0]])
        over = img.copy()
        for name in ("cw1", "cw2", "cw3"):
            p = warp_points(Hinv, np.array(self.layout["polygons"][name], float)) * [sx, sy]
            cv2.fillPoly(over, [p.astype(np.int32)], (255, 255, 255))
        cv2.addWeighted(over, 0.12, img, 0.88, 0, img)
        a, b = warp_points(Hinv, np.array(self.layout["lines"]["stop_line"], float)) * [sx, sy]
        state = int(self.signal.at(t)[0])
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), SIGNAL_COLORS.get(state, (200, 200, 200)), 2)

    def _draw_tracks(self, img: np.ndarray, rows: np.ndarray, t: float, sx: float, sy: float,
                     active_tracks: set[int]) -> None:
        for _f, _t, tid, x1, y1, x2, y2, _score, cls in rows:
            kind = KIND_BY_CLS.get(int(cls), "other")
            if kind == "other":
                continue
            tid = int(tid)
            col = (40, 40, 255) if tid in active_tracks else KIND_COLORS.get(kind, (200, 200, 200))
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(img, p1, p2, col, 2 if tid in active_tracks else 1)
            trail = self.trails.setdefault(tid, [])
            ground = ((x1 + x2) / 2 * sx, y2 * sy)
            if not trail or trail[-1] != ground:
                trail.append(ground)
                del trail[:-25]
            if len(trail) > 1:
                cv2.polylines(img, [np.array(trail, np.int32)], False, col, 1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, t: float) -> None:
        w, h = self.out_size
        state = int(self.signal.at(t)[0])
        cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.circle(img, (18, 17), 9, SIGNAL_COLORS.get(state, (120, 120, 120)), -1)
        cv2.putText(img, f"t={t:6.1f}s", (36, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        active = [e for e in self.events if e[0] <= t <= e[1]]
        x = 150
        for s, e, label, _ in active:
            col = CLASS_COLORS.get(label, (255, 255, 255))
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x, 6), (x + tw + 12, 28), col, -1)
            cv2.putText(img, label, (x + 6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            x += tw + 20
        # timeline strip with event segments, playhead and risk curve
        dur = self.va.meta.duration
        y0 = h - 46
        cv2.rectangle(img, (0, y0), (w, h), (20, 20, 20), -1)
        for s, e, label, _ in self.events:
            xa, xb = int(s / dur * w), max(int(e / dur * w), int(s / dur * w) + 2)
            cv2.rectangle(img, (xa, y0 + 4), (xb, y0 + 18), CLASS_COLORS.get(label, (255, 255, 255)), -1)
        if len(self.risk):
            xs = (self.risk[::10, 0] / dur * w).astype(int)
            ys = (h - 4 - self.risk[::10, 1] * 22).astype(int)
            cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, (0, 170, 255), 1, cv2.LINE_AA)
            cv2.line(img, (0, h - 15), (w, h - 15), (70, 70, 70), 1)
        px = int(t / dur * w)
        cv2.line(img, (px, y0), (px, h), (255, 255, 255), 1)

    # -- main loop -------------------------------------------------------------------
    def render(self, video: str | Path, out: str | Path, fps_out: float | None = None) -> Path:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.va.meta.fps
        step = max(1, int(round(fps / fps_out))) if fps_out else 1
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # track boxes are in ANALYSIS_SIZE pixels; frames are resized to out_size
        sx = self.out_size[0] / config.ANALYSIS_SIZE[0]
        sy = self.out_size[1] / config.ANALYSIS_SIZE[1]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = _ffmpeg_writer(out, self.out_size, fps / step)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / fps
                img = cv2.resize(frame, self.out_size, interpolation=cv2.INTER_AREA) if (w0, h0) != self.out_size else frame
                active = set().union(*[e[3] for e in self.events if e[0] <= t <= e[1] + 0.5] or [set()])
                self._draw_layout(img, t, sx, sy)
                self._draw_tracks(img, self._rows_at(int(round(t * self.va.meta.fps))), t, sx, sy, active)
                self._draw_hud(img, t)
                writer.stdin.write(img.tobytes())
            idx += 1
        cap.release()
        writer.stdin.close()
        writer.wait()
        return out
