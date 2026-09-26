"""Part B: causal accident-risk estimator.

The harness feeds every frame in order; we analyse every STRIDE-th one (a fixed
stride keeps the output deterministic) with the same detector as Part A, an
online ByteTrack, and periodic registration to the reference layout. The risk
at time t uses only what has been observed up to t. Cues:

* crossing conflict - two moving vehicles on paths that differ by more than
  CROSS_DEG whose footprints are predicted to overlap within the horizon;
* rear-end conflict - a vehicle closing fast on another one in the same lane
  (lane identity = angle of the ground point seen from the road's vanishing
  point, which is perspective-free because lane lines are rays from it);
* pedestrian conflict - a pedestrian on the carriageway (crossings included)
  about to be reached by a moving vehicle;
* hard braking - a sudden large speed drop of a fast vehicle.

On footage from another camera (no registration to the reference ever
succeeds) the layout is meaningless: every road user counts as on the road and
"same lane" becomes "within the two vehicles' widths of the follower's
travel line".

Each cue maps to [0, 1]; they are combined as independent evidence and
smoothed with a fast-attack / slow-release filter. Scales are calibrated on
the sample videos (no accident in them) so that ordinary traffic stays low
(scripts/tune_risk.py).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from . import config
from .detection import detect_batch, new_tracker
from .risk_model import ALARM, ALARM_CAP, BASE_FEATURES, FeatureHistory, RiskModel, base_features
from .scene import Registrar, load_layout, union_mask, warp_points
from .tracks import KIND_BY_CLS

RISK_FPS = 10.0
RISK_IMGSZ = 960
REGISTER_EVERY_SEC = 5.0
HISTORY_SEC = 2.0
HORIZON = 5.0

MOVING = 60.0            # px/s
CROSS_DEG = 40.0
CROSS_TTC_SCALE = 0.5    # s
REAR_MIN_CLOSING = 120.0  # px/s
REAR_TTC_SCALE = 0.6     # s
PED_TTC_SCALE = 0.5      # s
PED_WEIGHT = 0.6
BRAKE_DROP = 0.6
BRAKE_WINDOW = 0.8
BRAKE_MIN_SPEED = 200.0
BRAKE_WEIGHT = 0.5
BRAKE_ALL_FRAC = 0.6     # most fast vehicles "braking" in the same instant = a frozen frame / cut, not braking
JUMP_PX = 60.0           # a step longer than this and JUMP_BOXES box widths restarts the track's history
JUMP_BOXES = 1.0
FOOT_WIDTH = 0.7         # footprint half-width = FOOT_WIDTH * box width / 2
FOOT_DEPTH = 0.35        # footprint half-depth = FOOT_DEPTH * box height / 2
ATTACK, RELEASE = 0.6, 0.12
# Imminent contact: two road users whose footprints will touch within IMMINENT_TTC s while closing at
# >= IMMINENT_MIN_REL px/s, seen in IMMINENT_FRAMES consecutive analysed frames. It bypasses the
# smoothing (an alarm must start BEFORE the contact to count) and holds IMMINENT_SCORE for IMMINENT_HOLD s.
IMMINENT_TTC = 0.4
IMMINENT_MIN_REL = 300.0
IMMINENT_FRAMES = 2
IMMINENT_HOLD = 1.0
IMMINENT_SCORE = 0.75
IMMINENT_PED = False   # vehicle-pedestrian near-contacts are everyday traffic at the crossings
CLS_VOTES_KEEP = 30.0  # s: class votes of a track id unseen this long are forgotten (>> TRACK_BUFFER_SEC)
# The smoothed evidence value that is mapped to the 0.5 alarm threshold. Chosen on the
# sample videos (no accidents): ~0.2 false alarms per minute of ordinary traffic.
ALARM_POINT = 0.82


def calibrate(s: float) -> float:
    """Monotone map of the smoothed evidence to the reported score (ranking - and so AP - unchanged)."""
    if s < ALARM_POINT:
        return 0.5 * s / ALARM_POINT
    return 0.5 + 0.5 * (s - ALARM_POINT) / (1.0 - ALARM_POINT)


class _Hist:
    __slots__ = ("t", "xy", "half", "kind")

    def __init__(self, kind: str) -> None:
        self.t: deque = deque()
        self.xy: deque = deque()
        self.half: deque = deque()
        self.kind = kind

    def add(self, t: float, xy: np.ndarray, half: np.ndarray, width: float) -> None:
        if self.xy and np.linalg.norm(xy - self.xy[-1]) > max(JUMP_PX, JUMP_BOXES * width):
            self.t.clear()      # the id jumped to another object (or the footage was cut): new history
            self.xy.clear()
            self.half.clear()
        self.t.append(t)
        self.xy.append(xy)
        self.half.append(half)
        while self.t and t - self.t[0] > HISTORY_SEC:
            self.t.popleft()
            self.xy.popleft()
            self.half.popleft()

    def velocity(self, window: float = 0.6, ago: float = 0.0) -> np.ndarray | None:
        """Least-squares velocity over `window` s ending `ago` s before the last sample."""
        if len(self.t) < 3:
            return None
        t = np.array(self.t)
        end = t[-1] - ago
        m = (t >= end - window) & (t <= end + 1e-6)
        if m.sum() < 3:
            return None
        xy = np.array(self.xy)[m]
        tt = t[m] - t[m].mean()
        denom = (tt ** 2).sum()
        return None if denom <= 0 else (tt[:, None] * (xy - xy.mean(0))).sum(0) / denom

    def speed_drop(self) -> float | None:
        """Fraction of speed lost over the last BRAKE_WINDOW seconds (None: not a fast vehicle / too short)."""
        if len(self.t) < 10:
            return None
        t = np.array(self.t)
        xy = np.array(self.xy)
        sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
        ts = t[1:]
        before = sp[(ts < t[-1] - BRAKE_WINDOW) & (ts >= t[-1] - BRAKE_WINDOW - 0.6)]
        now = sp[ts >= t[-1] - 0.3]
        if len(before) < 2 or len(now) < 2:
            return None
        v0, v1 = float(np.median(before)), float(np.median(now))
        return None if v0 < BRAKE_MIN_SPEED else max(0.0, (v0 - v1) / v0)


def _axis_window(d0, dv, half):
    """Time interval during which |d0 + dv t| < half, per pair, one axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - d0) / dv
        t2 = (half - d0) / dv
    inside = np.abs(d0) < half
    lo = np.where(dv != 0, np.minimum(t1, t2), np.where(inside, -np.inf, np.inf))
    hi = np.where(dv != 0, np.maximum(t1, t2), np.where(inside, np.inf, -np.inf))
    return lo, hi


def overlap_time(xy, v, half):
    """(N, N) earliest future time at which footprints i and j start to overlap (inf if never)."""
    d0 = xy[None, :, :] - xy[:, None, :]
    dv = v[None, :, :] - v[:, None, :]
    hs = half[None, :, :] + half[:, None, :]
    lox, hix = _axis_window(d0[..., 0], dv[..., 0], hs[..., 0])
    loy, hiy = _axis_window(d0[..., 1], dv[..., 1], hs[..., 1])
    lo, hi = np.maximum(lox, loy), np.minimum(hix, hiy)
    ok = (lo < hi) & (lo > 0) & (lo < HORIZON)
    np.fill_diagonal(ok, False)
    return np.where(ok, lo, np.inf)


class _Lanes:
    """Perspective-free lane coordinate: angle of a ground point seen from the road's vanishing point."""

    def __init__(self) -> None:
        cfg = load_layout()["road_vp"]
        self.vp = np.array(cfg["xy"])
        self.split = cfg["split_deg"]
        self.width = (cfg["lane_deg_outbound"], cfg["lane_deg_approach"])

    def angle(self, xy: np.ndarray) -> np.ndarray:
        d = xy - self.vp
        return np.degrees(np.arctan2(d[..., 1], d[..., 0]))

    def lane_width(self, ang: np.ndarray) -> np.ndarray:
        return np.where(ang < self.split, self.width[0], self.width[1])


_MODEL_CACHE: list = []


def _model() -> RiskModel | None:
    if not _MODEL_CACHE:
        _MODEL_CACHE.append(RiskModel.load())
    return _MODEL_CACHE[0]


class OnlineRisk:
    """Causal risk from tracked road users; independent of how frames are obtained."""

    def __init__(self, fps: float) -> None:
        self.tracker = new_tracker(RISK_FPS)
        self.hist: dict[int, _Hist] = {}
        self.registrar = Registrar(REGISTER_EVERY_SEC)
        self.score = 0.0
        self.last_raw = 0.0
        self.last_cues: dict = {}
        self.lanes = _Lanes()
        self.road = union_mask(["approach", "outbound", "junction"]) & \
            ~union_mask(["median", "island_round", "island_tri1", "island_tri2", "island_3"])
        self.main_road = union_mask(["approach", "outbound"])
        self.layout = True      # False: footage from another camera, layout masks do not apply
        self.min_ttc = np.inf   # smallest time to contact among fast-closing pairs in the last frame
        self.base = np.zeros(len(BASE_FEATURES))
        self.features_hist = FeatureHistory()
        self.features = None     # the model's input for the last frame
        self.model = _model()
        self.stream = self.model.stream() if self.model is not None else None   # per-video model state
        self.model_p = 0.0       # smoothed model probability
        self.model_score = 0.0   # calibrated model score (after the alarm refractory)
        self.cls_votes: dict[int, dict[int, int]] = {}   # track id -> COCO class -> count (majority class)
        self.cls_seen: dict[int, float] = {}            # track id -> last time seen
        self.imm_run = 0
        self.imm_until = -np.inf
        self.imminent = False

    def update(self, img: np.ndarray, t: float, model) -> float:
        """Detect + track one analysed frame, then update the risk."""
        self.registrar.update(t, img)
        det = detect_batch(model, [img], imgsz=RISK_IMGSZ)[0]
        return self.observe(t, self.tracker.update(det, img), self.registrar.H, self.registrar.have_fit)

    def observe(self, t: float, tracks: np.ndarray, H: np.ndarray, layout: bool = True) -> float:
        """Update from tracker rows (x1, y1, x2, y2, id, score, cls, ...) observed at time t."""
        self.layout = layout
        live = set()
        for row in tracks:
            x1, y1, x2, y2, tid, _score, cls = row[:7]
            kind = KIND_BY_CLS.get(int(cls), "other")
            if kind in ("other", "animal"):
                continue
            votes = self.cls_votes.setdefault(int(tid), {})
            votes[int(cls)] = votes.get(int(cls), 0) + 1
            self.cls_seen[int(tid)] = t
            half = np.array([FOOT_WIDTH * (x2 - x1) / 2, FOOT_DEPTH * (y2 - y1) / 2])
            if kind == "person":
                half = np.array([(x2 - x1) / 2, 0.1 * (y2 - y1)])
            centre = warp_points(H, np.array([[(x1 + x2) / 2, y2]]))[0] - np.array([0.0, half[1]])
            self.hist.setdefault(int(tid), _Hist(kind)).add(t, centre, half, float(x2 - x1))
            live.add(int(tid))
        for tid in [k for k in self.hist if k not in live and t - self.hist[k].t[-1] > 1.0]:
            del self.hist[tid]
        if len(self.cls_seen) > 256:     # the tracker drops an id unseen for TRACK_BUFFER_SEC and never reuses it
            for tid in [k for k, ts in self.cls_seen.items() if t - ts > CLS_VOTES_KEEP]:
                del self.cls_seen[tid], self.cls_votes[tid]
        raw = self._raw_risk(live)
        self.last_raw = raw
        rate = ATTACK if raw > self.score else RELEASE
        self.score += rate * (raw - self.score)
        self.features = self.features_hist.push(t, self.base)
        if self.stream is not None:
            self.model_score = self.stream.step(t, self.base, self.features)
            self.model_p = self.stream.p
        self.imm_run = self.imm_run + 1 if self.min_ttc <= IMMINENT_TTC else 0
        if self.imm_run >= IMMINENT_FRAMES:
            self.imm_until = t + IMMINENT_HOLD
        self.imminent = t <= self.imm_until
        return self.score

    def report(self) -> float:
        """The score the estimator outputs. With a trained model whose file says combine "model" (the
        cross-validated choice): the model's calibrated score alone, after its alarm refractory. Otherwise the
        calibrated cue evidence - or, with a legacy model file, the larger of it and the model's score - raised
        to IMMINENT_SCORE on imminent contact."""
        if self.model is not None and self.model.combine == "model":
            s = self.model_score
        else:
            s = calibrate(self.score)
            if self.model is not None:
                s = max(s, self.model_score)
            if self.imminent:
                s = max(s, IMMINENT_SCORE)
        # the harness rounds scores to 4 decimals: keep sub-threshold scores clear of 0.5
        return float(np.clip(s if s >= ALARM else min(s, ALARM_CAP), 0.0, 1.0))

    # -- cues -------------------------------------------------------------------
    def _majority_cls(self, tid: int) -> int:
        votes = self.cls_votes.get(tid)
        return max(votes, key=votes.get) if votes else -1

    def _raw_risk(self, live: set[int]) -> float:
        ids, xy, v, half, kinds, drops, cls = [], [], [], [], [], [], []
        for tid in live:
            h = self.hist[tid]
            vel = h.velocity()
            if vel is None:
                continue
            ids.append(tid)
            xy.append(h.xy[-1])
            v.append(vel)
            half.append(h.half[-1])
            kinds.append(h.kind)
            cls.append(self._majority_cls(tid))
            if h.kind == "vehicle":
                d = h.speed_drop()
                if d is not None:
                    drops.append(d)
        cues = {"cross": 0.0, "rear": 0.0, "ped": 0.0, "brake": 0.0, "pair": None}
        self.last_cues = cues
        self.min_ttc = np.inf
        self.base = np.zeros(len(BASE_FEATURES))
        if not xy:
            return 0.0
        xy, v, half, kinds, ids = np.array(xy), np.array(v), np.array(half), np.array(kinds), np.array(ids)
        speed = np.linalg.norm(v, axis=1)
        on_road = self._lookup(self.road, xy) if self.layout else np.ones(len(xy), bool)
        veh = (kinds != "person") & on_road
        ped = (kinds == "person") & on_road

        if veh.sum() >= 2:
            cues["cross"], p1, t1 = self._crossing(xy[veh], v[veh], half[veh], speed[veh])
            cues["rear"], p2, t2 = self._rear_end(xy[veh], v[veh], speed[veh], half[veh])
            self.min_ttc = min(self.min_ttc, t1, t2)
            pair = p1 if cues["cross"] >= cues["rear"] else p2
            if pair is not None:
                vid = ids[veh]
                cues["pair"] = (int(vid[pair[0]]), int(vid[pair[1]]))
        if veh.any() and ped.any():
            cues["ped"], t3 = self._pedestrian(xy[veh], v[veh], half[veh], speed[veh], xy[ped], v[ped], half[ped])
            if IMMINENT_PED:
                self.min_ttc = min(self.min_ttc, t3)
        drops = np.array(drops)
        braking = drops > BRAKE_DROP
        # one vehicle braking hard is a cue; (almost) all of them at once is a frozen frame or a cut
        frozen = braking.sum() >= 2 and braking.sum() >= BRAKE_ALL_FRAC * len(drops)   # drops: fast vehicles only
        brake = 0.0 if frozen else float(drops.max(initial=0.0))
        cues["brake"] = float(np.clip((brake - BRAKE_DROP) / (1 - BRAKE_DROP), 0, 1)) * BRAKE_WEIGHT
        users = veh | ped
        if users.any():
            widths = np.where(kinds == "person", 2 * half[:, 0], 2 * half[:, 0] / FOOT_WIDTH)
            prev = [self.hist[i].velocity(ago=0.5) for i in ids]
            v_prev = np.array([p if p is not None else np.full(2, np.nan) for p in prev])
            self.base = base_features(xy[users], v[users], half[users], np.maximum(widths[users], 1.0), kinds[users],
                                      cues, self.min_ttc, v_prev[users], overlap_time, cls=np.array(cls)[users])
        keep = 1.0
        for k in ("cross", "rear", "ped", "brake"):
            keep *= 1.0 - cues[k]
        return 1.0 - keep

    @staticmethod
    def _crossing(xy, v, half, speed):
        """-> (cue, pair, time to contact of the fast-closing pairs)."""
        moving = speed > MOVING
        if moving.sum() < 2:
            return 0.0, None, np.inf
        ttc = overlap_time(xy, v, half)
        heading = np.arctan2(v[:, 1], v[:, 0])
        dh = np.degrees(np.abs((heading[None, :] - heading[:, None] + np.pi) % (2 * np.pi) - np.pi))
        valid = moving[None, :] & moving[:, None] & (dh > CROSS_DEG)
        ttc = np.where(valid, ttc, np.inf)
        rel = np.linalg.norm(v[None, :, :] - v[:, None, :], axis=2)
        fast = float(np.where(rel >= IMMINENT_MIN_REL, ttc, np.inf).min())
        i, j = np.unravel_index(np.argmin(ttc), ttc.shape)
        if not np.isfinite(ttc[i, j]):
            return 0.0, None, fast
        return float(np.exp(-ttc[i, j] / CROSS_TTC_SCALE)), (int(i), int(j)), fast

    def _rear_end(self, xy, v, speed, half):
        """Follower i closing on leader j in the same lane of the main road."""
        if self.layout:
            on_main = self._lookup(self.main_road, xy)
            ang = self.lanes.angle(xy)
            width = self.lanes.lane_width(ang)
        else:
            on_main = np.ones(len(xy), bool)
        best, pair, fast = 0.0, None, np.inf
        for i in np.flatnonzero(on_main & (speed > MOVING)):
            u = v[i] / speed[i]
            ahead = (xy - xy[i]) @ u
            if self.layout:
                same_lane = np.abs(ang - ang[i]) < 0.45 * width[i]
            else:
                lateral = np.abs((xy - xy[i]) @ np.array([-u[1], u[0]]))
                same_lane = lateral < (half[:, 0] + half[i, 0]) / FOOT_WIDTH * 0.5
            closing = speed[i] - (v @ u)
            cand = on_main & same_lane & (ahead > 0) & (closing > REAR_MIN_CLOSING)
            cand[i] = False
            if not cand.any():
                continue
            gap_t = ahead[cand] / closing[cand]
            quick = closing[cand] >= IMMINENT_MIN_REL
            if quick.any():
                fast = min(fast, float(gap_t[quick].min()))
            k = int(np.argmin(gap_t))
            r = float(np.exp(-gap_t[k] / REAR_TTC_SCALE))
            if r > best:
                best, pair = r, (int(i), int(np.flatnonzero(cand)[k]))
        return best, pair, fast

    @staticmethod
    def _pedestrian(vxy, vv, vhalf, vspeed, pxy, pv, phalf):
        moving = vspeed > MOVING
        if not moving.any():
            return 0.0, np.inf
        xy = np.concatenate([vxy[moving], pxy])
        v = np.concatenate([vv[moving], pv])
        half = np.concatenate([vhalf[moving], phalf])
        ttc = overlap_time(xy, v, half)
        nv = int(moving.sum())
        cross = ttc[:nv, nv:]
        if not np.isfinite(cross).any():
            return 0.0, np.inf
        rel = np.linalg.norm(vv[moving][:, None, :] - pv[None, :, :], axis=2)
        fast = float(np.where(rel >= IMMINENT_MIN_REL, cross, np.inf).min())
        return PED_WEIGHT * float(np.exp(-cross.min() / PED_TTC_SCALE)), fast

    @staticmethod
    def _lookup(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
        x = np.clip(np.round(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
        y = np.clip(np.round(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
        return mask[y, x]


class RiskEstimator:
    """The harness-facing wrapper: reset(meta) once per video, then step(frame, t) per frame."""

    def reset(self, meta: dict) -> None:
        from .pipeline import shared_model  # deferred: avoids loading torch at import time
        self.model = shared_model()
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(fps / RISK_FPS)))
        self.online = OnlineRisk(fps)
        self.i = 0
        self.score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.i
        self.i += 1
        if i % self.stride:
            return self.score
        if frame.shape[1] != config.ANALYSIS_SIZE[0]:
            frame = cv2.resize(frame, config.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        self.online.update(frame, t_sec, self.model)
        self.score = self.online.report()
        return self.score


def risk_curve_from_analysis(va) -> tuple[np.ndarray, np.ndarray]:
    """Causal replay of an analysis' tracks through the risk logic -> (t, calibrated score).

    Every analysed frame is fed in order, exactly as RiskEstimator.step would see
    it (minus the detector, whose outputs are the cached tracks). Used by the demo
    and the website, where running the detector twice is too slow.
    """
    online = OnlineRisk(va.meta.fps)
    fit_so_far = np.maximum.accumulate(va.reg_ok.astype(bool)) if len(va.reg_ok) else np.zeros(1, bool)
    table = va.tracks
    if not len(table):
        return np.zeros(0), np.zeros(0)
    table = table[np.argsort(table[:, 0], kind="stable")]
    frames, starts = np.unique(table[:, 0], return_index=True)
    ends = np.append(starts[1:], len(table))
    ts, scores = [], []
    for s, e in zip(starts, ends):
        rows = table[s:e]
        t = float(rows[0, 1])
        k = va.H_index(np.array([t]))[0]
        tracks = np.column_stack([rows[:, 3:7], rows[:, 2], rows[:, 7], rows[:, 8]])
        ts.append(t)
        # layout on from the first successful registration, as Registrar.have_fit in RiskEstimator
        online.observe(t, tracks, va.reg_H[k], bool(fit_so_far[min(k, len(fit_so_far) - 1)]))
        scores.append(online.report())
    return np.array(ts), np.array(scores)
