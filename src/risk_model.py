"""Learned accident-anticipation layer of Part B.

Per analysed frame the online estimator (src/risk.py) computes BASE_FEATURES from the tracked
road users it already holds: conflict cues, scale-free kinematics, pairwise conflicts predicted under
constant acceleration, vulnerable road users (pedestrians, bicycles, motorcycles) near moving cars, and
hard deceleration / swerving. FeatureHistory adds their short-term context (max over the last 1 s and
3 s, deviation from the 3-s mean).

RiskModel is a small ensemble trained on timed third-party CCTV crashes plus ordinary traffic
(scripts/train_risk_model.py):
  p = (1 - tcn_weight) * mean_k MLP_k(context features) + tcn_weight * mean_k TCN_k(last 63 base-feature frames)
MLP_k: one-hidden-layer networks on the standardised context features; TCN_k: causal dilated temporal
convolutions over the standardised base features of the last 63 analysed frames (frames before the stream
start count as "nothing observed"). p is smoothed (fast attack / slow release), calibrated so that alarm_p
maps to the 0.5 alarm threshold, and a refractory period keeps one incident to one alarm: after an alarm
starts at ta, no new alarm may start before ta + refractory (the score is capped just below 0.5 there).
Everything uses only the past.

A legacy model file (one network, no "mlps" key) still loads and keeps its old behaviour
(combine "max": the larger of the model and the hand-made cue score).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from . import config

BASE_CUES = ("cross", "rear", "ped", "brake", "imminent", "any_ttc", "proximity", "closing",
             "yaw", "accel", "speed", "n_moving", "n_people")
EXTRA_FEATURES = ("ttc_acc", "gap_acc", "approach_acc", "vru_prox", "vru_ttc", "decel", "lat_acc")
BASE_FEATURES = BASE_CUES + EXTRA_FEATURES
CONTEXT = ("now", "max1", "max3", "dev3")
FEATURE_NAMES = tuple(f"{b}_{c}" for c in CONTEXT for b in BASE_FEATURES)
MODEL_FILE = config.ROOT / "assets" / "risk_model.json"

MOVING_BW = 0.5          # box widths per second: a vehicle this fast is moving
REL_SPEED = 50.0         # px/s: pairs closing slower than this are ignored by any_ttc / ttc_acc / vru_ttc
NEAR_WIDTHS = 3.0        # pairs within this many mean box widths count for "closing"
ACC_GRID = np.round(np.arange(0.1, 3.01, 0.1), 2)   # s: prediction times of the constant-acceleration model
GAP_HORIZON = 2.0        # s: gap_acc looks this far ahead
APPROACH_WIDTHS = 1.5    # approach_acc counts pairs predicted to come within this many mean widths
ALARM, ALARM_CAP = 0.5, 0.499
MERGE_GAP = 2.0          # s: alarm runs closer than this are one alarm for the metric (evaluate.MERGE_GAP)


def base_features(xy, v, half, widths, kinds, cues, min_ttc, v_prev, overlap_time, cls=None) -> np.ndarray:
    """One frame's scale-free kinematic summary (all arrays are per live road user on the road;
    cls: majority COCO class per road user, used to treat motorcycles as vulnerable road users)."""
    f = dict.fromkeys(BASE_FEATURES, 0.0)
    for k in ("cross", "rear", "ped", "brake"):
        f[k] = float(cues.get(k, 0.0) or 0.0)
    f["imminent"] = float(np.exp(-min_ttc)) if np.isfinite(min_ttc) else 0.0
    veh = kinds != "person"
    f["n_people"] = float(np.log1p((~veh).sum()))
    if veh.any():
        sp_bw = np.linalg.norm(v[veh], axis=1) / widths[veh]
        f["speed"] = float(np.log1p(sp_bw.max()))
        f["n_moving"] = float(np.log1p((sp_bw > MOVING_BW).sum()))
        ok = veh & np.isfinite(v_prev).all(1)
        fast = ok & (np.linalg.norm(v, axis=1) > MOVING_BW * widths) & (np.linalg.norm(np.nan_to_num(v_prev), axis=1) > MOVING_BW * widths)
        if fast.any():
            a0 = np.arctan2(v_prev[fast, 1], v_prev[fast, 0])
            a1 = np.arctan2(v[fast, 1], v[fast, 0])
            yaw = np.degrees(np.abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)) / 0.5
            f["yaw"] = float(np.log1p(yaw.max() / 10.0))
        if ok.any():
            acc = np.linalg.norm(v[ok] - v_prev[ok], axis=1) / 0.5 / widths[ok]
            f["accel"] = float(np.log1p(acc.max()))
    idx = np.flatnonzero(veh & (np.linalg.norm(v, axis=1) > MOVING_BW * widths))
    if len(idx) >= 2:
        xy2, v2, h2, w2 = xy[idx], v[idx], half[idx], widths[idx]
        rel = np.linalg.norm(v2[None] - v2[:, None], axis=2)
        ttc = np.where(rel >= REL_SPEED, overlap_time(xy2, v2, h2), np.inf)
        f["any_ttc"] = float(np.exp(-ttc.min())) if np.isfinite(ttc.min()) else 0.0
        mw = (w2[None] + w2[:, None]) / 2
        d = np.linalg.norm(xy2[None] - xy2[:, None], axis=2) / mw
        np.fill_diagonal(d, np.inf)
        f["proximity"] = float(np.exp(-d.min()))
        dxy = xy2[None] - xy2[:, None]
        closing = -(dxy * (v2[None] - v2[:, None])).sum(2) / np.maximum(np.linalg.norm(dxy, axis=2), 1e-6) / mw
        closing = np.where(d < NEAR_WIDTHS, closing, 0.0)
        f["closing"] = float(np.log1p(max(0.0, closing.max())))
    _extra_features(f, xy, v, half, widths, kinds, v_prev, cls, overlap_time)
    return np.array([f[k] for k in BASE_FEATURES])


def _extra_features(f, xy, v, half, widths, kinds, v_prev, cls, overlap_time) -> None:
    """ttc_acc / gap_acc / approach_acc: pairs of moving vehicles predicted under constant acceleration
    (acceleration = (v - v 0.5 s ago) / 0.5, clipped to max(|v|, width) per s; a braking vehicle stops
    instead of reversing): exp(-first footprint overlap time), exp(-min predicted gap in mean widths within
    GAP_HORIZON), log1p(current gap - min predicted gap) of pairs predicted within APPROACH_WIDTHS.
    vru_prox / vru_ttc: pedestrians, bicycles and motorcycles vs moving cars: exp(-min distance in car widths),
    exp(-constant-velocity footprint overlap time) of pairs closing >= REL_SPEED.
    decel / lat_acc: max longitudinal deceleration and lateral (swerve) acceleration of moving vehicles, widths/s^2."""
    person = kinds == "person"
    cls = np.full(len(xy), -1) if cls is None else np.asarray(cls)
    speed = np.linalg.norm(v, axis=1)
    mov = ~person & (speed / widths > MOVING_BW)
    okv = np.isfinite(v_prev).all(1)
    acc = np.where(okv[:, None], (v - np.nan_to_num(v_prev)) / 0.5, 0.0)
    okp = mov & okv
    if okp.any():
        u = v[okp] / np.maximum(speed[okp], 1e-6)[:, None]
        a = acc[okp]
        f["decel"] = float(np.log1p(max(0.0, (-(a * u).sum(1) / widths[okp]).max())))
        f["lat_acc"] = float(np.log1p((np.abs(a[:, 0] * u[:, 1] - a[:, 1] * u[:, 0]) / widths[okp]).max()))
    idx = np.flatnonzero(mov)
    if len(idx) >= 2:
        X, V, A, Hf, W = xy[idx], v[idx], acc[idx], half[idx], widths[idx]
        amax = np.maximum(np.linalg.norm(V, axis=1), W)
        A = A * np.minimum(1.0, amax / np.maximum(np.linalg.norm(A, axis=1), 1e-6))[:, None]
        va = (V * A).sum(1)
        t_stop = np.where(va < 0, (V * V).sum(1) / np.maximum(-va, 1e-9), np.inf)
        tau = np.minimum(ACC_GRID[:, None], t_stop[None, :])
        P = X[None] + V[None] * tau[..., None] + 0.5 * A[None] * tau[..., None] ** 2
        i, j = np.triu_indices(len(idx), 1)
        hs = Hf[i] + Hf[j]
        d0 = X[j] - X[i]
        dP = P[:, j] - P[:, i]
        inside = (np.abs(dP) < hs[None]).all(2)
        ok = ~(np.abs(d0) < hs).all(1) & (np.linalg.norm(V[j] - V[i], axis=1) >= REL_SPEED)
        first = np.where(ok & inside.any(0), ACC_GRID[np.argmax(inside, axis=0)], np.inf)
        if np.isfinite(first).any():
            f["ttc_acc"] = float(np.exp(-first.min()))
        mw = (W[i] + W[j]) / 2
        dn = np.linalg.norm(d0, axis=1) / mw
        dmin = (np.linalg.norm(dP[ACC_GRID <= GAP_HORIZON], axis=2) / mw[None]).min(0)
        f["gap_acc"] = float(np.exp(-dmin.min()))
        f["approach_acc"] = float(np.log1p(max(0.0, np.where(dmin < APPROACH_WIDTHS, dn - dmin, 0.0).max())))
    vru = person | (kinds == "bicycle") | (cls == config.MOTORCYCLE)
    cars = mov & ~vru
    if vru.any() and cars.any():
        vi, ci = np.flatnonzero(vru), np.flatnonzero(cars)
        d = np.linalg.norm(xy[vi][:, None] - xy[ci][None], axis=2) / widths[ci][None]
        f["vru_prox"] = float(np.exp(-d.min()))
        tt = overlap_time(np.concatenate([xy[ci], xy[vi]]), np.concatenate([v[ci], v[vi]]),
                          np.concatenate([half[ci], half[vi]]))[:len(ci), len(ci):]
        rel = np.linalg.norm(v[ci][:, None] - v[vi][None], axis=2)
        tt = np.where(rel >= REL_SPEED, tt, np.inf)
        if np.isfinite(tt).any():
            f["vru_ttc"] = float(np.exp(-tt.min()))


class FeatureHistory:
    """Short-term context of the base features (causal)."""

    def __init__(self) -> None:
        self.t: deque = deque()
        self.x: deque = deque()

    def push(self, t: float, base: np.ndarray) -> np.ndarray:
        self.t.append(t)
        self.x.append(base)
        while self.t and t - self.t[0] > 3.0:
            self.t.popleft()
            self.x.popleft()
        tt, xx = np.array(self.t), np.array(self.x)
        m1 = tt >= t - 1.0
        return np.concatenate([base, xx[m1].max(0), xx.max(0), base - xx.mean(0)])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _TCNStack:
    """Streaming numpy version of K causal TCNs of identical shape (scripts/train_risk_model.TCN):
    1x1 conv F->C, residual dilated convs (kernel k), linear head on [hidden, current input].
    Per layer a ring buffer holds the past inputs its taps need; the buffers start in the state that
    a stream of "nothing observed" frames (raw zeros) produces, as in training."""

    def __init__(self, members: list[dict]) -> None:
        self.mean = np.array([m["mean"] for m in members])                  # (K, F)
        self.std = np.array([m["std"] for m in members])
        self.w_in = np.array([m["inp_w"] for m in members])                 # (K, C, F)
        self.b_in = np.array([m["inp_b"] for m in members])                 # (K, C)
        self.dil = [int(c["d"]) for c in members[0]["convs"]]
        self.w = [np.array([m["convs"][li]["w"] for m in members]) for li in range(len(self.dil))]   # (K, C, C, k)
        self.b = [np.array([m["convs"][li]["b"] for m in members]) for li in range(len(self.dil))]
        self.w_out = np.array([m["out_w"] for m in members])                # (K, C + F)
        self.b_out = np.array([m["out_b"] for m in members])                # (K,)
        self.k = self.w[0].shape[3]
        self.rf = 1 + sum((self.k - 1) * d for d in self.dil)
        z = (0.0 - self.mean) / self.std
        h = np.maximum(np.einsum("kcf,kf->kc", self.w_in, z) + self.b_in, 0.0)
        self.buf, self.pos = [], []
        for w, b, d in zip(self.w, self.b, self.dil):
            n = (self.k - 1) * d + 1
            self.buf.append(np.repeat(h[:, None, :], n, axis=1))            # (K, n, C)
            self.pos.append(n - 1)
            h = np.maximum(np.einsum("kcij,ki->kc", w, h) + b, 0.0) + h    # constant input: all taps equal

    def step(self, base: np.ndarray) -> np.ndarray:
        """Feed one analysed frame's base features; -> (K,) probabilities."""
        z = (base[None] - self.mean) / self.std
        h = np.maximum(np.einsum("kcf,kf->kc", self.w_in, z) + self.b_in, 0.0)
        for li, (w, b, d) in enumerate(zip(self.w, self.b, self.dil)):
            buf = self.buf[li]
            n = buf.shape[1]
            p = (self.pos[li] + 1) % n
            buf[:, p] = h
            self.pos[li] = p
            taps = buf[:, [(p - (self.k - 1 - j) * d) % n for j in range(self.k)]]   # (K, k, C): x[t-(k-1-j)d]
            h = np.maximum(np.einsum("kcij,kji->kc", w, taps) + b, 0.0) + h
        out = np.einsum("kc,kc->k", self.w_out, np.concatenate([h, z], axis=1)) + self.b_out
        return _sigmoid(out)


class RiskModel:
    """Ensemble of one-hidden-layer networks (or a logistic regression) on standardised context features,
    optionally mixed with causal TCNs over the base features; weights shared by every stream."""

    def __init__(self, params: dict) -> None:
        mlps = params.get("mlps") or [params]        # legacy file: a single network at top level
        self.mean = np.array([m["mean"] for m in mlps])            # (M, F)
        self.std = np.array([m["std"] for m in mlps])
        self.hidden = all(m.get("hidden") for m in mlps)
        if self.hidden:
            self.w1 = np.array([m["hidden"]["w"] for m in mlps])   # (M, F, H)
            self.b1 = np.array([m["hidden"]["b"] for m in mlps])   # (M, H)
        self.w = np.array([m["weights"] for m in mlps])            # (M, H) or (M, F)
        self.b = np.array([float(m["bias"]) for m in mlps])
        self.tcns = params.get("tcns") or []
        self.tcn_weight = float(params.get("tcn_weight", 0.5)) if self.tcns else 0.0
        self.alarm_p = float(params["alarm_p"])        # probability mapped to the 0.5 alarm threshold
        self.attack = float(params.get("attack", 1.0))  # fast-attack / slow-release smoothing of the probability
        self.release = float(params.get("release", 1.0))
        self.refractory = float(params.get("refractory", 0.0))
        self.combine = params.get("combine", "max")      # "model": report the model alone; "max": with the cues

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> "RiskModel | None":
        if not Path(path).exists():
            return None
        params = json.loads(Path(path).read_text())
        if list(params.get("features", [])) != list(FEATURE_NAMES):
            return None          # trained on another feature set
        return cls(params)

    def prob(self, x: np.ndarray) -> float:
        """Mean probability of the context-feature networks."""
        z = (x[None] - self.mean) / self.std
        if self.hidden:
            z = np.maximum(np.einsum("mf,mfh->mh", z, self.w1) + self.b1, 0.0)
        return float(_sigmoid(np.einsum("mh,mh->m", z, self.w) + self.b).mean())

    def calibrated(self, p: float) -> float:
        """Monotone map sending alarm_p to 0.5 (ranking unchanged)."""
        a = self.alarm_p
        return 0.5 * p / a if p < a else 0.5 + 0.5 * (p - a) / (1.0 - a)

    def stream(self) -> "ModelStream":
        return ModelStream(self)


class ModelStream:
    """Per-video state of a RiskModel: TCN buffers, smoothing and the alarm refractory. step() once per analysed frame."""

    def __init__(self, model: RiskModel) -> None:
        self.model = model
        self.tcn = _TCNStack(model.tcns) if model.tcns else None
        self.p = 0.0              # smoothed probability
        self.score = 0.0          # calibrated score after the refractory
        self.on = False           # an alarm run is in progress
        self.t_start = -np.inf    # start of the current / last alarm
        self.t_last = -np.inf     # last time the reported score was >= ALARM

    def step(self, t: float, base: np.ndarray, x: np.ndarray) -> float:
        m = self.model
        p = m.prob(x)
        if self.tcn is not None:
            p = (1.0 - m.tcn_weight) * p + m.tcn_weight * float(self.tcn.step(base).mean())
        self.p += (m.attack if p > self.p else m.release) * (p - self.p)
        self.score = self._refractory(t, m.calibrated(self.p))
        return self.score

    def _refractory(self, t: float, s: float) -> float:
        R = self.model.refractory
        if R <= 0:
            return s
        if s < ALARM:
            self.on = False
            return s
        if self.on or t - self.t_last < MERGE_GAP:    # the running alarm (or one the metric merges into it)
            self.on, self.t_last = True, t
            return s
        if t < self.t_start + R:                       # a new alarm this soon after the last one: suppressed
            return min(s, ALARM_CAP)
        self.on, self.t_start, self.t_last = True, t, t
        return s
