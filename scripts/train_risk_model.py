"""Train and cross-validate the learned Part B layer (src/risk_model.py).

    python scripts/train_risk_model.py --build          # replay every cached clip -> work/risk_dataset.npz
    python scripts/train_risk_model.py --cv             # grouped cross-validation, official Part B metric
    python scripts/train_risk_model.py --fit            # fit on everything -> assets/risk_model.json

The shipped model is a stack (--stack K): K one-hidden-layer networks on the context features and K causal
TCNs on the per-frame base features, p = 0.5 * mean(MLPs) + 0.5 * mean(TCNs), smoothing 0.6/0.12, the model
score alone (--combine model), a 10-s alarm refractory:
    python scripts/train_risk_model.py --build
    python scripts/train_risk_model.py --cv --stack 5 --fa-target 0.0                       # -> alarm point
    python scripts/train_risk_model.py --fit --stack 5 --alarm-p <point from --cv>

Data: Part B online-tracker caches (scripts/ext_cache.py) of the TAD benchmark clips with our crash
timings (dev/external/tad_labels.json; accident-free clips are negatives) and of the four sample
videos (ordinary traffic on the target camera, negatives). A frame is positive in the 5 s before a
crash (the official anticipation horizon), ignored during a crash and around its replays in edited
clips. Cross-validation holds out whole clips; the alarm point is set so that the held-out sample
videos give at most --fa-target false alarms per minute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
from src import risk  # noqa: E402
from src.risk_model import BASE_FEATURES, FEATURE_NAMES, MODEL_FILE  # noqa: E402

H = evaluate.H
NB = len(BASE_FEATURES)          # X[:, :NB] = the per-frame base features (the "now" block of the context)
TCN_CH, TCN_K, TCN_DIL = 16, 3, (1, 2, 4, 8, 16)   # receptive field 1 + 2 * 31 = 63 analysed frames


def clip_list(labels: dict, caches: list[Path]) -> list[tuple[str, Path, dict]]:
    out = []
    for cache in caches:
        for f in sorted(cache.glob("*.risk.npz")):
            name = f.name.replace(".risk.npz", ".mp4")
            if "_norm_" in name:
                out.append((name, f, {"case": "normal"}))
            elif name in labels:
                if labels[name]["case"] == "collision":
                    out.append((name, f, labels[name]))
            elif "samples" in cache.name:                 # the sample videos: ordinary traffic
                out.append((name, f, {"case": "sample"}))
    return out


def replay(path: Path):
    """Causal replay of cached tracker rows: (t, features, baseline score) per analysed frame."""
    risk._MODEL_CACHE[:] = [None]                       # features and baseline only
    d = np.load(path)
    online = risk.OnlineRisk(float(d["fps"]))
    rows = d["rows"][np.argsort(d["rows"][:, 0], kind="stable")]
    a = np.searchsorted(rows[:, 0], np.arange(len(d["t"])))
    b = np.searchsorted(rows[:, 0], np.arange(len(d["t"])), side="right")
    T, X, B = [], [], []
    for k in range(len(d["t"])):
        online.observe(float(d["t"][k]), rows[a[k]:b[k], 1:8], d["H"][k], bool(d["have_fit"][k]))
        T.append(float(d["t"][k]))
        X.append(online.features)
        B.append(online.report())
    return np.array(T), np.array(X), np.array(B), float(d["n_frames"]) / float(d["fps"])


def labels_for(t: np.ndarray, lab: dict) -> np.ndarray:
    """1 positive, 0 negative, -1 ignored."""
    if lab["case"] != "collision":
        return np.zeros(len(t), int)
    acc = [(lab["start"], lab["end"])]
    near = [tuple(x) for x in lab.get("extra", []) or []]
    return np.array([{None: -1, 1: 1, 0: 0}[evaluate.frame_label(x, acc, near)] for x in t])


def build(args) -> None:
    labels = json.loads(Path(args.labels).read_text())
    clips = clip_list(labels, [Path(c) for c in args.caches])
    T, X, Y, B, G, info = [], [], [], [], [], []
    for gi, (name, f, lab) in enumerate(clips):
        t, x, b, dur = replay(f)
        T.append(t); X.append(x); B.append(b); Y.append(labels_for(t, lab)); G.append(np.full(len(t), gi))
        info.append({"name": name, "case": lab["case"], "start": lab.get("start"), "end": lab.get("end"),
                     "extra": lab.get("extra") or [], "duration": dur})
    np.savez_compressed(args.dataset, t=np.concatenate(T), X=np.concatenate(X), y=np.concatenate(Y),
                        base=np.concatenate(B), g=np.concatenate(G), info=json.dumps(info))
    cases = [i["case"] for i in info]
    print(f"{len(info)} clips: {cases.count('collision')} crashes, {cases.count('normal')} normal, "
          f"{cases.count('sample')} sample videos; {sum(len(t) for t in T)} frames, "
          f"{int(sum((y == 1).sum() for y in Y))} positive")


# ------------------------------------------------------------------ model
def fit_logreg(X, y, l2: float, iters: int = 300):
    """Class-balanced L2 logistic regression by Newton's method (standardised inputs)."""
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = np.column_stack([(X - mean) / std, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    pos = y == 1
    sw = np.where(pos, 0.5 / max(pos.mean(), 1e-6), 0.5 / max((~pos).mean(), 1e-6))
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (sw * (p - y)) / len(y) + reg * w
        Hm = (Z * (sw * p * (1 - p))[:, None]).T @ Z / len(y) + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mean, std, w[:-1], w[-1]


def fit_mlp(X, y, l2: float, hidden: int = 16, epochs: int = 60, seed: int = 0):
    """Class-balanced one-hidden-layer network (ReLU), full-batch Adam; deterministic."""
    import torch
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = torch.tensor((X - mean) / std, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    pos = float((y == 1).mean())
    wt = torch.where(Y > 0.5, torch.tensor(0.5 / max(pos, 1e-6)), torch.tensor(0.5 / max(1 - pos, 1e-6)))
    net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=l2 * 1e-2)
    for _ in range(epochs * 10):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Z).squeeze(1), Y, reduction="none") * wt).mean()
        loss.backward()
        opt.step()
    l1, l2_ = net[0], net[2]
    return (mean, std, l2_.weight.detach().numpy()[0].astype(float), float(l2_.bias.item()),
            {"w": l1.weight.detach().numpy().T.astype(float), "b": l1.bias.detach().numpy().astype(float)})


def predict(model, X):
    mean, std, w, b = model[:4]
    z = (X - mean) / std
    if len(model) > 4 and model[4] is not None:
        z = np.maximum(z @ model[4]["w"] + model[4]["b"], 0.0)
    return 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30)))


def fit_model(kind: str, X, y, l2: float):
    if kind == "logreg":
        return fit_logreg(X, y, l2)
    return fit_mlp(X, y, l2, hidden=32 if kind == "mlp32" else 16)


def mlp_params(model) -> dict:
    mean, std, w, b = model[:4]
    hidden = model[4] if len(model) > 4 else None
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist(), "bias": float(b),
            "hidden": {"w": hidden["w"].tolist(), "b": hidden["b"].tolist()} if hidden is not None else None}


# ------------------------------------------------------------------ causal TCN over the base features
def _tcn_net(fin: int):
    """1x1 conv fin->C, residual dilated causal convs, linear head on [hidden, current input] (logits)."""
    import torch
    from torch import nn

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(fin, TCN_CH, 1)
            self.convs = nn.ModuleList([nn.Conv1d(TCN_CH, TCN_CH, TCN_K, dilation=d) for d in TCN_DIL])
            self.crop = [(TCN_K - 1) * d for d in TCN_DIL]
            self.rf = 1 + sum(self.crop)
            self.out = nn.Conv1d(TCN_CH + fin, 1, 1)

        def forward(self, x):            # (B, F, L + rf - 1), left-padded -> (B, L)
            h = torch.relu(self.inp(x))
            for conv, c in zip(self.convs, self.crop):
                h = torch.relu(conv(h)) + h[..., c:]
            return self.out(torch.cat([h, x[..., self.rf - 1:]], 1)).squeeze(1)

    return TCN()


def clip_spans(G) -> list[tuple[int, int]]:
    """(start, stop) frame index range of every clip (clips are contiguous in the dataset)."""
    n = int(G.max()) + 1
    a = np.searchsorted(G, np.arange(n))
    b = np.searchsorted(G, np.arange(n), side="right")
    return [(int(x), int(z)) for x, z in zip(a, b)]


def _tcn_batch(Z, y, spans, chunk, ctx, pad):
    """Pieces of <= chunk frames of every clip in spans, each with ctx frames of left context (pad before the
    clip start): inputs (P, ctx + C, F), labels (P, C), loss mask (P, C), pieces [(a, b)]."""
    pieces = [(s0, a, min(a + chunk, s1)) for s0, s1 in spans for a in range(s0, s1, chunk)]
    C = max(b - a for _, a, b in pieces)
    B = np.tile(pad[None, None, :], (len(pieces), ctx + C, 1)).astype(np.float32)
    Y = np.zeros((len(pieces), C), np.float32)
    M = np.zeros((len(pieces), C), bool)
    for i, (s0, a, b) in enumerate(pieces):
        lo = max(s0, a - ctx)
        B[i, ctx - (a - lo):ctx + (b - a)] = Z[lo:b]
        Y[i, :b - a] = y[a:b] == 1
        M[i, :b - a] = y[a:b] >= 0
    return B, Y, M, [(a, b) for _, a, b in pieces]


def fit_tcn(Xb, y, spans, l2: float = 1.0, steps: int = 600, seed: int = 0, chunk: int = 256):
    """Class-balanced causal TCN on the clips in spans (frames with y >= 0), full-batch Adam lr 0.01.
    Inputs standardised on the training frames; frames before a clip start are raw zeros ("nothing observed").
    -> (member params for src.risk_model, torch net, mean, std)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = np.zeros(len(y), bool)
    for s0, s1 in spans:
        m[s0:s1] = y[s0:s1] >= 0
    mean, std = Xb[m].mean(0), Xb[m].std(0) + 1e-6
    Z = ((Xb - mean) / std).astype(np.float32)
    pad = (-mean / std).astype(np.float32)
    net = _tcn_net(Xb.shape[1]).to(dev)
    B, Y, M, _ = _tcn_batch(Z, y, spans, chunk, net.rf - 1, pad)
    xb = torch.tensor(B, device=dev).transpose(1, 2)
    yb, mb = torch.tensor(Y, device=dev), torch.tensor(M, device=dev)
    pos = float(Y[M].mean())
    wt = torch.where(yb > 0.5, 0.5 / pos, 0.5 / (1 - pos)) * mb
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=l2 * 1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xb), yb, reduction="none") * wt).sum() / mb.sum()
        loss.backward()
        opt.step()
    net.eval()
    sd = {k: v.detach().cpu().numpy().astype(float) for k, v in net.state_dict().items()}
    params = {"mean": mean.tolist(), "std": std.tolist(), "inp_w": sd["inp.weight"][:, :, 0].tolist(),
              "inp_b": sd["inp.bias"].tolist(),
              "convs": [{"w": sd[f"convs.{i}.weight"].tolist(), "b": sd[f"convs.{i}.bias"].tolist(), "d": d}
                        for i, d in enumerate(TCN_DIL)],
              "out_w": sd["out.weight"][0, :, 0].tolist(), "out_b": float(sd["out.bias"][0])}
    return params, net, mean, std


def predict_tcn(fitted, Xb, spans, chunk: int = 256) -> np.ndarray:
    """Causal batch prediction of the frames in spans (whole clips from their first frame); other frames 0."""
    import torch
    _, net, mean, std = fitted
    dev = next(net.parameters()).device
    Z = ((Xb - mean) / std).astype(np.float32)
    pad = (-mean / std).astype(np.float32)
    B, _, _, pieces = _tcn_batch(Z, np.zeros(len(Xb), int), spans, chunk, net.rf - 1, pad)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(B, device=dev).transpose(1, 2))).cpu().numpy()
    out = np.zeros(len(Xb))
    for i, (a, b) in enumerate(pieces):
        out[a:b] = p[i, :b - a]
    return out


def fit_stack(X, y, spans, k: int, l2: float, seed0: int = 0):
    """k MLPs on the context features and k TCNs on the base features X[:, :NB], trained on the clips in spans."""
    m = np.zeros(len(y), bool)
    for s0, s1 in spans:
        m[s0:s1] = y[s0:s1] >= 0
    mlps = [fit_mlp(X[m], y[m], l2, hidden=16, seed=seed0 + i) for i in range(k)]
    tcns = [fit_tcn(X[:, :NB], y, spans, l2, seed=seed0 + i) for i in range(k)]
    return mlps, tcns


def predict_stack(mlps, tcns, X, spans, tcn_weight: float) -> np.ndarray:
    p = np.mean([predict(mo, X) for mo in mlps], axis=0)
    if tcns:
        p = (1 - tcn_weight) * p + tcn_weight * np.mean([predict_tcn(tc, X[:, :NB], spans) for tc in tcns], axis=0)
    return p


def refractory(t, s, seconds: float):
    """The run-time alarm refractory (src.risk_model.ModelStream) on one clip's reported curve."""
    if seconds <= 0:
        return s
    out, on, t_start, t_last = s.copy(), False, -np.inf, -np.inf
    for i in range(len(s)):
        if s[i] < evaluate.THETA:
            on = False
        elif on or t[i] - t_last < evaluate.MERGE_GAP:
            on, t_last = True, t[i]
        elif t[i] < t_start + seconds:
            out[i] = min(s[i], 0.499)
        else:
            on, t_start, t_last = True, t[i], t[i]
    return out


def smooth(p, attack, release):
    out, s = np.empty_like(p), 0.0
    for i, x in enumerate(p):
        s += (attack if x > s else release) * (x - s)
        out[i] = s
    return out


def calib(p, point):
    return np.where(p < point, 0.5 * p / point, 0.5 + 0.5 * (p - point) / (1 - point))


# ------------------------------------------------------------------ evaluation
def curves_by_clip(D, score):
    out = {}
    for gi, inf in enumerate(D["info"]):
        m = D["g"] == gi
        out[gi] = (D["t"][m], score[m])
    return out


def part_b(D, curves, fa_clips=("sample",)):
    gt, pv = {}, {}
    for gi, inf in enumerate(D["info"]):
        if inf["case"] == "sample":
            continue
        ev = []
        if inf["case"] == "collision":
            ev = [[inf["start"], inf["end"], "accident"]] + [[a, b, "near_miss"] for a, b in inf["extra"]]
        t, s = curves[gi]
        gt[inf["name"]] = {"events": ev}
        pv[inf["name"]] = {"events": [], "risk": [[float(a), float(b)] for a, b in zip(t, s)]}
    return evaluate.evaluate_part_b(gt, pv) or {}


def fa_per_min(D, curves, case):
    n, mins = 0, 0.0
    for gi, inf in enumerate(D["info"]):
        if inf["case"] == case:
            t, s = curves[gi]
            n += len(evaluate.alarm_starts([[float(a), float(b)] for a, b in zip(t, s)]))
            mins += inf["duration"] / 60
    return n / mins if mins else float("nan")


def choose_point(D, raw_curves, fa_target, case="sample"):
    """Smallest alarm point whose calibrated curves give <= fa_target false alarms/min on `case` clips."""
    for point in np.round(np.arange(0.05, 0.995, 0.01), 3):
        curves = {k: (t, calib(s, point)) for k, (t, s) in raw_curves.items()}
        if fa_per_min(D, curves, case) <= fa_target:
            return float(point)
    return 0.99


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return {"t": d["t"], "X": d["X"], "y": d["y"], "base": d["base"], "g": d["g"], "info": json.loads(str(d["info"]))}


def cv(args) -> None:
    D = load_dataset(args.dataset)
    G = D["g"]
    groups = np.unique(G)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(groups, rng.permutation(len(groups)) % args.folds))
    folds = np.array([fold_of[g] for g in G])
    have_samples = any(i["case"] == "sample" for i in D["info"])
    fa_case = "sample" if have_samples else "normal"
    print(f"alarm point set on held-out '{fa_case}' clips at <= {args.fa_target} false alarms/min")
    # baseline: the current estimator's reported score
    base_curves = curves_by_clip(D, D["base"])
    rb = part_b(D, base_curves)
    print(f"current estimator      : Score_B {rb.get('score_b', 0):.3f} AP {rb.get('ap', 0):.3f} F1 {rb.get('f1_alarm', 0):.3f} "
          f"(P {rb.get('alarm_precision', 0):.2f} R {rb.get('alarm_recall', 0):.2f}) mTTA {rb.get('mtta_sec', 0):.2f}s "
          f"FA/min {fa_case} {fa_per_min(D, base_curves, fa_case):.2f}")
    if args.stack:
        cv_stack(args, D, fold_of, folds, base_curves, fa_case)
        return
    for kind, l2 in [(k, l) for k in args.models for l in args.l2]:
        oof = np.zeros(len(G))
        for k in range(args.folds):
            tr = (folds != k) & (D["y"] >= 0)
            model = fit_model(kind, D["X"][tr], D["y"][tr], l2)
            te = folds == k
            oof[te] = predict(model, D["X"][te])
        for att, rel in ((1.0, 1.0), (0.6, 0.12)):
            raw = {gi: (t, smooth(s, att, rel)) for gi, (t, s) in curves_by_clip(D, oof).items()}
            point = choose_point(D, raw, args.fa_target, fa_case)
            model_curves = {gi: (t, calib(s, point)) for gi, (t, s) in raw.items()}
            for mode in ("model", "max"):
                curves = model_curves if mode == "model" else {
                    gi: (t, np.maximum(s, base_curves[gi][1])) for gi, (t, s) in model_curves.items()}
                r = part_b(D, curves)
                print(f"{kind:6s} l2 {l2:<5} smooth {att}/{rel:<4} {mode:5s}: Score_B {r.get('score_b', 0):.3f} AP {r.get('ap', 0):.3f} "
                      f"F1 {r.get('f1_alarm', 0):.3f} (P {r.get('alarm_precision', 0):.2f} R {r.get('alarm_recall', 0):.2f}) "
                      f"mTTA {r.get('mtta_sec', 0):.2f}s  point {point:.2f}  FA/min {fa_case} {fa_per_min(D, curves, fa_case):.2f} "
                      f"normal {fa_per_min(D, curves, 'normal'):.2f}", flush=True)


def _line(r, extra=""):
    return (f"Score_B {r.get('score_b', 0):.3f} AP {r.get('ap', 0):.3f} F1 {r.get('f1_alarm', 0):.3f} "
            f"(P {r.get('alarm_precision', 0):.2f} R {r.get('alarm_recall', 0):.2f}) mTTA {r.get('mtta_sec', 0):.2f}s{extra}")


def cv_stack(args, D, fold_of, folds, base_curves, fa_case) -> None:
    """Grouped CV of the stack (same folds / alarm-point rule / metric as the single models)."""
    spans = clip_spans(D["g"])
    att, rel = _smoothing(args)
    oof = np.zeros(len(D["g"]))
    for k in range(args.folds):
        tr = [sp for gi, sp in enumerate(spans) if fold_of[gi] != k]
        te = [sp for gi, sp in enumerate(spans) if fold_of[gi] == k]
        mlps, tcns = fit_stack(D["X"], D["y"], tr, args.stack, args.l2[0])
        p = predict_stack(mlps, tcns, D["X"], te, args.tcn_weight)
        oof[folds == k] = p[folds == k]
        print(f"  fold {k}: {args.stack} MLPs + {args.stack} TCNs", flush=True)
    if args.oof_out:
        np.save(args.oof_out, oof)
    raw = {gi: (t, smooth(s, att, rel)) for gi, (t, s) in curves_by_clip(D, oof).items()}
    point = choose_point(D, raw, args.fa_target, fa_case)
    model_curves = {gi: (t, calib(s, point)) for gi, (t, s) in raw.items()}
    for mode in ("model", "max"):
        for R in sorted({0.0, args.refractory}):
            curves = {gi: (t, refractory(t, s if mode == "model" else np.maximum(s, base_curves[gi][1]), R))
                      for gi, (t, s) in model_curves.items()}
            r = part_b(D, curves)
            print(f"stack {args.stack}+{args.stack} w {args.tcn_weight} smooth {att}/{rel} {mode:5s} refractory {R:>4}s: "
                  + _line(r, f"  point {point:.2f}  FA/min {fa_case} {fa_per_min(D, curves, fa_case):.2f} "
                             f"normal {fa_per_min(D, curves, 'normal'):.2f}"), flush=True)
    print(f"alarm point for --fit: --alarm-p {point:.2f}")


def _smoothing(args):
    """--attack/--release, defaulting to 0.6/0.12 for the stack and 1.0/1.0 (none) for a single model."""
    d = (0.6, 0.12) if args.stack else (1.0, 1.0)
    return (args.attack if args.attack is not None else d[0]), (args.release if args.release is not None else d[1])


def _trained_on(D, what):
    return (f"{sum(i['case'] == 'collision' for i in D['info'])} timed TAD crashes, "
            f"{sum(i['case'] == 'normal' for i in D['info'])} TAD normal clips, "
            f"{sum(i['case'] == 'sample' for i in D['info'])} sample videos; {what}")


def fit_stack_file(args, D) -> None:
    spans = clip_spans(D["g"])
    att, rel = _smoothing(args)
    mlps, tcns = fit_stack(D["X"], D["y"], spans, args.stack, args.l2[0])
    if args.alarm_p is None:
        raise SystemExit("--fit --stack needs --alarm-p from --cv --stack (held-out sample videos)")
    params = {"features": list(FEATURE_NAMES), "base_features": list(BASE_FEATURES),
              "mlps": [mlp_params(m) for m in mlps], "tcns": [tc[0] for tc in tcns], "tcn_weight": args.tcn_weight,
              "alarm_p": args.alarm_p, "attack": att, "release": rel, "refractory": args.refractory,
              "combine": args.combine,
              "trained_on": _trained_on(D, f"{args.stack} mlp16 + {args.stack} tcn (rf 63), l2 {args.l2[0]}")}
    MODEL_FILE.write_text(json.dumps(params))
    print(f"wrote {MODEL_FILE} ({args.stack} MLPs + {args.stack} TCNs, alarm_p {args.alarm_p:.3f}, "
          f"refractory {args.refractory}s, combine {args.combine})")


def fit(args) -> None:
    D = load_dataset(args.dataset)
    if args.stack:
        fit_stack_file(args, D)
        return
    args.attack, args.release = _smoothing(args)
    ok = D["y"] >= 0
    model = fit_model(args.models[0], D["X"][ok], D["y"][ok], args.l2[0])
    p = predict(model, D["X"])
    raw = {gi: (t, smooth(s, args.attack, args.release)) for gi, (t, s) in curves_by_clip(D, p).items()}
    fa_case = "sample" if any(i["case"] == "sample" for i in D["info"]) else "normal"
    # the alarm point comes from cross-validation (--alarm-p): chosen on these same clips the model has
    # seen, the sample videos would look quieter than new footage of the camera does
    point = args.alarm_p if args.alarm_p is not None else choose_point(D, raw, args.fa_target, fa_case)
    mean, std, w, b = model[:4]
    hidden = model[4] if len(model) > 4 else None
    params = {"features": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist(),
              "bias": float(b), "alarm_p": point, "attack": args.attack, "release": args.release,
              "hidden": {"w": hidden["w"].tolist(), "b": hidden["b"].tolist()} if hidden is not None else None,
              "trained_on": f"{sum(i['case'] == 'collision' for i in D['info'])} timed TAD crashes, "
                            f"{sum(i['case'] == 'normal' for i in D['info'])} TAD normal clips, "
                            f"{sum(i['case'] == 'sample' for i in D['info'])} sample videos; {args.models[0]}, l2 {args.l2[0]}"}
    MODEL_FILE.write_text(json.dumps(params, indent=1))
    print(f"wrote {MODEL_FILE} (alarm_p {point:.3f})")
    if hidden is None:
        for i in np.argsort(-np.abs(w))[:10]:
            print(f"  {FEATURE_NAMES[i]:18s} {w[i]:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "work" / "ext_annot" / "labels.json"))
    ap.add_argument("--caches", nargs="+", default=[str(ROOT / "work" / "ext_cache"), str(ROOT / "work" / "ext_cache_samples")])
    ap.add_argument("--dataset", default=str(ROOT / "work" / "risk_dataset.npz"))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--models", nargs="+", default=["logreg"], choices=["logreg", "mlp", "mlp32"])
    ap.add_argument("--fa-target", type=float, default=0.2)
    ap.add_argument("--alarm-p", type=float, default=None, help="alarm point from --cv (held-out sample videos)")
    ap.add_argument("--attack", type=float, default=None, help="smoothing attack (default 0.6 stack, 1.0 single)")
    ap.add_argument("--release", type=float, default=None, help="smoothing release (default 0.12 stack, 1.0 single)")
    ap.add_argument("--stack", type=int, default=0, help="K: K MLPs (16 hidden) + K causal TCNs (0: single model)")
    ap.add_argument("--tcn-weight", type=float, default=0.5, help="weight of the TCN mean in the stack")
    ap.add_argument("--refractory", type=float, default=10.0, help="stack: s after an alarm start without a new alarm")
    ap.add_argument("--combine", default="model", choices=["model", "max"], help="stack: report the model alone or "
                    "the larger of it and the hand-made cue score")
    ap.add_argument("--oof-out", default=None, help="--cv --stack: save the out-of-fold probabilities (.npy)")
    args = ap.parse_args()
    if args.stack and args.l2 == [0.01, 0.1, 1.0]:
        args.l2 = [1.0]
    if args.build:
        build(args)
    if args.cv:
        cv(args)
    if args.fit:
        fit(args)


if __name__ == "__main__":
    main()
