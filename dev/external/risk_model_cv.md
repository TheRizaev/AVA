# Part B learned layer — cross-validation on third-party crashes

`python scripts/train_risk_model.py --build --cv --models mlp --l2 1.0 --fa-target 0.0`

Data: 200 timed crashes and 127 accident-free clips of the TAD benchmark (our timings,
`tad_labels.json`; 9 compilations of different crashes are excluded) plus the four sample videos
of the target camera (ordinary traffic, negatives). 331 clips, 75 657 analysed frames, 7 984
positive (the 5 s before a crash). Five folds, whole clips held out. The alarm point is set on the
held-out sample videos (the target camera) and then applied unchanged to the TAD clips.

| estimator (held-out clips) | Score_B | AP | alarm F1 | precision / recall | mTTA | false alarms / min, sample videos |
|---|---|---|---|---|---|---|
| hand-made cues only (before) | 0.122 | 0.01 | 0.28 | 0.48 / 0.20 | 0.3 s | 0.00 |
| learned network only | 0.373 | 0.22 | 0.58 | 0.45 / 0.79 | 2.7 s | 0.00 |
| **max(network, cues) — shipped** | **0.355** | 0.19 | 0.56 | 0.43 / 0.80 | 2.7 s | 0.00 |

Model: one hidden layer of 16 ReLU units on 52 standardised features (13 per-frame conflict and
kinematic features, scale-free, plus their max over 1 s and 3 s and deviation from the 3-s mean),
class-balanced cross-entropy, weight decay 1e-2, fast-attack / slow-release smoothing 0.6 / 0.12;
alarm point 0.77 = the lowest with no false alarm on the held-out sample videos. Final fit on all
clips: `--fit --models mlp --l2 1.0 --attack 0.6 --release 0.12 --alarm-p 0.77` ->
`assets/risk_model.json` (27 KB). Replayed on the sample videos, the shipped estimator raises one
alarm in 18.4 min.

Caveat: at this alarm point the network also alarms about 3 times per minute on the accident-free
TAD clips (other cameras, mostly motorways): the point is calibrated for the target camera, not
for them, so the TAD precision and F1 above are optimistic for other footage, while AP is not
affected by the threshold.
