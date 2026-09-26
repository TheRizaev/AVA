# Part B learned layer — cross-validation on third-party crashes

## Shipped: 20 features, 5 MLPs + 5 TCNs, learned score alone, 10-s alarm pause

`python scripts/train_risk_model.py --build`, then `--cv --stack 5 --fa-target 0.0`, then
`--fit --stack 5 --alarm-p 0.75` -> `assets/risk_model.json` (5 MLPs + 5 TCNs, 660 KB).

Same data, folds and alarm rule as the first version below (whole clips held out; alarm point =
the lowest with no false alarm on the held-out sample videos, applied unchanged to TAD).

| estimator (held-out clips) | Score_B | AP | alarm F1 | precision / recall | mTTA | FA/min, sample videos |
|---|---|---|---|---|---|---|
| first version: max(network, cues) | 0.355 | 0.19 | 0.56 | 0.43 / 0.80 | 2.7 s | 0.00 |
| stack, learned score alone, no pause | 0.459 | 0.34 | 0.66 | 0.56 / 0.81 | 2.9 s | 0.00 |
| stack, max with the cues, 10-s pause | 0.448 | 0.30 | 0.68 | 0.57 / 0.82 | 2.9 s | 0.00 |
| **stack, learned score alone, 10-s pause — shipped** | **0.470** | 0.34 | 0.70 | 0.61 / 0.81 | 2.8 s | 0.00 |

What changed and what each part is worth (5 replicates each, fold map of the protocol):
* 7 more per-frame features (time to contact and gap under constant acceleration, pedestrians /
  two-wheelers near moving cars, hard deceleration, sideways acceleration): +0.01 Score_B, AP
  +0.02 to +0.03 (also with the first seconds of every clip removed).
* Reporting the learned score alone instead of the max with the cue evidence: +0.02 on every seed.
* A 10-s pause after an alarm start (removes repeat alarms on one incident): +0.02 to +0.03.
* Causal temporal convolutions (16 channels, kernel 3, dilations 1–16) averaged 50/50 with the
  networks: the largest gain, but about half of it comes from the clip start (below).
* Ensembling 5 + 5 members mainly removes bad seeds (spread 0.006 -> 0.003).
Other fold maps give 0.467 and 0.473. The stack adds about 0.6 ms per analysed frame.

Caveat: TAD clips are cut around the crash (contact a median 3.8 s after the first frame), so a
model can partly learn "the clip has just started". With the first 62 analysed frames of every clip
removed (31 crashes left) the stack scores 0.31 against 0.24 for the first version.

## First version (26 Sep)

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
