# Work status

## Final state (2026-09-26)
- `predictions_samples.json`: official harness run on all four samples — 1.41–1.45× the video
  duration (budget 3×), format VALID; determinism checked.
- Dev score (official `evaluate.py`, all labelled classes, all four videos end to end,
  `scripts/export_metrics.py`): **Score A = 0.835** — red_light 1.00, stop_line 1.00, stopped_vehicle 1.00, congestion 0.89, illegal_turn 0.87, jaywalking 0.77, solid_line_crossing 0.62, failure_to_yield 0.54; micro F1@0.5 0.71. Part B not scorable on the samples (no accident in them); the
  official Part B output raises one alarm in their 18.4 min.
- Reported classes: `src/config.py` `ENABLED_CLASSES` (illegal_u_turn and near_miss off, see
  `dev/decisions.md`).
- Part B learned layer: a 16-unit network on 52 kinematic features, trained on the 200 crashes we
  timed in the TAD benchmark + ordinary traffic (`scripts/train_risk_model.py`,
  `assets/risk_model.json`); held-out Score_B on TAD 0.36 vs 0.12 with the cues alone, alarm point
  set for zero false alarms on held-out sample videos (`dev/external/risk_model_cv.md`). The accident rule,
  with the new "crashed car" prompt branch, reaches F1 0.36 / 0.22 / 0.11 at tIoU 0.3 / 0.5 / 0.7
  (was 0) and stays silent on the sample traffic.
- Website data, annotated videos and metrics regenerated from the final output.

## Changes of 2026-09-25
- Dev labels completed (C3902 160–240 s annotated and verified; three illegal right turns found by
  the detector confirmed): 76 events on 18.4 min.
- failure_to_yield ignores pedestrians standing on the zebra (refuge / kerb end); jaywalking merge
  gap 6 s.
- Hazard pass every 1 s with a "crashed car" prompt; accident rule branch (2).
- Part B: layout-free fallback for other cameras, id-jump history reset, frozen-frame guard,
  imminent-contact alarm (≤0.4 s to contact, closing ≥300 px/s, vehicles only).
- illegal_u_turn stays off: the team confirmed U-turns are permitted at this junction.
- Code review (3 reviewers + adversarial verification): 15 findings confirmed and fixed.
- Ablations (detector L/M/S, YOLO11-L, 960 px, 5 fps; `dev/ablations.json`): the submitted setting is
  best; they exposed false classes (kinematic accident branch, 1-s wrong-way) under noisier tracks,
  so the kinematic accident branch was dropped and wrong_way needs 2 s / 100 px.
- Space for the public site + demo: `deploy/hf_space/`, `scripts/build_space.py --upload USER/SPACE`
  (tested CPU-only locally). Clean-environment install + run verified (identical output).
- Score A 0.787 -> 0.835 from four error analyses (one investigator per weak class, frames of every
  error) and an adversarial check of each change: jaywalking (hidden feet, re-linked fragments,
  island hops), stopped_vehicle (roadside-kerb distance, car-park apron), solid_line_crossing
  (old-lane evidence clear of the line); the failure_to_yield change was reverted after the check.

## Changes of 2026-09-26
- Part B learned anticipation layer (above); all 277 TAD accident clips timed (`dev/external/tad_labels.json`).
- Ablations + class confusion on the website; team roster with contributions; Space deployment files.

## Dev labels
- `dev/labels.json` (76 events on 18.4 min) = the verified sweep events merged with
  `scripts/collect_labels.py` (the raw annotation log stays out of the repository) + detector-candidate
  checks `dev/fp_check_*.json`.

## Still needed from the team
- A public GitHub repository (tag the submission commit) and hosting for the site + demo
  (e.g. a Hugging Face Space running `uvicorn demo.app:app`); then fill `website/data/links.json`.
