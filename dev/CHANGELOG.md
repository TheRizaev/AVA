# Changelog

## 2026-09-27 — v1.0
- Score A 0.835 -> 0.9015 on the dev labels. Every change had to follow the official start/end
  convention or fix a logic bug, pass a leave-one-video-out check for any threshold and survive an
  independent re-run: illegal_turn ends once the vehicle has driven through cw3; solid_line_crossing
  ends when the rear wheels are over the line; congestion must be a jam of the main road; buses at the
  kerb are serving a stop; moving off on red+yellow is not red-light running; failure_to_yield is judged
  against the vehicle's remaining path. Three proposals were rejected as fits to the dev set. The same
  code scores 0.88 on the first 76 labels.
- 7 dev label corrections after a blind re-check (`label_corrections.json`): 81 events.
- Part B v2: 20 per-frame kinematic features, 5 MLPs + 5 causal temporal convolutions, the learned
  score alone and a 10-s pause between alarm starts. Held-out Score_B on TAD 0.470 (first version
  0.355, cues alone 0.122); no alarm in the 18.4 min of sample traffic.
- More crash data (SO-TAD, ACCIDENT) tried for Part B and rejected (`../README.md`).
- Demo: 2-s risk warm-up and a size hint for original 4K files. Site and demo on Hugging Face Spaces.
- Cleanup: broken and one-off scripts removed, `scripts/README.md`.

## 2026-09-26
- Part B learned anticipation layer (first version, a 16-unit network on 52 features); all 277 TAD
  accident clips timed (`external/tad_labels.json`).
- Ablations and class confusion on the website; team roster with contributions.

## 2026-09-25
- Dev labels completed (C3902 160–240 s annotated and verified; three illegal right turns found by
  the detector confirmed): 76 events on 18.4 min.
- Score A 0.787 -> 0.835 from one error analysis per weak class and a check of each change:
  jaywalking (hidden feet, re-linked fragments, island hops), stopped_vehicle (roadside-kerb
  distance, car-park apron), solid_line_crossing (old-lane evidence clear of the line); a
  failure_to_yield change was reverted after the check.
- failure_to_yield ignores pedestrians standing on the zebra (refuge / kerb end); jaywalking merge
  gap 6 s.
- Hazard pass every 1 s with a "crashed car" prompt, which became the accident rule.
- Part B: layout-free fallback for other cameras, id-jump history reset, frozen-frame guard,
  imminent-contact alarm (≤0.4 s to contact, closing ≥300 px/s, vehicles only).
- illegal_u_turn stays off: U-turns are permitted at this junction.
- Code review: 15 findings confirmed and fixed.
- Ablations (detector L/M/S, YOLO11-L, 960 px, 5 fps): the submitted setting is best; they exposed
  false classes under noisier tracks, so the kinematic accident branch was dropped and wrong_way
  needs 2 s / 100 px.
- Clean-environment install and run verified (identical output).
