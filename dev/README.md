# Dev set: our labels of the four sample videos

`labels.json` is in the organizers' ground-truth format
(`{"<video>.MP4": {"duration", "fps", "events": [[start, end, label], ...]}}`) and is what
`scripts/eval_dev.py` / `evaluate.py --gt dev/labels.json` score against.

## How the labels were made

1. **Blind sweep.** Each video was split into 64–80 s windows; for every window an annotator went
   through contact sheets at 1 frame/s (with an enlarged inset of the vehicle signal head burned
   into every tile), then zoomed on the stop line, the three crossings and the carriageway at
   0.2–0.5 s steps (`scripts/annotate_tools.py`) and listed every event of the short classes that
   starts in the window. A second pass per video covered the long classes (stopped_vehicle,
   congestion, road_obstacle, fire_smoke) at 4 s steps.
2. **Independent verification.** Every reported event was re-examined by a second annotator who
   defaulted to rejecting it unless the evidence was clear, corrected the class if needed and
   re-timed start/end to ~0.2 s following the task's start/end conventions.
3. **Merge.** Verified events were merged per class exactly like the organizers' convention
   (simultaneous events of one class = one segment).

The annotation brief with the class conventions and scene facts is `BRIEF.md`.

## Coverage

All four videos are labelled end to end (18.4 min, 81 events after merging). Of 127 first-pass
events 32 were rejected by the independent re-check, and 14 detector events that the first pass
had missed (lane changes over the solid dividers, jaywalkers, three right turns from the wrong
lane) were confirmed by the third check and added (`fp_check_*.json`). Windows where the rules and
the labels still disagreed were re-checked blind (frames only, no detector output); the 7
medium/high-confidence corrections (3 missed jaywalkers, a lane change split into its two lane
changes, a missed lane change, one failure_to_yield removed under the refuge policy and one added)
are in `label_corrections.json` and applied by `scripts/apply_label_corrections.py`.

## Caveats

These are our labels, not the organizers'. Judgement calls that the official annotators may have
made differently are listed in `decisions.md` together with the policy the detector follows.
