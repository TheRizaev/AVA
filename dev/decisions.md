# Judgement calls and the policy the detector follows

Each item: what is ambiguous, what the dev annotators did, and what `src/` implements.

## illegal_u_turn — disabled
About ten clean U-turns (approach → around the median tip → outbound) happen in 18 minutes of
footage. Nothing in view prohibits them (no sign or marking visible from the camera), and the
annotators rejected them as *illegal* U-turns or rated them low. The detector exists
(`trajectory_rules.illegal_u_turn`, origin/destination on the layout) but the class is not
reported: predicting a class that the test set does not contain costs a whole class of macro-F1.
The team confirmed (2026-09-25) that a U-turn at this junction is permitted, so it is not a violation.

## near_miss — disabled
Every hard-braking candidate was rejected on inspection: at this junction vehicles brake hard
all the time when joining a queue or yielding at a crossing, with no evasive manoeuvre and no
real conflict. The rule stays in the code, off by default.

## stop_line — start at the red onset
Vehicles already standing past the stop line when the signal turns red (typically because the
junction ahead is blocked) are stop-line violations from the red onset to green, as long as they
do not drive on into the junction during red (then they are neither stop_line nor red_light:
they crossed the line on green).

## failure_to_yield — a crossing pedestrian in the vehicle's path, measured along the crossing
The annotators count a pedestrian who is *on the zebra* (not waiting on the kerb at its end) and
close to where the vehicle crosses; they do not count people the vehicle passes behind after they
cleared its lane, people far along the crossing, or people standing on the zebra for a long time
(waiting at the refuge between the islands, standing at the kerb end of cw2). The rule measures
positions along the crossing's own axis (0..1 kerb to kerb, roughly metric): a pedestrian within
0.25 of the vehicle's crossing point, outside the 13 % kerb ends, who has moved at least 55 px
along the crossing within ±3 s (actually crossing, not standing), and is not walking away after
clearing the lane. Buses are excluded (their box covers far more than their path), and so are
mopeds pushed on foot. Consecutive vehicles cutting through the same group are one segment
(2 s merge gap), as the annotators merged them.

## jaywalking — about 1 m from the kerb, off the zebra
Not jaywalking: people within ~1 m of the kerb (loading a parked car, boarding at the bus stop),
people walking along the edge of a zebra, people between the stop line and cw1 (walking round
cars stopped on the crossing). Jaywalking: people cutting diagonally across the junction corner
from cw1 to the islands / cw3, or across the outbound road beside cw2. Tolerances scale with the
person's box height: >0.6 body heights inside the carriageway, >0.15 body heights from any zebra;
≥1 s; the event is extended back/forward to the moment the feet crossed the kerb; segments closer
than 6 s are merged (groups of pedestrians crossing one after another are one event).
Three tracking artefacts are repaired before the rule runs: a person box much shorter (<0.6×)
than a standing person at that image row has its feet hidden behind someone in front, so its
ground point is interpolated from the visible samples (or projected from the head at the ends of
a track); fragments of one person split by an occlusion (new id ≤1.5 s later within one body
height) are re-linked; and the kerb extension continues across ≤2 s walked over an island tip.

## stopped_vehicle — at the roadside kerb, traffic keeps passing it
A vehicle standing ≥10 s counts only while traffic next to it keeps moving (kerbside stop:
dropping off, loading). Queues of any kind (signal, junction, downstream jam) do not count. Boxes
cut by the frame edge are ignored. "Kerbside" means the roadside kerb: next to the median tip or an
island a standing car is a turner waiting for a gap (on the outbound road away from the junction the
median-side lane is a kerb lane again). The paved car-park apron at the east end of cw2 (layout
polygon car_park_apron_east) is off the road: cars wait there at the barrier.

## congestion — not a signal queue
With signal queues included every red phase would be "congestion" (13 events on the samples);
the rule reports only a jam that persists 10 s into green. No such jam occurs on the samples.

## accident — appearance only, zero tolerance for false alarms on ordinary traffic
Predicting `accident` on a test set without accidents would add a class with F1 = 0, so the rule
must stay silent on the 18 minutes of sample traffic (it does). It is the YOLOE "crashed car" prompt
≥ 0.45 in two consecutive samples (1 s apart) at one place on the carriageway; its peak on the
samples is 0.37 (single hit) and 0.31/0.33 (two hits at one place). The event is the 3 s around the
first hit: on 36 timed crashes of the TAD benchmark the prompt first fires ~0.5–1.5 s after the
contact and the vehicles are at rest ~1.5 s later (`dev/external/`). A kinematic branch (abrupt stop
next to another road user) was dropped: it found none of the 36 real crashes and fired on ordinary
traffic in every cheaper ablation setting (`dev/ablations.json`). For the same reason wrong_way needs
2 s and 100 px against the flow (1 s was reached by one noisy track at 5 fps).

## illegal_turn — right turns into the side street from any lane but the kerb lane
The rightmost approach lane is a dedicated right-turn lane (solid divider near the stop line); 19
of 23 right turns start from it. Right turns from lane 2 or 3 cut across it; the third check
confirmed all three such detector events that the first pass had missed.

## solid_line_crossing — the old lane must be seen clear of the line
A box that already straddles a divider is no evidence that the vehicle was in the old lane (tall
vehicles and boxes sliding onto a nearer car produced such "lane changes"). The rule needs a
sample clear of the line by half a car (capped per lane, the median lane's centre is only ~0.8°
from its divider); the event starts when the side reaches the line.
The event ends when the rear wheels are over the line too: the box bottom-centre is the vehicle's
front, and a vehicle changing lane is angled across the divider, so its rear crosses later - once it
has driven about its own length further, or where it stops.

## illegal_turn — end when the vehicle has driven through cw3
The official end is "vehicle completes the turn". The exit heading differs by path (~153° through
the slip lane, ~180° round island_3), so a heading threshold ended the junction-centre turns
seconds early; the turn now ends when the vehicle leaves the side street's crossing (cw3).

## congestion — a junction jam must be a jam of the main road
Cross-street cars queued in front of cw2 while pedestrians have their phase are a yield queue of
their own approach, not a jam; at least 3 of the ≥5 crawling vehicles in the junction must move
with the main road.

## stopped_vehicle — buses at the kerb are serving a stop
A bus standing at the roadside kerb with traffic passing it is a scheduled dwell at the stop on
the outbound road, not a stopped-vehicle incident (our annotators rejected every such case).

## red_light — moving off on red+yellow is an early start, not red-light running
Red+yellow lasts 3.0 s in every cycle and the queue front moves off with it (0.26-0.40 s before
green in 3 of 14 green phases on the samples); crossings in the last 1.0 s before green are not
reported.

## Dev label corrections
Windows where the rules and our first labels disagreed were re-checked blind (frames only, no
detector output); medium/high-confidence verdicts are applied by `scripts/apply_label_corrections.py`
from `label_corrections.json` (7 corrections; 2 low-confidence ones are listed but not applied).
