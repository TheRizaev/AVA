# Annotation protocol — dev set of the four sample videos

We label traffic events in the sample videos of one fixed CCTV camera over a large signalised
intersection in Tashkent. The labels are the ground truth we tune and evaluate the rules on, so
precision matters more than speed; an annotator who is not sure says so in the confidence field
rather than guessing. How the passes were organised (blind sweep, independent re-check, check of
detector candidates, blind re-check of disagreements) is in `README.md`.

## Tools

Frames come from the 1080p proxies of the videos (`scripts/annotate_tools.py`, run from the
repository root):

```
# contact sheet: 3x3 grid, one frame per --step seconds, timestamps burned in
python scripts/annotate_tools.py sheet <VIDEO> <t0> <t1> [--step 1] [--cols 3]
# zoomed region (crop = x1 y1 x2 y2 in 1920x1080 frame pixels), finer time steps
python scripts/annotate_tools.py sheet <VIDEO> <t0> <t1> --step 0.25 --crop 200 350 1000 700
# one large frame, optionally cropped
python scripts/annotate_tools.py frame <VIDEO> <t> [--crop x1 y1 x2 y2]
```
Each command writes a JPEG and prints its path. VIDEO is one of C3896 (340 s), C3897 (318 s),
C3902 (318 s), C3905 (128 s). A 9-frame sheet at `--step 1` covers 9 s. Full-frame tiles are 640 px
wide, so small things (pedestrians' feet, lane markings) need a `--crop` zoom. Every uncropped
tile has an enlarged inset (top right) of the vehicle signal head: top lamp red, middle yellow,
bottom green (in bright sun the lit lamp is dim — compare the lamps).

Reference pictures: the scene layout drawn on each video's background (`scripts/draw_layout.py`):
orange `approach` = inbound carriageway (traffic comes toward the camera, top left to bottom right),
blue `outbound` = carriageway going away (right to top left), grey `junction`, green `cw1/cw2/cw3` =
zebra crossings, magenta = pedestrian islands, white = median, red `stop_zone` whose upper edge is
the approach stop line, yellow circles = signal lamps; and the flow field of normal vehicle motion
(`scripts/build_scene_maps.py`).

## Scene facts
* Main road: divided carriageway from top-left. Approach traffic queues at the stop line
  (white line just above cw1) and then goes straight to the bottom-right, or turns right
  into the lower-left side street (through cw3), or left.
* Traffic from the right side of the frame goes up-left through cw2 into the outbound carriageway.
* Vehicle signal head on the median tip (~x=1157,y=373..407) = main-road phase; cycle ≈ 75 s:
  red (~40 s) → red+yellow → green → flashing green → yellow → red.
  A pedestrian head on the gantry pole (x≈267, y≈495..512) is in phase with it.
* Pedestrians normally use the sidewalks, cw1/cw2 (across the main road, while the main road is red)
  and cw3 (across the side street, while the main road is green).

## Classes (use these ids exactly) and the annotators' start/end conventions

| id | definition | start | end |
|---|---|---|---|
| accident | contact between road users, or road user and fixed object | first frame contact visible | all involved stop moving or leave frame |
| near_miss | sharp braking or swerving to avoid a collision; no contact | onset of evasive action | road users clear of each other |
| red_light | a vehicle crosses the stop line while its signal is red | front of vehicle crosses stop line | vehicle leaves the intersection or the frame |
| wrong_way | vehicle moves against the traffic direction of its lane, incl. driving in the oncoming lane | vehicle enters opposing lane | returns to correct lane or leaves frame |
| illegal_u_turn | U-turn where markings/signs prohibit it | vehicle starts turning | completes the turn |
| stopped_vehicle | vehicle stationary on the carriageway >= 10 s, NOT in a queue at a signal | vehicle stops | moves again or is removed |
| jaywalking | pedestrian on the carriageway outside a crossing | steps onto road | leaves the road |
| failure_to_yield | vehicle drives through a crossing while a pedestrian is on it or stepping onto it | vehicle enters crossing | vehicle leaves crossing |
| illegal_turn | turn from the wrong lane or in a prohibited direction | starts turning | completes turn |
| solid_line_crossing | lane change / manoeuvre across a solid marking | wheel crosses line | vehicle fully in new lane |
| stop_line | vehicle stops past the stop line on red without entering the intersection | vehicle stops | signal turns green |
| congestion | traffic at a standstill or crawling across all lanes of a direction | queue stops moving | queue clears |
| road_obstacle | debris, animal or fallen object on the carriageway | obstacle appears | obstacle removed |
| fire_smoke | visible fire or smoke from a vehicle or on the road | first visible smoke | smoke clears / frame ends |

Rules: one event = one contiguous segment of one class. Two simultaneous events of the SAME
class (e.g. two jaywalkers at once) = ONE segment covering both. Events running past the end of
the video end at the video duration. Times in seconds from the first frame, to ~0.25 s.
Pedestrians standing on a crossing, an island or the median are NOT jaywalking. Cyclists /
scooter riders are vehicles, not pedestrians. Passengers boarding a bus at the bus stop are not
on the carriageway. A normal queue waiting at the red signal is NOT stopped_vehicle.

## What each label records
For every event: label, start, end, confidence (high / medium / low), where (image region / lane /
crossing), and a one-line description of what was seen (vehicle colour and type, direction).
Doubtful cases that were looked at and rejected are listed too, with the reason: they help judge
false positives of the rules.
