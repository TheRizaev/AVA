# scripts/

Development tools. The submission itself needs none of them: `run_submission.py` imports only
`solution.py` and `src/`. Every script runs from the repository root (`python scripts/<name>.py`)
and writes its intermediate files under `work/` (git-ignored).

## Assets the submission loads

| script | builds |
|---|---|
| `make_backgrounds.py` | median background per sample video and its registration to the reference (`work/bg`); the lighting bank `assets/bank/` and `assets/reference.jpg` were taken from it |
| `extract_tracks.py` | track tables of the samples (`work/tracks`) |
| `build_scene_maps.py` | flow field learned from those tracks (`assets/flow.npz`) and the occupancy maps of the EDA |
| `analyze_samples.py` | full cached analysis of each sample video (`work/cache/*.npz`): detection, tracking, registration, lamps, hazards; used by the evaluation and website scripts |
| `make_hazard_weights.py` | `weights/yoloe26l_hazards.pt` (YOLOE-26L with our text prompts baked in; needs internet once) |
| `ext_cache.py` | tracker rows of third-party crash clips and of the samples, for Part B training |
| `train_risk_model.py` | the Part B learned layer (`--build`, `--cv --stack 5`, `--fit --stack 5 --alarm-p P` -> `assets/risk_model.json`) |

## Evaluation on our dev labels

| script | does |
|---|---|
| `eval_dev.py` | runs the rules on the cached analyses and scores them with the official metric; lists every miss and false alarm |
| `export_metrics.py` | scores `predictions_samples.json` and writes `website/data/metrics.json` |
| `confusion.py` | class confusion of matched segments (`dev/confusion.json`) |
| `ablations.py` | detector / input size / frame-rate variants vs Score A and runtime (`dev/ablations.json`) |
| `check_determinism.py` | runs Part A twice in separate processes and compares the events |
| `apply_label_corrections.py` | applies `dev/label_corrections.json` to `dev/labels.json` |
| `annotate_tools.py` | contact sheets and zooms of a time window, used to label the samples |

## Part B and the accident rule on third-party footage

| script | does |
|---|---|
| `ext_sheet.py` | contact sheets of a clip, used to time the first contact of each crash |
| `eval_external.py` | scores Part B and the accident rule on the timed clips (`dev/external/`) |
| `tune_risk.py` | replays cached tracks through the hand-made cues and prints their distribution |

## Website and demo

| script | does |
|---|---|
| `plot_tracks.py`, `draw_layout.py` | EDA figures: every track on the background, the layout overlay |
| `build_site_data.py` | EDA and results data of the website (`website/data/`) |
| `render_samples.py` | annotated versions of the sample videos (`website/media/`) |
| `build_space.py` | the Hugging Face Spaces: `--static` for the site, without it for the Docker demo server |

Website order: `analyze_samples` → `extract_tracks` → `build_scene_maps` → `draw_layout` / `plot_tracks` →
`build_site_data` → `render_samples` → `export_metrics`, `confusion`, `ablations` → `build_space`.

## Benchmarks

| script | does |
|---|---|
| `bench_decode.py` | compares ways of decoding the 4K 10-bit 4:2:2 camera files |
