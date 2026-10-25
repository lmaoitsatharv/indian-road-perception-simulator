# Lean video streaming and human-reviewed scoring

## Supplied labels scored — 19 October 2026

All 12 scoring frames and 6 diagnostic frames were supplied as reviewed labels.
The 12-frame result against those labels is **90.79% road IoU**, **95.17% Dice**,
and **97.43% pixel accuracy**, with 100% coverage. Diagnostic road IoU is 87.04%,
separate from the primary score. Score report: `runs/real_video_reviewed_20261019_01/SCORES_REPORT.md` (local artifact, supplied separately). ·
Comparison gallery: `runs/real_video_reviewed_20261019_01/index.html` (local artifact, supplied separately)..
These are hand-labelled development-frame measurements, not whole-video accuracy.

## What is complete

The full local gallery/report are recorded under `runs/real_stream_20261019_01/`. They are not distributed with source; see [the public results overview](docs/RESULTS.md).

The unchanged locked road model processed all three videos in two modes:

| Mode | Measured result | What it tests |
|---|---|---|
| Lean lossless file stream | **52.25 fps**, all 3,690 frames | Capacity without display, mask PNG writes or video encoding |
| Source-paced file replay | **29.92 processed fps**, all 3,690 frames, zero drops | Frames released on the original 30 fps schedule |

All **7,380 masks** exactly matched the existing saved predictions. Paced
release-to-result median was **29.20 ms**, p95 **33.34 ms**, maximum **58.81 ms**.
There were no results beyond the fixed 100 ms stale-frame budget in this run.

The earlier 16.23 fps diagnostic exporter also drew frames, saved every mask
and encoded MP4s. It measures a different workload. The lean benchmark does
not include a physical camera, display, recording or driving decisions; it is
not a full-camera real-time guarantee. No prediction quality changed.

## Repeat the bounded benchmark

Open PowerShell at the repository root:

```powershell
.\run_road_stream.ps1 -OutputDir 'runs\my_stream_benchmark_01'
```

Use a fresh output folder. It reuses the existing WSL GPU environment and the
verified three source clips. Model loading is once, input RGB normalization
is once, batch size is one and the threshold remains `>=0.5`.

The queue holds at most two waiting frames. Lossless mode waits when full;
source-paced mode drops the oldest waiting frame on overload. Frames older
than 100 ms are rejected before inference; late completions are marked. These
conditions are recorded rather than hidden. Nothing in this benchmark sends
commands to a car or MATLAB controller.

The run is bounded to 20 minutes and a 3,072 MiB TensorFlow GPU pool. Creating
`STOP.json` in that new result folder requests a stop. The benchmark does not start a driving controller. Exact setup, progress/failures and checkpoint
preservation checks are in `config.json` and `run_status.json`.

## Reviewed-label protocol

The clips lack dense whole-video road ground truth. Selected scoring and diagnostic frames were separately reviewed; the completed scores are reported above. The original local annotation pack under `runs/real_video_labels_20261019_01/` supports this procedure:

1. Enter a reviewer name or alias.
2. Draw a road polygon, fill the remaining area as non-road, then exclude
   visible vehicle/person bodies. Keep uncertain boundaries ignored.
3. Check the entire frame and tick **Reviewed**. Repeat for the 12 scoring
   frames. Six diagnostic frames are available separately.
4. Export the JSON and keep that file. No labels are uploaded anywhere.

The editor initially labels every pixel as unlabelled/ignored (255), not as a
model prediction. Road/non-road reference labels use 1/0; this separate ignore
value does not change the existing dataset or prediction masks, which stay 0/1.
The display restores the original 16:9 proportions while labels align with the
exact 512x512 model input.

The local prepared pack embeds its eighteen source images and is separate from the source template. Reference exports and autosaves use the documented mask schema; the provider images are not included in this public checkout.

The rule is **visible carriageway surface**, including lane/crossing paint,
shadows and worn pavement. Exclude sky, buildings, vehicle/person bodies,
raised footpaths and kerbs. Do not hallucinate the hidden road under an
occluding object. Leave ambiguous shoulder/edge pixels ignored. This is a
declared real-video review rubric, not a rewrite of the original IDD labels.

At least 80% of each reviewed frame must be labelled. All 12 fixed scoring
frames are required for a primary aggregate. The 6 model-informed diagnostic
frames are reported separately. The scorer never treats model predictions as
ground truth. Original images, masks and prediction hashes are checked before
scoring, and reviewed reference masks are stored in a new result folder.

After exporting, run this from the project folder, replacing the annotation
path with the JSON you actually saved:

```powershell
.\.venv\Scripts\python.exe road_video_labels.py score `
  --pack 'runs\real_video_labels_20261019_01' `
  --annotations '.\annotations\reviewed_video_labels.json' `
  --output-dir 'runs\my_reviewed_video_scores_01'
```

Read `SCORES_REPORT.md` for road IoU, Dice, pixel accuracy and label coverage.
`scores.json` contains exact counts/provenance; reviewed masks are saved too.
Incomplete scoring labels leave the primary score empty. Ignored pixels are
excluded and their proportion is reported.

These frames come from already inspected development clips. The score
describes **these reviewed frames**, not the entire videos or a new
independent test set. Do not tune the model on these labels while presenting
their score as independent evaluation. A new independent video collection is
still needed for a stronger generalization test.

The label format and scorer were checked separately from model evaluation. Manual annotation requires reviewing the entire visible frame and retaining an auditable export.

## Next integration gate

For further integration, test actual camera acquisition with display enabled and
measured capture-to-result age. Calibrate camera geometry and add independent
obstacle/actor observations before using road masks to steer the simulated
car. The MATLAB traffic behavior, braking layer and future RL experiment remain
separate from this perception pipeline.
