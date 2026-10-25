# Road-model observation in the CARLA neighbourhood

The observation pipeline records a fixed front camera, runs the locked morning-best
model on its images, and produces a replay with image, road reference, prediction and
disagreement panels. It is a recorded evaluation. Capture and inference run in
separate stages so CARLA and TensorFlow do not compete for the laptop GPU's memory.
The rule driver continues to use simulator state; the predictions never enter its
commands. The model is loaded once with `training=False` and the existing protocol.

The later asphalt/quality pass is documented in
[SIMULATION_QUALITY.md](SIMULATION_QUALITY.md). Its model camera and viewer share
the same disabled motion-blur, bloom and lens-flare settings. It adds exact
instance-ID evidence for decorative asphalt surfaces; older captures retain their
original protocol and measurements.

## Camera and alignment

RGB, raw semantic IDs and metric depth use the same rigid camera mount at
ego-relative `(1.4, 0.30, 1.35)` metres, 95-degree horizontal field of view,
960x540 resolution and zero radial lens distortion. They capture at 5 Hz while
physics remains at 20 Hz. The viewer camera can be changed independently.

Frames are paired by simulator frame number. Timestamp and camera-pose comparisons
must pass before a pair is saved. Each pair records the corresponding controller
decision, exact pose, source-image hashes and native class histogram. Queue drops
and unmatched images are retained in the capture summary. Replay uses the recorded
timestamps rather than assuming every displayed interval is identical.

RGB is resized directly with bilinear interpolation to the trained 512x512 input.
Reference labels use nearest-neighbour resizing. RGB normalization occurs once to
[-1,1]; prediction uses the locked >=0.5 threshold without filtering. Saved masks
remain 0/1. Class 255 means an excluded reference pixel, not a prediction class.

## Why the reference needs depth and geometry

The first actual check found **zero native road-tag pixels** on this generated
OpenDRIVE map. Its road and pavement meshes appear under native `Unlabeled` tag 0.
Treating that image as a road reference would produce a misleading score. The
original failed reference check remains in the local capture history.

Known native classes identify objects such as cars, people, buildings and sky.
Roads 1 and RoadLine 24 are positive where available. For unlabeled procedural
surfaces, a saved driving-lane footprint is built from OpenDRIVE waypoints sampled
every 0.5 m and rasterized at 0.1 m. The aligned depth camera projects visible
surface points into that footprint. A point on the flat road surface inside the
footprint becomes road; known actors remain non-road, preserving their occlusion.

Unlabeled geometry beyond 80 m and uncertain edges within 0.2 m of the footprint
boundary are excluded. The road-height tolerance is 0.08 m; near-map-road depth
residuals and flat-surface coverage are checked independently of model predictions.
This is a **geometry-assisted reference**, with finite raster/edge precision. It
is not a native semantic ground-truth repair or a human annotation. The replay
colors excluded pixels purple and reports evaluated-pixel coverage.

The footprint covers OpenDRIVE **driving lanes**. Parking bays and other paved
shared areas are outside that positive definition, even where they look traversable.
This class-definition difference and the approximate junction footprint can account
for some disagreements; a blue region is not automatically a model defect.

The map geometry, depth, calibration and source labels are saved so the reference
can be audited. Raw tags are preserved. The reference generation never reads model
predictions, and it does not change the rendered map or traffic controller.

CARLA documents [class IDs and raw depth encoding](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)
and [delayed camera delivery in synchronous mode](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/).
The procedural-road tagging limitation is also reported in the project's
[upstream issue](https://github.com/carla-simulator/carla/issues/4113).

## Repeat the pipeline

Capture a bounded, controlled episode using the existing owned-session launcher:

```powershell
.\.venv-carla\Scripts\python.exe setup/carla/run_session.py --india --smoke --observe-road --quality Low --speed 3 --case mixed --seed 7 --sim-seconds 110 --seconds 420
```

Use `-ObserveRoad` with `Start_Indian_Traffic.ps1` for an interactive capture.
Closing through Esc retains any captured evidence. A completed evaluation requires
clean session shutdown and a passed native rollout.

After the capture closes, pass its actual episode folder to the inference launcher:

```powershell
.\run_carla_road_evaluation.ps1 -Episode 'runs\YOUR_CAPTURE\episode_01_mixed_seed7' -OutputDir 'runs\NEW_COMPARISON'
```

`-Episode` accepts multiple episode folders. Keep rendering quality, reference
settings and scenario versions explicit when comparing results. This launcher uses
the existing WSL GPU inference environment and retained local model/data metadata.
A source-only clone needs those dependencies supplied separately.
Use `-WslDistribution`, `-WslPython` and `-Device cpu` to select another installed
inference environment explicitly; the defaults match the owner's verified GPU setup.
Install `requirements-camera-observation.txt` into a separate inference environment
when reproducing this evaluator. Its shared video-audit helpers import PyAV 16.1.0.
For WSL GPU use, also follow the existing CUDA/TensorFlow setup in the inference
guide. The CARLA capture environment remains the separate `.venv-carla` environment.

Open the generated `index.html`. Select a seed, Play/Pause, scrub through frames,
choose 1x to 3x playback, or inspect the eight lowest-IoU frames per scene.
The page includes all report data directly, so it needs no fetch request or server.

## Saved evidence and checks

Each capture has `road_camera/frames.jsonl`, raw RGB/class/depth files, camera
settings, map footprint and waypoint samples. The comparison contains every binary
prediction and reference, per-frame counts, class-level prediction fractions, exact
timestamps, comparison images, the eight worst-frame probability arrays, runtime
timings, source snapshots and an interactive replay.

`setup/verify_carla_road_metrics.py` tests ignored pixels, empty denominators and
visible-ground/actor-occlusion projection. `setup/verify_carla_road_result.py`
independently recounts saved PNG masks, checks the retained probability thresholds,
and queries sampled depth-projected road points against the saved OpenDRIVE map.
Checkpoint files, model weights in memory, selection metadata, split manifest and
completed test reports are fingerprinted before and after processing.

The score measures agreement with this simulator reference on selected synthetic
development scenes. The completed 689-image real-data test stays separate.
Capture throughput and isolated model-computation timings do not establish live
camera-to-control latency. Vehicle/pedestrian perception and the later bounded RL
policy still need their own implementation and comparisons.

## Measured first pass

The 20 October 2026 IST run processed 550 frames each from mixed seeds 7 and 42.
Pooled road IoU was **78.72%**, Dice **88.10%**, precision **81.73%**, recall
**95.53%**, and evaluated-pixel coverage **97.56%**. The weakest frames were near
48.1 seconds at the broad junction: **46.08%** and **46.50%** road IoU.
Model-call median times were 16.81/16.66 ms, excluding the rest of the pipeline.
The scored captures used Low rendering quality after an Epic native setup crash;
the failed evidence is retained. These related seeds are not independent samples.

The independent check recounted all 1,100 masks, queried 1,760 projected points
against the map with zero lane mismatches, and checked 177,762 known-actor pixels
for correct reference occlusion. Source model/protocol files and loaded weights
retained their fingerprints. Read [the portable measurements](results/camera_observation.json).
Full local evidence and replay are in `runs/carla_road_observation_20261020_01/`.

The generated player data and JavaScript syntax were checked, and representative
comparison images were inspected. Automated browser interaction with local-file
pages was blocked by the browser URL policy, so the playback controls were not
automatically exercised. No alternate browser surface or protocol was used to
bypass that restriction.
