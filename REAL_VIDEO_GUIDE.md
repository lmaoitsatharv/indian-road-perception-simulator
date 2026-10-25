# Real driving video: play, inspect and repeat

For the separate lean benchmark and reviewed-label protocol, see [streaming](STREAMING_GUIDE.md).
The current benchmark measures 52.25 fps without display/recording and no drops
on 30 fps file replay, with exactly unchanged masks. The original export-heavy
demo and its measurements below remain valid for that different workload.
The supplied 12 scoring-frame labels have now been evaluated: road IoU 90.79%
and pixel accuracy 97.43% against those labels. See
the reviewed-frame score report: `runs/real_video_reviewed_20261019_01/SCORES_REPORT.md` (local artifact, supplied separately)..
Whole-video accuracy remains unmeasured; these are selected development frames.

## Open the result

Open the real-video demo: `runs/real_video_20261019_02/index.html` (local artifact, supplied separately).. It contains
three playable original-versus-prediction MP4s. Use each video's normal Play,
Pause, seek and fullscreen controls. Playback follows the original 30 fps
timeline; this is an offline result, not a live camera feed.

The run report: `runs/real_video_20261019_02/VIDEO_REPORT.md` (local artifact, supplied separately). records the measured
speed, selected visual problems and limitations. Each example has three panels:
original frame, predicted binary mask and green road overlay. There is no
ground-truth mask panel for these clips.

## What is being run

- The existing locked **morning-best** U-Net/MobileNetV2 checkpoint, loaded once.
- Every decoded frame from all three external MOTOR clips, with original
  presentation timestamps. Actual decoded totals: 900 + 1,140 + 1,650 = **3,690
  frames**, **123 seconds**.
- Explicit direct bilinear resize from 1920x1080 RGB to 512x512. This changes
  proportions, matching the existing image inference convention; it is not a
  calibrated camera transform.
- Normalization exactly once: raw RGB /127.5 -1. Inference uses `training=False`.
- Threshold `>=0.5`. No smoothing, morphology, test-time augmentation or updated
  weights/BatchNorm. The display alone enlarges the binary mask with nearest
  interpolation and adds green to the original frame.
- A 3,072 MiB TensorFlow GPU pool limit and a 30-minute process wall-time limit.

The two longer source MP4 headers overstate their frame counts by six each.
Complete decoding verifies continuous 30 fps timestamps and durations of 38 and
55 seconds. The exporter records this discrepancy and uses the decoded frames;
it does not create extra frames to satisfy the incorrect headers. The initial
strict preflight failure remains in `runs/real_video_20261019_01`.

## Repeat on the same three source clips

Open PowerShell at the repository root and choose a fresh output folder:

```powershell
.\run_road_video_prediction.ps1 -OutputDir 'runs\my_real_video_01' -Resize
```

This uses the existing Ubuntu-24.04 GPU environment. Video decoding/encoding
uses **PyAV 16.1.0**, installed without dependencies in that environment; the
unchanged TensorFlow runtime loads the model. Existing output directories are
refused. A failed/stopped run keeps its partial files and `run_status.json`.
Creating `STOP.json` inside this new run folder requests a stop between frames.
This exporter does not control vehicles.

For a separately reviewed set of one to three clips, the Python entry point
accepts `--manifest`, using the same manifest format with a SHA256 for each
external video. The manifest must explicitly say there are no road ground-truth
masks. This exporter supports constant-rate, unrotated clips up to three minutes
and 3840x2160; unsupported timing is rejected rather than silently retimed.

## Result files

| File/folder | What it contains |
|---|---|
| `index.html` | Offline playback gallery; no internet needed for local videos |
| `config.json` | Source/license metadata, exact preprocessing, encoder and runtime configuration |
| `run_status.json` | Completion/failure state, timing, device and preservation checks |
| `source_video_metadata.json` | Full decoded source counts, PTS timeline and header discrepancies |
| `01_03_070/comparison.mp4` (and two other clip folders) | Original beside green predicted road, source-speed playback |
| `*/masks_01/frame_XXXXXX.png` | One lossless uint8 **0/1** mask per source frame |
| `*/per_frame.csv` | Frame index, source PTS/time base, relative time, inference time and temporal diagnostics |
| `*/examples/frame_XXXXXX/` | Selected RGB input, float32 probabilities, binary data, display previews and `frame.json` |
| `*/clip_result.json` | Measured per-clip statistics and selected frame pointers |
| `independent_verification.json` | Recount of every saved mask and output display timestamp |
| `source_snapshot/` | Exact inference code and final model protocol used for the run |

Example selection is fixed start/quarter/middle/three-quarter/end frames, plus
the minimum/maximum predicted road fraction and largest raw mask change.
These are review candidates, not an automatically measured failure ranking.
White 0/255 previews are for display; the actual data masks remain 0/1.

## Reading the measurements

**Inference time** includes the TensorFlow call and synchronized result copy;
it excludes decoding, input resize, normalization and output export. The
separate prediction-call timer includes normalization. The model is warmed up
before video timings are collected.

**Complete processing fps** also includes sequential decoding, resizing,
prediction, binary PNG writes, drawing, and MP4 encoding/flush. It excludes
model startup, preflight, selected-example export and subsequent verification.
Neither the MP4's 30 fps playback setting nor inference-only throughput proves
that the entire system runs in real time. The report also records total wall
time with startup and verification.

**Raw consecutive-mask change** compares predictions in adjacent camera frames.
Real camera motion, occlusion and changed viewpoints affect it. There is no
motion compensation, so it is not pure flicker, accuracy or ground-truth IoU.

## Further evaluation

Review and annotate a small, separately held-out real-video set with an explicit
road/kerb/obstacle definition. Add motion-aware temporal checks and benchmark a
lean live pipeline separately from the export-heavy diagnostic pipeline. Before
metre-based planning, establish camera calibration and independent actor/obstacle
observations, with braking/STOP/SLOW/PROCEED checks in simulation.

These recordings use a motorcycle-mounted camera; their appearance and geometry
differ from the training images and MATLAB renderer. Green road predictions are
not a safe-driving permission. No RL policy or camera-controlled car is delivered
by this video step. The existing final test result remains historical evidence;
the reserved dataset was not reevaluated or used to tune this step.

Source: [MOTOR](https://huggingface.co/datasets/varunpaturkar/MOTOR), Varun A.
Paturkar, Shankar Gangisetty and C. V. Jawahar, ICRA 2026. CC BY-NC 4.0; attribution
and non-commercial research use. Original MP4 bytes and source hashes remain
under `media/real_driving_20261017/`.

Timestamp handling follows the [PyAV project's encoding example](https://github.com/PyAV-Org/PyAV/blob/main/examples/numpy/generate_video_with_pts.py).
Local full-file decoding and timestamp recounts are the verification evidence.
