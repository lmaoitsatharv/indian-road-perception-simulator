# Simulator adaptation with protected real predictions — 21 October 2026

## Current decision

Real photos, videos and unknown sources keep the **unchanged locked morning-best
model**. Simulator experiments have a separate, explicit inference option. No
image-colour heuristic decides which model to use, and neither model controls
the traffic actors.

The strongest simulator checkpoint was chosen using the same 60 development
frames, before inspecting the fresh confirmation scores. It is the existing
`runs/road_boundary_adaptation_20261020_02/epoch_02.keras`, reused for this
restricted purpose. That checkpoint previously failed real-image retention.
It is **not approved as a replacement for the real-road model**.

## Fresh simulator confirmation

Clear mixed-traffic seed 137: 110 matched frames at 1 Hz, no drops. All models
received the same RGB images and geometry-assisted reference. Uncertain edges
and distant unknown pixels were ignored, with 98.1868% pixel coverage.

| Model | Road IoU | Dice |
|---|---:|---:|
| Locked morning best | 85.5243% | 92.1974% |
| Previous experimental candidate 03 | 88.4814% | 93.8887% |
| Frozen simulator-only choice | **91.3765%** | **95.4939%** |

The simulator-only choice gains 5.8521 percentage points over the locked model
and 2.8951 points over the previous candidate on this recording. This is a new
seed in the same synthetic town, not a real-road test or a calibrated safety
result. Reusing an existing checkpoint is not a new training accuracy gain.
This recording is now inspected; it cannot serve as an untouched future check.

## New training trial: rejected for shared use

The new trial rehearsed 1,024 real training images alongside 110 synthetic
training frames. A frozen copy of the locked model supplied teacher logits
only for those real training images. The encoder and every BatchNorm layer
were frozen. Two decoder epochs used learning rate 0.000001, distillation
weight 4, threshold 0.5 and a single RGB normalization to [-1, 1].

| Measurement | Matched locked baseline | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|
| Real validation road IoU, 687 images | 92.1262% | 92.1259% | 92.1242% |
| Real validation boundary IoU, 5 px | 55.1879% | 55.1877% | 55.1758% |
| Synthetic development road IoU | 81.7624% | 82.7921% | 83.3029% |
| Reviewed real-video road IoU, 12 scoring frames | 90.7925% | 90.7575% | 90.7522% |
| Reviewed real-video road IoU, 6 diagnostic frames | 87.0451% | 86.9214% | 86.8672% |

Every guard group requires both road IoU and boundary IoU to equal or exceed the
matched baseline. Integer ratios determine the decision, without rounding
allowances. **Both epochs failed; no shared replacement was selected.** The
reviewed videos are repeated development guards, not independent whole-video
accuracy measurements. Validation and reviewed-video images never enter training.

Epoch 2 improved road IoU on all 60 synthetic development frames but worsened
361 of 687 real validation images. Those examples and both failed checkpoints
remain saved. Aggregate scores alone cannot guarantee every image or unseen
real road improves.

## How the real path is protected

- `idd_inference.py`, `predict_road.py` and `predict_road_video.py` are byte-identical
  to the task's protection snapshot and still load the locked morning-best model.
- All three original checkpoints, deployment lock, model catalog and split
  manifest are unchanged. No reserved 689-image test image was opened or evaluated.
- `idd_simulation_inference.py` accepts a hashed, simulator-only model manifest.
  Only the saved CARLA evaluator exposes the opt-in `--simulation-model` argument;
  its default also remains the locked model.
- A candidate is never selected from an input image's appearance. Complete native
  CARLA capture metadata and matching camera-file hashes are checked before
  simulator evaluation.

Independent checks recounted all 2,295 rehearsal masks, including 5 px boundary
counts with a separate integral-image implementation. They also recounted all
330 fresh model masks. Sampled depth-to-map checks found no road points outside
driving lanes, and known actors and raised footpaths remained non-road in the
reference. The original checkpoint and real inference code were checked again
after the work.

## Evidence

- All 110 fresh five-column comparisons: `runs/model_retention_20261021_01/index.html` (local artifact, supplied separately).
- Full confirmation report: `runs/model_retention_20261021_01/REPORT.md` (local artifact, supplied separately).
- New training report and exact guard decisions: `runs/road_rehearsal_20261021_01/MODEL_REPORT.md` (local artifact, supplied separately).
- All 18 reviewed frames plus real-image gains and regressions: `runs/road_rehearsal_20261021_01/index.html` (local artifact, supplied separately).
- [Portable result summary](results/model_retention.json)

All synthetic scores use the declared depth/OpenDRIVE reference. Native custom
road semantic tags remain absent; this reference has raster and class-definition
limits. The original completed real test result does not apply to simulator weights.
