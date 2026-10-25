# Road-boundary and recording fixes — 20 October 2026

## Scene and reference

The same locked model scored **79.06% → 85.43% road IoU** across 1,100 simulator frames.
Raised paving with block joints now distinguishes footpaths from asphalt. The underlying driving-lane definition was retained.
On the selected 40.5-second frame, full-frame IoU changed from 77.05% to 83.03%.
On commonly evaluated pixels where both references agree, IoU changed from 77.32% to 83.11%.
The references disagree on 825 of 256,794 common valid pixels, reflecting visible-geometry/occlusion differences.

The independent audit recounted all 1,100 masks and checked 1,760 projected road points: none fell outside driving lanes.
It also checked 501,927 footpath pixels remain non-road, plus 4,607,057 known-actor pixels.

## Recording performance

Four analysis streams used to encode/save on the simulation tick thread. Their lossless writing now uses two bounded background encoders, with ordered commits and failure propagation.
RGB/semantic/instance pixels and float32 depth remain exact. No frame dropping or depth quantization was used for the speed gain.
A lower 1 Hz analysis-capture option is also available; it does not change the physics or viewer camera.

| Saved run | Paired frames | Simulated time / wall time |
| --- | ---: | ---: |
| Mixed seed 7 | 550 | 0.46× |
| Mixed seed 42 | 550 | 1.82× |
| Mixed seed 11 | 550 | 2.72× |

The final parallel-encoder run averaged 2.72× with steady windows near 2.95× at selected 3×. No paired frames were dropped.
It measured about 4.2 ms per world tick and 1.1 ms for traffic controls; image encoding was the substantial bottleneck.
The before/after runs use different seeds and interactive control histories, so the ratio is an observed throughput comparison, not a strict causal benchmark.

## Separate model candidate

Read `runs/road_boundary_adaptation_20261020_03/MODEL_REPORT.md` for the candidate's measured real-validation, synthetic-development and withheld comparisons. Trial _02 failed retention and remains preserved.
The training recording with the invalid overcast sky was rejected before optimization; `road_boundary_adaptation_20261020_01` preserves that preparation and its reason.
The three original model files and deployment lock remain protected. The new candidate uses a separate checkpoint and never opens the reserved real-data test images.

## Evidence and limits

- Full locked-model replay: `runs/carla_paving_observation_20261020_01/index.html`.
- Machine-readable report: `runs/road_boundary_fix_20261020_01/results.json`.
- Current user-facing simulation remains hand coded using simulator state; road predictions are not connected to control and no RL was trained.
- Reference, weather, same-town adaptation and close-clearance limits remain explicit in the saved reports.


### Measured model trials

| Trial / epoch | Decoder LR | Real validation IoU | Real boundary IoU (5px) | Simulator development IoU | Retention passed |
| --- | ---: | ---: | ---: | ---: | --- |
| road_boundary_adaptation_20261020_02 / 1 | 3e-06 | 92.009% | 54.585% | 84.747% | False |
| road_boundary_adaptation_20261020_02 / 2 | 3e-06 | 91.898% | 54.200% | 86.542% | False |
| road_boundary_adaptation_20261020_03 / 1 | 1e-06 | 92.095% | 54.978% | 82.892% | True |
| road_boundary_adaptation_20261020_03 / 2 | 1e-06 | 92.048% | 54.748% | 83.870% | True |

Latest completed trial: `runs/road_boundary_adaptation_20261020_03`. Selected epoch: **2**. The deployment lock is unchanged.
The frozen candidate's withheld simulator IoU was **80.392% → 81.646%** on 110 frames, decoded after selection.
On the known 40.5-second development example, the frozen candidate improved the new scene from 83.033% to 86.817% IoU. This example was not used to change selection.
Model gallery: `runs/road_boundary_adaptation_20261020_03/index.html`.

## Final native startup check

All 224 native footpath mesh bounds were checked after a static setup tick: top height 0.16 m and zero sampled driving-lane intrusions.
An initial assertion read stale actor bounds before world advancement and stopped startup. That failure was preserved; reading after the setup tick fixed the check, and the viewer launched successfully.
