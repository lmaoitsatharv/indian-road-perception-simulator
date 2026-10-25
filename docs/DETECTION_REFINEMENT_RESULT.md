# Detector refinement result — detection_train_20261025_02

Completed 3 epochs; selected epoch 3. Training process exit code: 0. Both saved checkpoints reopen with finite tensors.

Same 17,675-image project validation set at 1280 input. Original detector weights and all 17 protected road/inference/driver fingerprints are unchanged.

mAP50: **62.24%** (+0.56 percentage points against the starting parent at the same resolution).

mAP50–95: **41.52%** (+0.40 points).

| Class | AP50 before | AP50 after | Change, percentage points |
| --- | ---: | ---: | ---: |
| person | 62.88% | 63.25% | +0.37 |
| rider | 70.63% | 70.77% | +0.14 |
| car | 80.58% | 81.51% | +0.92 |
| motorcycle | 79.84% | 79.92% | +0.08 |
| bicycle | 51.49% | 51.38% | -0.11 |
| autorickshaw | 83.86% | 83.60% | -0.26 |
| bus | 79.27% | 80.69% | +1.42 |
| truck | 75.55% | 76.80% | +1.25 |
| animal | 9.06% | 11.49% | +2.43 |
| traffic_light | 54.23% | 53.85% | -0.37 |
| traffic_sign | 59.28% | 59.70% | +0.42 |
| other_vehicle | 33.58% | 33.99% | +0.41 |

This three-epoch run used FP32. The independently matched standalone before/after comparison is documented in [the pilot report](DETECTION_RESOLUTION_PILOT_RESULT.md); use that report for the exact 62.245% / 41.525% result and its fixed evaluation settings. These are development validation scores, not an independent test or nearby-distance/tracking evaluation. No automatic deployment or vehicle control occurred.

Read [the machine-readable result](results/detection_refinement.json) for individual gains/regressions and checkpoint hashes.
