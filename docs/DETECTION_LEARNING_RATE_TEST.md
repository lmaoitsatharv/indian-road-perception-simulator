# Paired learning-rate screen

This completed development experiment compared **0.000030** with **0.0000375 (+25%)**, changing only learning rate between its arms. It did not justify changing the main rate.

## Protocol

Both arms used copied, hash-checked 256-image training and 192-image validation cohorts, identical frozen epoch-30 initialization, augmentation, optimizer, precision and sixteen optimizer updates. Hashes of augmented input tensors verified matching inputs. Raw FP32 EMA/recovery states avoided an export-precision mismatch.

The CPU screen ran alongside a separate GPU trainer. Batch 1 / nominal batch 16 and limited coverage differ from a full run. The selection gate was declared before candidate results. Sample results are screening evidence, not full-validation scores or an independent test.

## Results

| Arm | Sample mAP50 | Sample mAP50-95 |
| --- | ---: | ---: |
| Current rate, 0.000030 | 62.242% | 42.612% |
| Higher rate, 0.0000375 | 62.169% | 42.543% |

Neither sample mAP50 exceeded the 63.233% starting reference. The higher rate showed no convincing benefit and was not promoted. The separate full pilot's current result is recorded in [the matched report](DETECTION_RESOLUTION_PILOT_RESULT.md).

Full arm reports, the predeclared review policy, selection/input hashes and dashboard capture remain local under `runs/detection_lr_probe_20261025_01/`; they are not required source files. The experiment did not automatically deploy a checkpoint or enable vehicle control.
