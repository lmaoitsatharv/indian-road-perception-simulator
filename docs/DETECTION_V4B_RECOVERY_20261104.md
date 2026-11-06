# v4b recovery fine-tune: did bicycle and traffic light come back? (measured, validation, 4 Nov 2026)

**Question.** The v4 detector (`806a7163`) found far more land animals than the old one (`f7bb7e4c`) but gave back about 1 point of
bicycle AP50 and about 0.7 of traffic-light AP50 ([why](DETECTION_V4_BICYCLE_LIGHT_DROP_20261029.md)). v4b is a second, gentler
fine-tune started from v4 to try to win those back without losing the animal gain.

**Short answer.** v4b is **slightly better than v4 on the three classes we cared about, but it does not fully recover the old
bicycle and traffic-light scores.** Every difference here is under 1 point of AP50 except animal on the land-only set, comes from a
single run, and may be within run-to-run variation, which we have not measured. Treat it as "a little better, not clearly different", not as a
proven improvement. **Nothing has been deployed.** The deployed checkpoint is still `f7bb7e4c`.

## What v4b is

| Item | Value |
|---|---|
| Starting weights | v4 (`806a7163`) |
| Checkpoint | `4da08b09`, `runs/detection_train_20261029_02/weights/best.pt` |
| Training | 1 epoch, 1280 px, learning rate 1.5e-05 (half of v4), rare-class sampler boosts bicycle ×3, animal ×4, traffic light ×3 |
| Recipe | `runs/detection_v4b_recovery_prep_20261029_01/recipe.json` |

## Result: measured, validation set (17,675 images, 319,438 boxes)

All three checkpoints scored at the same setting: input 1536, NMS 0.6, same images, same labels. Numbers are AP50 in points.

| Class | Old `f7bb7e4c` | v4 `806a7163` | v4b `4da08b09` | v4b minus v4 | v4b minus old |
|---|---:|---:|---:|---:|---:|
| bicycle | 53.7 | 52.7 | 53.3 | +0.6 | -0.4 |
| traffic_light | 57.8 | 57.1 | 57.2 | +0.1 | -0.6 |
| animal | 11.4 | 20.0 | 20.8 | +0.7 | +9.4 |
| person | 67.9 | 67.9 | 68.0 | +0.1 | +0.1 |
| rider | 74.6 | 74.5 | 74.6 | +0.1 | 0.0 |
| car | 83.6 | 83.8 | 83.7 | -0.1 | +0.1 |
| motorcycle | 82.9 | 83.1 | 83.1 | 0.0 | +0.2 |
| autorickshaw | 84.8 | 84.8 | 84.5 | -0.2 | -0.3 |
| bus | 81.0 | 80.8 | 80.8 | -0.1 | -0.2 |
| truck | 77.7 | 77.7 | 77.8 | 0.0 | +0.1 |
| traffic_sign | 63.6 | 63.8 | 63.5 | -0.3 | -0.1 |
| other_vehicle | 38.8 | 38.6 | 38.4 | -0.1 | -0.4 |
| **all classes (mAP50)** | **64.81** | **65.41** | **65.47** | **+0.06** | **+0.66** |

Source: `runs/detection_v4b_eval_20261029_1536_nms060/analysis.json` (v4b), `runs/detection_v4_eval_20261029_1536_nms060/analysis.json` (v4),
`runs/detection_setting_probe_20261028_1536_nms060/analysis.json` (old).

## Result: measured, land-animal validation set `val_land` (17,603 images)

This set drops the animal boxes that are not land animals, so its animal score uses a **different protocol** from the table above.
Do not compare its animal number with the one above.

| Class | Old | v4 | v4b | v4b minus v4 |
|---|---:|---:|---:|---:|
| animal | 15.9 | 32.0 | 33.7 | +1.7 |
| bicycle | 53.8 | 52.9 | 53.4 | +0.6 |
| traffic_light | 57.8 | 57.1 | 57.2 | +0.1 |
| all classes (mAP50) | 65.21 | 66.43 | 66.56 | +0.13 |

Source: `runs/detection_v4b_valland_20261029/analysis.json`, `runs/detection_v4_valland_20261029_new/analysis.json`,
`runs/detection_v4_valland_20261029_old/analysis.json`. The animal labels are model-reviewed, not human-checked, so the animal numbers
are less certain than the other classes.

## How to read it

- **Bicycle:** v4b gets back about 60% of what v4 gave up (0.6 of 1.0 points) (-1.0 → -0.4 versus old). It is still below the old model.
- **Traffic light:** barely moved (+0.1 versus v4). Still 0.6 below the old model. The lower learning rate and the ×3 boost did not
  fix it.
- **Animal:** kept and slightly improved. This is the point of the whole v4 line.
- **Everything else:** within ±0.4. We have not measured run-to-run variation, so we cannot say whether differences this small are real.
- v4b has **not** been scored on the 20-image real-road holdout (68 boxes). That set is too small to separate differences this size, and
  we keep it out of tuning, so it should only be looked at once when a deployment decision is made.

## Not claimed

- No claim that v4b is better than v4. The gap is 0.06 points of overall mAP50, from one training run.
- No test-set, real-road, or deployment-ready claim. These are validation numbers only.
- The 1280 px score printed at the end of training (0.627 all-class AP50) uses a different input size and must not be compared with the
  1536 tables above.

## Decision still open

Deploy v4, v4b, or keep `f7bb7e4c`. The tradeoff is roughly +9 points of animal AP50 (validation) against about 0.4–0.6 points on
bicycle and traffic light. That call is not made here.
