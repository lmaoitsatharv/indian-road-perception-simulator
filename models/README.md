# Models and evaluation scope

This directory contains architecture, selection and checkpoint-verification metadata. **Checkpoint binaries and datasets are not included in the repository.** The [current perception manifest](perception_models.json) lists the active model roles, hashes, input contracts and result references. The [retained road-model catalog](retained_models.json) records the real-road validation references; the [real inference lock](segmentation_final_lock.json) fixes the selected checkpoint and protocol.

Read [Getting started](../docs/GETTING_STARTED.md) for prerequisites, [Results](../docs/RESULTS.md) for evaluation scope and the [documentation index](../docs/INDEX.md) for technical reports.

## Current model roles

| Role | Architecture and input | Latest measured result | Scope |
| --- | --- | --- | --- |
| Locked real-road segmentation | U-Net decoder with MobileNetV2 encoder; 512x512 RGB; one normalization to [-1,1]; threshold 0.5 | **91.6549% road IoU / 95.6457% Dice** | 689 real test images; checkpoint and protocol fixed before test access |
| Separate simulation-road segmentation | Same visual architecture with validity-masked loss/metrics; 512x512 RGB; threshold 0.5 | **99.9288% road IoU / 99.9644% Dice** | 500 synthetic test images; **98.2174%** valid-pixel coverage |
| Separate object detector | YOLO11s adapted to 12 project classes; input setting 1280 | **62.245% mAP50 / 41.525% mAP50-95** | 17,675 project validation images / 319,438 objects; matched GPU FP32 evaluation |

The metrics describe different tasks and distributions and must not be compared as interchangeable accuracy scores. The detector uses a documented project class mapping and filtered validation membership; its result is not an official IDD benchmark or a labelled independent test. No model currently supplies vehicle-control commands. The traffic driver uses rules and simulator actor states; the live road preview is observation-only.

## Checkpoint identities

These paths identify the expected local artifact layout. They are not download links or files included in a clone.

| Role | Expected project-relative path | Selected epoch |
| --- | --- | ---: |
| Locked real road | `runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras` | 1 of its fine-tuning stage |
| Simulation road | `runs/simulation_frozen_20261022_01/best.keras` | 10 |
| Latest detector | `runs/detection_train_20261025_02/weights/best.pt` | 3 of its adaptation stage |

SHA-256 fingerprints:

- Locked real road: `93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725`
- Simulation road: `8d8c9fd7a3880bb69643c4973db8f14c46242c178fae6927d06b5f9d6aaaf8af`
- Latest detector: `f7bb7e4c5959520d797dcde5547fdcded9ece1e9651a3e7bd6ffa2273749182c`

Verify the exact artifact hash before loading. A matching project fingerprint establishes content identity with the recorded artifact, not that an arbitrary checkpoint is trustworthy. Load only artifacts from a trusted source; model formats and their deserializers may execute code.

## Real-road selection and retained references

The real-road network first trained a decoder above a frozen ImageNet-pretrained MobileNetV2 encoder. Separate fine-tuning experiments evaluated selected encoder blocks, augmentation, learning rate and BatchNorm behavior. The selected model uses the fixed preprocessing and probability threshold recorded in `segmentation_final_lock.json`.

| Retained real-road reference | Validation road IoU | Validation boundary IoU, 5 px | Role |
| --- | ---: | ---: | --- |
| Original | 92.0345% | 54.9087% | Validation reference |
| Selected real model | 92.1258% | 55.1886% | Locked real-image/video inference |
| Boundary-focused variant | 92.0538% | 55.2361% | Separate validation experiment |

These are pooled road-class validation scores over the same 687 images at threshold 0.5. Only the selected real model was evaluated on the 689-image real test set. Its test precision was 95.19% and recall 96.11%. The inspected test results must not be used to select a new model while continuing to describe the set as untouched.

Additional retained reference hashes are original `652c3e537e6b944430304d103a80bdff21a67daddcba1d9590e49dfbff0765c3` and boundary-focused variant `ee996bb45056b1bfa315330e0066bccf71ed86d6cdbb8bff55d6c93d1cba8c85`.

See [model and training details](../MODEL_AND_TRAINING.md), [real-model retention](../docs/REAL_MODEL_RETENTION.md) and the [portable segmentation result](../docs/results/segmentation.json). Synthetic adaptation did not satisfy strict real-retention checks, so simulator-only checkpoints do not replace the real inference default.

## Simulation model

The separate simulation model starts with an ImageNet encoder and a fresh decoder, using 4,000 synthetic training images and 500 validation images. Its validity mask excludes uncertain labels from both training loss and evaluation counts; validity is target metadata, not a model input.

The validation-selected epoch-10 checkpoint was fixed before evaluating the 500-image synthetic test. A separate 480-frame fresh-scene confirmation scored **99.7478% road IoU**, with **98.0143%** valid-pixel coverage. These constrained synthetic evaluations do not establish real-road transfer, city-independent generalization or collision-avoidance performance.

Recorded-CARLA and native-live observation use separate explicit scope manifests. Neither grants permission to replace real-road inference or feed predictions into vehicle controls. See [simulation training](../docs/SIMULATION_TRAINING.md), [test results](../docs/SIMULATION_TEST.md), [fresh confirmation](../docs/SIMULATION_CONFIRMATION.md) and [live observation](../docs/LIVE_SIMULATION_OBSERVATION.md).

## Detector

The detector's classes are person, rider, car, motorcycle, bicycle, autorickshaw, bus, truck, animal, traffic light, traffic sign and other vehicle. Their order and source mapping are defined by [detection_policy.json](../detection_policy.json). The clean training view has 56,729 unique retained training images; validation remains 17,675 images.

The latest three-epoch 1280 trial starts from the completed 30-epoch refinement checkpoint. Compared with that parent using the same full validation, resolution, precision and standalone evaluation settings, it gains **0.557 percentage points mAP50 / 0.400 points mAP50-95**. Bicycle, autorickshaw and traffic-light AP50 decline slightly. Animal AP50 remains weak at 11.487%; the candidate is not an all-class improvement and was not automatically deployed.

See [the matched detector result](../docs/DETECTION_RESOLUTION_PILOT_RESULT.md), [data import](../docs/IDD_DETECTION_IMPORT.md), [clean-view refinement](../docs/DETECTION_IMPROVEMENT.md) and [portable matched evidence](../docs/results/detection_resolution_pilot.json). Official unlabelled test images were not used as negatives or locally scored for accuracy.

## Loading and reproducibility

- Supply trusted weights, prepared data and required local provenance separately. A configuration file or dependency installation does not provide them.
- Use the existing inference loaders so checkpoint identity, input normalization and scope checks remain in place. For road inputs, do not normalize RGB twice; masks stay binary 0/1.
- Keras custom loss/metric objects are defined in `idd_model.py`, `idd_boundary.py` and the simulation-specific modules. Use the task's loader rather than treating all checkpoints as interchangeable.
- Fine-tuning requires an explicit trained parent and a fresh run directory. An inference checkpoint may omit optimizer state and therefore may not support exact training continuation.
- Keep the real segmentation, simulator segmentation and detector recipes/evaluation splits separate. Model-to-control integration, destination navigation and bounded RL remain planned work.

The repository includes [model configuration](model_config.json), [current role metadata](perception_models.json), [retained real references](retained_models.json) and [the real protocol lock](segmentation_final_lock.json). Large model caches, checkpoints and full experiment galleries remain external artifacts.
