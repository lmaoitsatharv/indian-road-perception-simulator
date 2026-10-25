# IDD object-detector training report — 24 October 2026

This report preserves the first completed 640-input detector result. The same weights later scored **55.87% mAP50 at 960 input** in matched validation, followed by a completed 30-epoch refinement and three-epoch 1280 pilot. Read [the improvement analysis](DETECTION_IMPROVEMENT.md) and [latest matched result](DETECTION_RESOLUTION_PILOT_RESULT.md).

**YOLO11s completed all 20 training epochs and the full final validation. Epoch
20 was selected as the best checkpoint.** Final validation mAP50 is **47.26%**
and mAP50–95 is **31.24%**. Cars, autorickshaws and buses are the strongest
classes. Pedestrian recall and the animal/other-vehicle classes need improvement.

The worker received SIGKILL after printing the complete final results and ended
with return code **−9** at **09:44:28 IST**. The cause is not established. This
was not a clean process exit. Both saved checkpoints were independently reopened,
all their model tensors were finite, and the selected model passed GPU forward
and native-image inference checks. The recorded scores and usable weights are
preserved; the original failed exit record has not been rewritten.

Machine-readable results: [detection_training.json](results/detection_training.json).

## What was trained

This is the separate **real-image object detector**, which learns labelled
bounding boxes for people, riders, vehicles, animals and traffic objects in
Indian road scenes. It is separate from both trained road-segmentation models.
It does not estimate distance in metres, track identities or control the car.

| Item | Recorded value |
| --- | --- |
| Run | `runs/detection_train_20261024_02/` |
| Model initialization | Official COCO-pretrained YOLO11s |
| Prepared dataset | `dataset/idd117k_detection_v2/` |
| Training entries / boxes | 67,159 / 1,042,884 |
| Validation images / boxes | 17,675 / 319,438 |
| Official unlabelled test images | 24,644; excluded from training and scoring |
| Classes | 12 |
| Input / batch | 640 pixels / 4 |
| Optimizer / initial learning rate | AdamW / 0.0001 |
| Maximum epochs / early-stop patience | 20 / 5; all 20 completed |
| Seed / workers / AMP | 42 / 4 / enabled |
| Mosaic | Enabled initially, closed for the last five epochs |
| Runtime | Isolated WSL; PyTorch 2.9.1+cu128, Ultralytics 8.3.225 |
| GPU | RTX 4060 Laptop GPU |
| Framework training and epoch-validation elapsed time | 8 h 43 min 25 s; excludes startup and final checkpoint validation |
| Successful optimizer steps recorded | 27,388 |

Original image sizes were retained. Runtime resizing preserves aspect ratio and
transforms boxes with the image; training uses padding/mosaic augmentation.
Different native resolutions did not require stretching the saved data.
The executed recipe and arguments are preserved with the run. The recipe's
original preparation-stage notes are historical; its settings remain unchanged.

## Final validation scores

These are the final selected-checkpoint results on the **validation set**, using
the project's filtered split and custom 12-class mapping. They are not an
untouched test result or an official IDD benchmark score.

| Metric | Score | Meaning |
| --- | ---: | --- |
| Precision | **65.48%** | Mean per-class correctness of retained detections at the evaluator's selected operating point |
| Recall | **42.72%** | Mean per-class share of labelled objects found at that operating point |
| mAP50 | **47.26%** | Mean detection score across confidence thresholds, requiring at least 50% box overlap |
| mAP50–95 | **31.24%** | Stricter mean score over box-overlap thresholds from 50% to 95% |

Precision and recall are macro averages across classes, not object-count-weighted
rates over all 319,438 annotations. Their operating point is selected by the
evaluator's F1 procedure; no deployment confidence threshold has been tuned.
mAP50 is not a claim that 47.26% of objects were detected.

### Per-class results

All scores below are percentages. Per-class values come from the final log,
which prints approximately three significant digits. Counts are exact.

| Class | Validation boxes | Precision | Recall | AP50 | AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Person | 34,085 | 70.7 | 37.9 | 44.2 | 24.9 |
| Rider | 60,983 | 73.6 | 47.6 | 54.1 | 31.8 |
| Car | 46,700 | 79.3 | 63.7 | 70.4 | 52.0 |
| Motorcycle | 56,618 | 73.9 | 60.1 | 64.7 | 38.8 |
| Bicycle | 764 | 55.6 | 32.5 | 34.5 | 22.0 |
| Autorickshaw | 11,745 | 69.7 | 70.5 | **71.9** | 55.4 |
| Bus | 5,425 | 74.1 | 64.2 | 69.7 | 54.6 |
| Truck | 12,444 | 69.1 | 59.4 | 64.3 | 48.0 |
| Animal | 963 | 64.2 | **5.71** | 17.3 | 8.06 |
| Traffic light | 2,657 | 52.2 | 24.9 | 25.4 | 11.9 |
| Traffic sign | 22,222 | 63.2 | 34.5 | 37.2 | 21.5 |
| Other vehicle | 64,832 | 40.2 | **11.7** | 13.3 | 5.8 |

Rider refers to a person riding a vehicle; the motorcycle/bicycle is a separate
object class. Other vehicle merges the source fallback/caravan/trailer/train
categories under the documented project mapping.

For the goal of identifying nearby people and vehicles, this is a first trained
candidate. Major vehicle classes are stronger than pedestrian detection, but
person recall of 37.9% is a clear weakness at the reported operating point.
There is no nearby-only score, distance calibration or tracking evaluation yet.
Object size, occlusion and lighting need a measured error breakdown before
attributing the missed detections to a particular cause.

## Learning progress and plots

| Checkpoint in training | mAP50 | mAP50–95 |
| --- | ---: | ---: |
| Epoch 1 | 35.76% | 23.35% |
| Epoch 5 | 44.47% | 29.22% |
| Epoch 10 | 46.14% | 30.46% |
| Epoch 15 | 46.90% | 30.97% |
| Epoch 20, epoch-validation CSV | 47.27% | 31.24% |
| Final saved-best-checkpoint validation | 47.26% | 31.24% |

From epoch 1 to epoch 20, mAP50 improved by **11.51 percentage points** and
mAP50–95 by **7.89 points**. Epoch 1 is already trained; this is not a comparison
with an evaluated COCO baseline. Training and validation losses both decreased;
validation improvement was still positive but slow near epoch 20. More training
has not been tested and is not guaranteed to fix the weaker classes.

The final validation reran the stripped saved checkpoint. Its small numerical
differences from the epoch-20 CSV are retained rather than mixed into one result.
The pinned framework selects checkpoints by validation mAP50–95, whose maximum
was at epoch 20. Stored checkpoint metrics and all 20 history rows corroborate
that selection.

Training and validation curves: `runs/detection_train_20261024_02/results.png` (local artifact, supplied separately).

Normalized confusion matrix: `runs/detection_train_20261024_02/confusion_matrix_normalized.png` (local artifact, supplied separately).

The large background row highlights missed objects. This matrix uses its own
matching/confidence settings; its rounded cells are not the precision/recall
table's selected operating point.

The curves, normalized confusion matrix, and the first validation label/prediction
montages were visually inspected. The first montage contains adjacent night
frames and is illustrative, not a representative sample of the whole dataset.

- Precision–recall curves: `runs/detection_train_20261024_02/BoxPR_curve.png` (local artifact, supplied separately).
- Validation labels: `runs/detection_train_20261024_02/val_batch0_labels.jpg` (local artifact, supplied separately).
- Validation predictions: `runs/detection_train_20261024_02/val_batch0_pred.jpg` (local artifact, supplied separately).

## Saved checkpoints and verification

Use **`runs/detection_train_20261024_02/weights/best.pt`** for subsequent detector
experiments. `last.pt` is preserved separately. Both are 19,168,346 bytes.

| File | SHA-256 |
| --- | --- |
| `best.pt` | `57f1d54b93c794677d26fd22773284de2752d878477bb9ac04a761cc7f035153` |
| `last.pt` | `1b466db45741ef125e88c66a0f8b20f3e9cbe43753eba02fe545cd6ff8fe2380` |

Both files have the expected 12 class names and 499 finite model-state tensors.
The best checkpoint passed a GPU forward check with output shape `1 × 16 × 8400`
and a native 1920×1080 validation-image inference check with finite, bounded boxes.
This smoke check verifies usability, not an additional accuracy estimate.
Checkpoint hashes were unchanged afterward; reporting performed zero optimizer
steps. The framework stripped optimizer state and set stored `epoch` to −1
during finalization. The embedded 20-row history preserves the completed epoch;
these are usable inference/fine-tuning weights, not exact optimizer-resume files.

All **17 protected fingerprints** were independently rehashed after the worker
ended and match the pre-training values. The real-road and simulation-road
checkpoints, inference code/defaults, original split metadata and rule driver
remain separate. The COCO initialization file and saved training recipe are
unchanged. The temporary system-wake request was released at 09:44:34 IST.

Verification evidence:

- Checkpoint reload and GPU checks: `runs/detection_train_20261024_02/report_verification_02/checkpoint_verification.json` (local artifact, supplied separately).
- Post-run protected-file checks: `runs/detection_train_20261024_02/completion_protection.json` (local artifact, supplied separately).
- Final validation log excerpt: `runs/detection_train_20261024_02/final_validation_excerpt.txt` (local artifact, supplied separately).
- Full epoch history: `runs/detection_train_20261024_02/results.csv` (local artifact, supplied separately).
- Original worker exit: `runs/detection_train_20261024_02_launcher/exit.json` (local artifact, supplied separately).

`status.json` and `batch_progress.json` retain stale training snapshots because
SIGKILL bypassed trainer cleanup. Read this report and the original launcher exit
together; do not interpret those stale files as an active training process.

## Limits and next work

The dataset audit removed exact train/validation pixel overlap and recording
overlaps under the documented identity policy. It retains 7,914 repeated-pixel
training entries across source identities: 59,245 distinct training pixel images
in 67,159 entries. Source annotations can contain class noise and inconsistent
ordering/labels; every possible near-duplicate was not certified absent. Read
[the preprocessing report](IDD_DETECTION_IMPORT.md) for the selection audit.

The animal class has only 1,778 training boxes; person has 140,187, so weak
pedestrian recall cannot simply be attributed to having few person labels.
The broad other-vehicle grouping also requires source-label review. No per-size,
occlusion or day/night breakdown has been computed yet.

1. Audit missed people, riders, animals and other vehicles on validation images,
   then measure performance by object size, occlusion and lighting where labels
   permit. Review the broad other-vehicle mapping and source-label examples.
2. Select confidence thresholds from validation precision–recall tradeoffs for
   the observation demo. Measure the complete camera-to-display latency;
   batched validation reported 1.7 ms inference plus 0.1 ms preprocessing and
   1.1 ms postprocessing per image, which is not a live-camera FPS measurement.
3. Choose any longer/higher-resolution or data-refinement experiment from those
   errors, in a new run. Reserve a separate labelled recording for an independent
   final evaluation; the current validation set has already selected the model.

No new training, test-set scoring, detector-to-control connection or RL was
started while preparing this report.
