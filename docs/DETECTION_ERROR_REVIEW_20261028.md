# Detector error review and improvement plan — 28 October 2026

**What this is:** a measured breakdown of where the current YOLO11s detector loses accuracy, plus
inference-only setting tests and a ranked plan for improving it. **Nothing was trained, no default was
changed and no official test image was used.** All numbers below are **project validation results**
(17,675 images / 319,438 labelled boxes), the same set used to pick earlier checkpoints, so they are
development evidence, not an independent test score.

Checkpoint reviewed: `runs/detection_train_20261025_02/weights/best.pt` (epoch 3 of the 1280 trial,
sha256 `f7bb7e4c…`). Script: `setup/analyse_detection_size_brightness.py` (new; the older
`setup/analyse_detection_errors.py` is untouched).

## How it was checked

- The script wraps the same Ultralytics validator and settings as the matched 25 October evaluation
  (GPU FP32, 1280, conf 0.001, NMS IoU 0.7, max 300 boxes). Its overall scores reproduce the saved
  matched result (`matched_final_1280.json`) to within 0.0005 — they matched exactly.
- Per box it records class, size and whether it was found at confidence ≥ 0.25 with IoU ≥ 0.5
  (one-to-one matching). Ground-truth totals were checked against `preparation.json`.
- Before/after hashes: checkpoint, protected project files and the validation label cache are unchanged.
- Smoke tests first: 300 images (145 s, mostly one-time startup; the loop itself took about 5 s) and 100
  images (6.3 s). Full pass: 552 s (about 9 minutes), 10.9 ms GPU inference per image.

**Definitions.** *Size*: square root of the box area, scaled to a 640-pixel longest side — small < 32,
medium 32–96, large ≥ 96 (same bins as the older analysis). *Brightness*: mean brightness of the image
(0–255). It is only a rough stand-in for lighting, not a measured time of day or weather. *Recall* = share of
labelled boxes found; *precision* = share of predicted boxes that are correct (both at confidence ≥ 0.25).

## 1. Where the score is lost

### Small objects are almost the whole problem

| Size | Share of labels | Recall | Share of all misses |
| --- | ---: | ---: | ---: |
| Small | 79.3% | 51.9% | **94.4%** |
| Medium | 15.9% | 87.8% | 4.8% |
| Large | 4.8% | 93.6% | 0.8% |

Four out of five labelled objects are small (far away or partly hidden), and they account for 94% of
missed boxes. Medium and large objects — usually the nearby ones that matter most for braking — are
found 88–94% of the time. A box's size is not a measured distance, so "large ≈ near" is only a rough guide.

### Per class (confidence ≥ 0.25)

| Class | Labels | Recall | Precision | Small recall | Medium recall | Large recall | AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 34,085 | 55.1% | 75.7% | 49.3% | 87.3% | 90.6% | 63.2% |
| rider | 60,983 | 63.1% | 79.9% | 56.6% | 88.2% | 94.1% | 70.8% |
| car | 46,700 | 74.2% | 82.7% | 64.2% | 91.1% | 94.8% | 81.5% |
| motorcycle | 56,618 | 74.2% | 82.0% | 68.1% | 92.7% | 96.4% | 79.9% |
| bicycle | 764 | 50.4% | 65.8% | 36.1% | 78.9% | 73.0% | 51.4% |
| autorickshaw | 11,745 | 80.5% | 76.7% | 70.0% | 94.7% | 96.8% | 83.6% |
| bus | 5,425 | 73.9% | 81.2% | 51.6% | 86.6% | 96.7% | 80.7% |
| truck | 12,444 | 70.6% | 75.1% | 58.4% | 85.6% | 89.5% | 76.8% |
| animal | 963 | **4.7%** | 53.6% | 3.9% | 10.9% | 0/8 | **11.5%** |
| traffic light | 2,657 | 49.8% | 66.8% | 49.2% | 74.2% | 0/1 | 53.9% |
| traffic sign | 22,222 | 53.6% | 74.5% | 52.0% | 80.8% | 76.7% | 59.7% |
| other vehicle | 64,832 | **31.9%** | 53.1% | 30.1% | 58.3% | 74.4% | **34.0%** |

### Two classes drag the average down

mAP is a plain average over the 12 classes. **Animal (11.5%) and other vehicle (34.0%) pull it from
70.2% (the mean of the other ten classes) down to 62.2%.**

- **Animal — too little training data.** The training view has only **860 animal boxes, fewer than the 963
  in validation**; every other class has thousands. Even the 4× rare-class sampling cannot make up for
  that. 88% of validation animals are small, and for 91% of all animal labels no prediction overlaps them at all.
- **Other vehicle — large, mixed class.** It is the most common validation class (64,832 boxes, 20%) and
  causes **34% of all misses and 29% of all false detections** (18,215 of 62,127). It merges IDD's
  "vehicle fallback", caravan, trailer and train. 97% of its misses are small. A mixed class with unclear
  boundaries is a likely source of inconsistent labels. That is a hypothesis to check by looking at
  examples, not a measured fact.

### Misses are mostly "nothing detected", not "wrong class"

For missed boxes, the most common situation is that **no prediction overlaps the object at all** (for
example person 40%, traffic sign 46%, other vehicle 61%, animal 91% of all their labels). Class mix-ups
are small. The notable ones are: bicycle → motorcycle (10% of bicycle labels); bus → truck / other
vehicle (3% each); car → other vehicle (3%); person ↔ rider (3% / 1.5%). So the main gap is
*finding* small objects, not *naming* them.

### Crowded images are harder

| Labels per image | Images | Recall |
| --- | ---: | ---: |
| 0–9 | 7,698 | 69.2% |
| 10–19 | 3,699 | 61.5% |
| 20–39 | 4,077 | 59.4% |
| 40+ | 2,201 | 56.1% |

### Brightness is a minor factor

Overall brightness has a much smaller effect than size. Brightness thirds are narrow (cut points 91.8
and 101.6 on 0–255), so the darkest and brightest 10% (below 81.0 / above 110.8) are more telling:

- In the darkest 10%, small-car recall falls to 51.5% (65.9% in the middle 80%).
- In the brightest 10%, recall falls for bus (59.0% vs 73.7%), other vehicle (22.6% vs 32.0%) and
  traffic light (20.0% vs 49.5%, but only 25 labels).
- Brighter images contain most of the animals (434 of 963 in the brightest 10%), probably rural scenes.
  That mixes up brightness and scene type, so it is not evidence that light itself hurts.

## 2. Inference-only setting tests (no retraining)

Same checkpoint, same 17,675 validation images, GPU FP32. Only the setting in the first column changed.

| Setting | mAP50 | mAP50-95 | Recall (F1 point) | Small recall @0.25 | GPU ms/image | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1280, NMS 0.7 (current) | 62.245% | 41.525% | 55.78% | 51.9% | 10.9 | baseline |
| **1536**, NMS 0.7 | **64.162%** | **42.586%** | 57.73% | 56.1% | 15.7 | **+1.92 / +1.06 points** |
| 1280, NMS **0.6** | 62.768% | 41.512% | 56.48% | 51.7% | 11.0 | +0.52 / −0.01 |
| 1280 + test-time augmentation | 61.742% | 41.343% | 55.58% | 55.2% | 23.8 | worse: rejected |
| 1792, NMS 0.7 | 64.293% | 42.267% | 57.87% | 58.2% | 21.8 | diminishing: mAP50-95 falls, bus −4.1, animal −3.2 |
| **1536, NMS 0.6** | **64.813%** | **42.679%** | 58.37% | 55.9% | 15.5 | **best: +2.57 / +1.15 points** |

**Best setting, per-class AP50 change versus the current 1280 / NMS 0.7:** person +4.7, rider +3.8,
car +2.1, motorcycle +3.0, bicycle +2.3, autorickshaw +1.2, bus +0.3, truck +0.9, traffic light +3.9,
traffic sign +3.9, other vehicle +4.8; **animal −0.1** (11.5% → 11.4%). It is a near all-class gain,
unlike the 25 October training trial. GPU inference time rises from 10.9 to 15.5 ms per image
(about 42%; batch 2 versus batch 4, so approximate).

Going larger than 1536 stops helping: at 1792 the strict mAP50-95 falls, and large-object classes
(bus, truck) and animals get worse. A likely explanation (not tested) is that larger input makes
small objects easier to see while big objects become larger than the sizes the model learned at 1280.

**Caution:** these settings were chosen by looking at validation scores, so the best one will look
slightly better here than on genuinely new images. An honest final number needs a separate labelled set.

## 3. Ranked improvement plan

Ranked by (likely gain) ÷ (cost and risk), using the evidence above. "Planned" items are proposals;
none has been started.

| Rank | Idea | Evidence | Cost | Status |
| ---: | --- | --- | --- | --- |
| 1 | **Use 1536 input with NMS IoU 0.6 at inference** | Measured +2.57 mAP50 / +1.15 mAP50-95, 11 of 12 classes up, no training | No training; about 42% more GPU time per image | Measured; **adopting it as default needs your decision** |
| 2 | **Get more animal training data** | 860 train vs 963 val boxes; 4.7% recall | Data work; then a short fine-tune | Planned |
| 3 | **Audit `other_vehicle` labels** (review about 200 misses and false detections) | 34% of misses, 29% of false detections, mixed class | A few hours of review, no GPU | Planned |
| 4 | **Short fine-tune at the larger size** (3 epochs, like the 1280 trial) | 960→1280 gave a resolution gain, then training added +0.56 | About 3.7 h/epoch at 1280 → roughly 5–6 h/epoch at 1536; total ≈ 16–18 h | Planned; needs approval |
| 5 | **Tiled (sliced) inference for small objects** | Small objects are 94% of misses | Inference-only test; slower at run time | Planned |
| 6 | **Larger model (YOLO11m)** | Not measured in this project; usually stronger on small objects | Long training; may not fit well in 8 GB | Hypothesis |
| 7 | **Per-class confidence thresholds** for any live use | Precision/recall trade-off varies a lot by class | Cheap | Planned; does not change mAP |

Notes on each:

1. **Larger input.** The cheapest real gain, measured. It helps exactly where the losses are (small
   objects). The project target of 70% mAP50 is still not reached (64.8%), so it is a step, not a fix.
   Check the extra GPU time if the detector later runs live beside CARLA. 1792 is not worth it.
2. **Animal data.** Options: (a) check whether IDD images excluded during cleaning contain usable animal
   labels, keeping the existing leakage rules; (b) a small, targeted labelling pass; (c) an outside
   dataset. General datasets such as COCO include cows, dogs and horses, but they are not Indian-road
   imagery, so an Indian-road source is preferred.
3. **Other vehicle.** If the labels prove inconsistent, the options are relabelling, splitting the class,
   or ignoring very small boxes of that class in training. Any change to how the class is scored is a
   *different evaluation protocol* and must be reported separately, never as an accuracy gain.
4. **Fine-tune at 1536.** Only worth starting after items 1–3, so it learns from better data. On this laptop
   GPU, two earlier long runs hit CUDA errors, so keep FP32, saving every epoch and the matched before/after
   comparison.
5. **Tiled inference.** Splits each image into overlapping tiles so small objects look bigger. Test on
   validation the same way as section 2.
6. **Larger model.** Worth a later comparison, not first.
7. **Thresholds.** Needed for live observation (priority 2 in the project plan), where one global 0.25 is a
   poor fit.

**Not recommended:** test-time augmentation (measured worse), input larger than 1536 (measured
diminishing), repeating identical long training runs, or scoring on the official unlabelled test images.

## Decision recorded

Project decision: **1536 input / NMS IoU 0.6** becomes the recorded inference setting, to be written into
`models/perception_models.json` **at the start of the next training session, before any training**. It is
**not applied yet**; the recorded setting is still 1280.

**Update, 29 October 2026:** the setting is now written into `models/perception_models.json` as
`recorded_inference_setting` (1536 / NMS 0.6, validation mAP50 0.6481 / mAP50-95 0.42679). It was chosen on this same validation set,
so it remains development evidence, not an independent test. The deployed detector checkpoint is unchanged.

## Summary for the team

- Current detector: 62.245% mAP50 / 41.525% mAP50-95 (validation). Changing only inference settings
  to 1536 / NMS 0.6 measures **64.813% / 42.679%** on the same validation set, with no retraining.
- The real limits are small objects (94% of misses), very little animal training data (860 boxes) and
  the mixed `other_vehicle` class (34% of misses). Those need data work before more training is useful.
- No default has been changed. `models/perception_models.json` still records 1280 (at the time of this review; see the 29 October update above).

## Evidence

- Full review: `runs/detection_error_review_20261028_01/` (`analysis.json`, `per_image_counts.npz`, `console.log`)
- Setting tests: `runs/detection_setting_probe_20261028_{1536,nms060,tta1280,1792,1536_nms060}/`
- Smoke tests: `runs/detection_error_review_20261028_smoke`, `_smoke2`, `runs/detection_setting_probe_20261028_smoke1536`, `_smoke1792`
- Baseline: `runs/detection_pilot_recovery_20261025_01/matched_final_1280.json`
