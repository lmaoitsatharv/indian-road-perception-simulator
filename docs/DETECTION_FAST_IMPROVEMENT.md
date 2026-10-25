# Fast detector improvement check — 25 October 2026

**The strongest quick candidate measured here is 1280-pixel inference with the
epoch-23 checkpoint.** On a fixed 192-image validation sample, it improves mAP50
by 2.24 percentage points and mAP50–95 by 2.65 points over 960 input. Person AP50
improves by 5.63 points. At the screening stage, full validation was still required: animal AP50 fell and CPU inference took 1.67 times as long. The subsequent [full resolution comparison](DETECTION_FULL_RESOLUTION_20261025.md) and [matched training pilot](DETECTION_RESOLUTION_PILOT_RESULT.md) are complete.

This screening was performed while the separate 960-pixel parent trained. It used copied
checkpoints and CPU-only evaluation, performed no optimizer steps, and did not
change the training recipe or inference defaults. All protected fingerprints
and both diagnostic checkpoint copies passed their final integrity checks.

## Epoch 3 and the full validation history

These scores use the unchanged **17,675-image validation set**. The starting
reference is the original IDD-trained detector evaluated at 960 pixels; epoch
numbers below belong to the subsequent 30-epoch refinement run.

| Model | mAP50 | mAP50–95 |
| --- | ---: | ---: |
| Starting reference, 960 input | 55.87% | 37.17% |
| Refinement epoch 3 | 51.34% | 33.44% |
| Refinement epoch 11 | 56.13% | 37.21% |
| Refinement epoch 20 | 57.64% | 38.35% |
| Refinement epoch 23 | **57.75%** | **38.46%** |

Epoch 23 gains **6.41 / 5.02 percentage points** over epoch 3, but only
**1.88 / 1.30 points** over the starting reference. Much of the early run
recovered its initial score drop; epoch 11 is the first completed epoch above
the starting reference on both metrics. Epochs 20 to 23 add just **0.11 / 0.11
points**, so extending the same recipe indefinitely has weak recent returns.

Epoch 3's score row is retained. Its actual weights were not retained in the
periodic checkpoints or recovery copies, so this check cannot rerun that exact
model. The periodic save interval is five epochs. Epoch 3 is not evidence for
rolling back from the current best model.

## Paired resolution screening

The probe contains **192 copied validation images and 4,225 labelled objects**,
with a fixed seed and at least eight images containing each class. Selection
then fills uniformly to 192. This deliberately improves rare-class coverage;
the resulting scores are **development screening, not the full validation
score or an independent test result**. Bicycle has only 13 boxes and animal
27 boxes in this probe.

Both latest-checkpoint variants use the same frozen epoch-23 weights, images,
labels, 12 classes, FP32 precision, batch 1, confidence floor 0.001, NMS IoU 0.7
and maximum 300 detections. Input resolution is the only changed evaluation
setting. The original starting model is also evaluated on the same sample.

| Same 192-image sample | mAP50 | mAP50–95 | CPU inference per image |
| --- | ---: | ---: | ---: |
| Starting model, 960 | 57.24% | 37.92% | 177.5 ms |
| Epoch 23, 960 | 60.35% | 39.54% | 193.8 ms |
| Epoch 23, 1280 | **62.59%** | **42.19%** | 323.2 ms |

CPU timings were measured during active GPU training. They are not GPU speed
benchmarks or camera-to-display latency. The 1280 setting improves sample AP50
for 11 of 12 classes, but this is not a universal improvement:

| Class | Epoch 23 at 960 | Epoch 23 at 1280 | AP50 change, points |
| --- | ---: | ---: | ---: |
| Person | 55.29% | 60.92% | +5.63 |
| Rider | 63.96% | 68.11% | +4.15 |
| Car | 78.72% | 81.43% | +2.71 |
| Motorcycle | 77.06% | 81.03% | +3.97 |
| Bicycle | 33.21% | 37.38% | +4.16 |
| Autorickshaw | 87.18% | 90.42% | +3.24 |
| Bus | 77.60% | 79.01% | +1.40 |
| Truck | 75.35% | 76.60% | +1.26 |
| Animal | 40.68% | 24.99% | **−15.69** |
| Traffic light | 53.70% | 57.81% | +4.10 |
| Traffic sign | 54.41% | 60.24% | +5.83 |
| Other vehicle | 27.01% | 33.09% | +6.08 |

At fixed confidence 0.25 and matching IoU 0.5, small-person recall improves from
**124/329 (37.69%) to 148/329 (44.98%)**. The small bin is defined from native
box area after normalizing the image's longest side to 640; it does not measure
physical distance. At approximately 75% person precision, sample person recall
rises from 47.33% to 54.45%. Confidence choices fitted on this sample need wider
validation and do not improve mAP by themselves.

## Follow-up recommendations recorded at screening

1. **Validate 1280 on the full unchanged validation set after the GPU is free.**
   Compare the same final checkpoint at 960 and 1280, including per-class
   results and actual GPU latency. Investigate the animal regression before
   selecting an all-class default. This is the quickest measured route to
   better detection without another training run, at the cost of more work
   for each image. The probe does not establish a new full-set score.
2. **Use short, controlled three-epoch pilots for the next training change.**
   Start from a copied current best checkpoint, measure its baseline first,
   save every epoch, and change one factor at a time. Compare against that
   baseline rather than counting recovery from an initial drop as net progress.
   A three-epoch pilot screens candidates; it does not guarantee convergence.
   The parent recipe already disabled mosaic for its final five epochs; doing that again would not be a new factor.
3. **Target the remaining misses with more varied, correctly labelled Indian
   training scenes.** Prioritize small/occluded people, bicycles, animals and
   ambiguous other vehicles. The current training set has only 120 unique
   animal-containing images; its existing 4× sampling boost repeats those
   scenes. More repetition alone does not add visual diversity. Preserve
   validation membership and audit new training labels before short pilots.
4. **If 1280 is too expensive, screen sliced inference on hard scenes.**
   This can preserve small-object detail through image crops, but needs
   multiple model passes and a merge step. It remains an untested candidate
   here, not a speed improvement. Benchmark larger training batches only in
   a separate short throughput check when the GPU is free; batch changes
   should be isolated and should not be assumed to improve accuracy.

Increasing maximum detections is not the first priority: the validation cache
contains at most 120 labelled objects per image, below the existing cap of 300.
False positives can still occupy prediction slots, so this is a prioritization
decision rather than proof that the cap can never matter.

## Evidence and documentation

- Complete paired probe and integrity result: `runs/detection_fast_probe_20261025_01/comparison.json` (local artifact, supplied separately).
- Fixed selection, image and label hashes, epoch-3 score: `runs/detection_fast_probe_20261025_01/selection.json` (local artifact, supplied separately).
- Full validation learning curve: `runs/detection_train_20261024_04/results.csv` (local artifact, supplied separately).
- Original full-set resolution analysis: `runs/detection_improvement_20261024_01/analysis.json` (local artifact, supplied separately).
- [Training recipe](../detection_refinement_clean_recipe.json)
- [Ultralytics validation settings](https://docs.ultralytics.com/modes/val/)
- [Ultralytics sliced inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/)

Runtime checks use the project's pinned Ultralytics 8.3.225. Current online
documentation may describe newer releases. The official unlabelled test set
was not used in this check. Real and simulation road models remain separate.
