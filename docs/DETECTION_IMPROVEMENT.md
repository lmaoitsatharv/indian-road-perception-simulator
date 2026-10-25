# Detector improvement — 24 October 2026

The first change is already measured: evaluating the existing detector at
**960-pixel input raises mAP50 from 47.30% to 55.87%** on the same full
17,675-image validation set. **All 12 classes improve in AP50.** This gain comes
from preserving more image detail; no weights changed during that comparison.

The separate clean-view refinement `runs/detection_train_20261024_04/` used the verified epoch-20 detector and the data described below. It completed 30 epochs; its final saved model scored 58.054% mAP50 / 38.638% mAP50-95 at 960. A later [1280 pilot](DETECTION_RESOLUTION_PILOT_RESULT.md) has its own matched result.
Initial setup `_03` was stopped
before any batch/optimizer execution when the annotation conflicts were confirmed;
its exit and restart reason remain preserved.
This page records the changes and evidence, separate from the historical
20-epoch report.

**Historical startup verification during epoch 1/30:** The first ten actual training
batches had completed seven optimizer updates with finite sampled losses. Peak GPU
allocation in those batches was 2.40 GiB (2.56 GiB reserved). Both framework
caches exactly match the audited images, negatives and every class's box count;
all 17 protected fingerprints and the starting detector weights are unchanged.
The actual augmented first-batch preview was visually inspected. These are
startup checks, not an additional validation score.

## Measured resolution comparison

Both evaluations used the same checkpoint, validation labels, 12-class mapping,
batch size, FP32 inference and detection settings. Only input resolution changed.
The small difference between the matched 640 result below and the original
47.26% report reflects evaluation precision/settings, not additional training.

| Metric | Matched 640 input | Matched 960 input | Gain, percentage points |
| --- | ---: | ---: | ---: |
| mAP50 | 47.30% | **55.87%** | +8.58 |
| mAP50–95 | 31.29% | **37.17%** | +5.87 |
| Mean precision | 65.32% | **70.96%** | +5.64 |
| Mean recall | 42.85% | **50.29%** | +7.43 |

| Class | AP50 at 640 | AP50 at 960 |
| --- | ---: | ---: |
| Person | 44.22% | **55.76%** |
| Rider | 54.20% | **63.93%** |
| Car | 70.37% | **77.20%** |
| Motorcycle | 64.71% | **74.39%** |
| Bicycle | 34.57% | **43.38%** |
| Autorickshaw | 71.94% | **78.60%** |
| Bus | 69.68% | **77.20%** |
| Truck | 64.37% | **72.46%** |
| Animal | 17.26% | **18.24%** |
| Traffic light | 25.57% | **37.11%** |
| Traffic sign | 37.26% | **48.99%** |
| Other vehicle | 13.39% | **23.18%** |

Full validation resolution comparison: `runs/detection_improvement_20261024_01/resolution_comparison.png` (local artifact, supplied separately).

Higher resolution requires more computation. These are development validation
results, not an independent test, camera-to-display latency or vehicle-control
evaluation. Individual size/threshold slices can regress despite improved AP.

## What the misses show

At 640 input, 84.8% of validation person boxes and 94.3% of other-vehicle boxes
fall in the small-object bin. The bin uses square-root box area after scaling
the native image's longest side to 640 pixels: small <32, medium 32–96 and large
≥96. It is an image-size diagnostic, not a distance-in-metres label.

At the fixed confidence 0.25 and matching IoU 0.5:

| Person box size | Labelled boxes | Recall at 640 | Recall at 960 |
| --- | ---: | ---: | ---: |
| Small | 28,914 | 26.91% | **39.69%** |
| Medium | 4,524 | 85.72% | **88.57%** |
| Large | 647 | **93.20%** | 92.58% |

Higher resolution recovers 3,695 additional small pedestrians at this threshold,
while finding four fewer large pedestrians. Across all person sizes, the
framework's selected-F1 recall rises from 38.04% to 47.58%; these are different
operating-point measurements from the fixed-threshold size table.

Threshold tuning alone did not attain the development target of 80% pedestrian recall at
75% precision: the 960 validation precision–recall curve gives about **47.44%
recall at 75.03% precision**, confidence approximately 0.245. Saved curves retain
the tradeoff rather than presenting increased false detections as a solved model.

The original training view has animals in only 396 entries, with 1,778 boxes;
95.2% of those boxes are small. Its 140,187 person boxes show that person misses
cannot simply be explained by too few labels. The clean view below retains 860
animal boxes in 120 unique images and 114,695 person boxes. Rare-image sampling
increases exposure, but does not create additional scene diversity.

## Training-data cleanup

An additional check confirmed that all 2,516 repeated-pixel groups with different
label serializations also differ after sorting class/box tuples and rounding
geometry to 0.1 native pixel. Of those, 1,911 disagree on per-class object counts;
the difference is not merely line ordering. The audit does not establish which
annotation version is correct.

The new `dataset/idd117k_detection_v3/` view therefore excludes all 5,045 entries
in those ambiguous groups, and removes 5,385 extra identical-pixel/identical-label
entries. It keeps one deterministic representative for each consistent group.
There are **56,729 unique training images / 879,434 boxes** and the unchanged
**17,675 validation images / 319,438 boxes**. Every retained manifest row equals
its original, and no label was automatically rewritten or merged. Source data
and both previous prepared views remain intact.

Selection verification confirms no repeated training pixels, no selected member
of an ambiguous group, exact validation membership/labels and no train/validation
pixel overlap. The independent audit reopens every saved label and compares
sampled source images/boxes. Image hard links preserve native pixels and avoid
another full image copy. The complete exclusion log is retained.

## Recorded clean-view refinement settings

The clean-view recipe is `detection_refinement_clean_recipe.json`. All
56,729 retained training identities stay eligible, with 56,729 primary-image draws
per epoch. Native images/labels and retained provider split membership are
unchanged; full validation membership is identical. The official 24,644
unlabelled test images remain excluded.

- **Start from trained weights:** verified `detection_train_20261024_02/weights/best.pt`.
  This is a new fine-tuning optimizer, not an exact continuation of stripped optimizer state.
- **Increase image detail:** 960 input instead of 640; batch 4 on the same GPU.
- **Continue learning:** up to 30 additional epochs, early-stop patience 8,
  AdamW at 0.0003, cosine decay, one warmup epoch and zero bias warmup learning rate.
- **Reduce aggressive shrinking:** mosaic probability 0.25 and scale range 0.3.
  Mosaic closes for the last five epochs.
- **Increase rare-image exposure:** bounded primary-image weights of 4× for
  animal images and 2× for bicycle/traffic-light images, combined by maximum.
  The sampler retains a defensive pixel-group weight correction; every group in
  the clean view contains one identity. Actual expected exposures are recorded
  in `training_sampling.json` when the loader is built.
- **Preserve recovery files:** full periodic checkpoints every five epochs,
  plus best/last checkpoints. Training and its final report use a fresh folder.

Sampling uses replacement: every identity has positive probability, but a given
epoch does not guarantee visiting every identity. Auxiliary mosaic selections
follow the framework's normal buffer behaviour. Validation is never weighted.

Both Windows and WSL passed eight refinement checks, covering sampling,
duplicate correction, validation exclusion, invalid boosts, actual metadata,
changed-provenance rejection and continued readiness of the original recipe.
A 4×960 GPU forward and native-size postprocessing check passed. The clean-view
sampling/provenance/modified-audit checks passed on both platforms. The complete
native source audit checked all 74,404 saved labels and 143 source pairs, with
maximum sampled coordinate error 0.0000009601 pixels. Source-versus-saved previews
from both releases were visually inspected and match. Actual backward memory and
optimizer execution are recorded by the first training batches.

## Evidence and progress

- Resolution/error-analysis proof: `runs/detection_improvement_20261024_01/analysis.json` (local artifact, supplied separately).
- All dataset object-size counts: `runs/detection_improvement_20261024_01/dataset_object_sizes.json` (local artifact, supplied separately).
- 640 validation diagnostics: `runs/detection_improvement_20261024_01/diagnostics_640.json` (local artifact, supplied separately).
- 960 validation diagnostics: `runs/detection_improvement_20261024_01/diagnostics_960.json` (local artifact, supplied separately).
- 960 forward readiness: `runs/detection_improvement_20261024_01/refinement_runtime/runtime_verification.json` (local artifact, supplied separately).
- Windows refinement checks: `runs/detection_refinement_checks_20261024_windows/verification.json` (local artifact, supplied separately).
- WSL refinement checks: `runs/detection_refinement_checks_20261024_wsl/verification.json` (local artifact, supplied separately).
- Duplicate annotation audit: `runs/detection_improvement_20261024_01/training_duplicate_annotation_audit.json` (local artifact, supplied separately).
- Clean selection verification: `dataset/idd117k_detection_v3/provenance/selection_verification.json` (local artifact, supplied separately).
- Clean data preparation: `dataset/idd117k_detection_v3/preparation.json` (local artifact, supplied separately).
- Independent saved-label/source audit: `runs/detection_clean_audit_20261024_02/verification.json` (local artifact, supplied separately).
- Source-versus-saved comparison gallery: `runs/detection_clean_audit_20261024_02/index.html` (local artifact, supplied separately).
- Clean-view Windows checks: `runs/detection_clean_refinement_checks_20261024_windows/verification.json` (local artifact, supplied separately).
- Clean-view WSL checks: `runs/detection_clean_refinement_checks_20261024_wsl/verification.json` (local artifact, supplied separately).
- Current launcher preflight: `runs/detection_train_20261024_04_launcher/preflight.json` (local artifact, supplied separately).
- Current training log: `runs/detection_train_20261024_04_launcher/training.log` (local artifact, supplied separately).
- [Verified training startup](results/detection_refinement_start.json)
- [Portable improvement summary](results/detection_improvement.json)
- Preserved setup restart reason: `runs/detection_train_20261024_03_launcher/startup_restart.json` (local artifact, supplied separately).

After clean-view startup, read `runs/detection_train_20261024_04/batch_progress.json`,
`first_training_batches.json`, `training_sampling.json`, `progress.json` and
`results.csv`. The worker automatically verifies saved checkpoints and writes
`docs/DETECTION_REFINEMENT_RESULT.md` and `docs/results/detection_refinement.json`
after training, including per-class improvements/regressions against the
original detector at the same 960 resolution. Its original process exit remains
recorded separately if cleanup fails.

The bounded system-wake helper followed the owned worker and released when it ended, without changing the power plan.

The earlier 70% mAP50, 45–50% strict mAP and higher recall/precision values are
development goals, not guaranteed outcomes. No candidate is automatically
deployed. Both road-segmentation checkpoints, real inference/defaults, original
detector and original training recipe remain separate.

The implementation was checked against the installed pinned Ultralytics 8.3.225
source. General parameter guidance is in the official
[training documentation](https://docs.ultralytics.com/modes/train/) and
[evaluation guide](https://docs.ultralytics.com/guides/model-evaluation-insights/).
