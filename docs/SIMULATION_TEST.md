# Simulation test score — 22 October 2026

The separate simulation model's validation-selected **epoch-10** checkpoint was
scored on all **500 reserved simulation test images**, with threshold **0.5**.
The checkpoint and protocol were fixed before opening any test image contents.
No training, fine-tuning, threshold search or checkpoint comparison used test
results. The completed real-data test was not reopened and real inference is
unchanged.

| Test metric | Result |
|---|---:|
| Road IoU, global | **99.9288%** |
| Road Dice, global | **99.9644%** |
| Road precision | 99.9448% |
| Road recall | 99.9840% |
| Valid-pixel accuracy | 99.9756% |
| Mean per-image masked BCE + soft Dice loss | 0.001288713 |
| Valid-pixel coverage | **98.2174%** |

Scores use 128,735,569 valid pixels out of 131,072,000. The 2,336,431 uncertain
pixels (1.7826%) are excluded from both loss terms and all confusion counts.
Global overlap uses pooled counts, rather than averaging image percentages:
44,038,649 true positives, 24,324 false positives, 7,050 false negatives and
84,665,546 true negatives.

## Results by layout

| Generated layout | Test images | Road IoU | Road Dice |
|---|---:|---:|---:|
| Crossroads | 125 | 99.8036% | 99.9017% |
| North T-junction | 125 | 99.9640% | 99.9820% |
| South T-junction | 125 | 99.9658% | 99.9829% |
| Straight street | 125 | 99.9865% | 99.9932% |

There is one held-out episode per layout. Weather and traffic-case breakdowns
are saved too, but share these same four episodes; they do not independently
isolate effects of weather or traffic type. Episodes/seeds stay disjoint from
training/validation; geometry families and the generic asset kit remain shared.
This establishes performance on the prepared synthetic test distribution, not
unseen Indian cities or real roads. Pixels within an episode are correlated.

## Verification and visual inspection

TensorFlow's masked global IoU/Dice matched an independent NumPy count during the
single inference pass. A separate script without TensorFlow then reopened all
500 saved binary prediction PNGs, checked their hashes and independently
recounted every image and all episode/layout/weather/case groups against the
test masks and validity. Every count and global score matched. Checkpoint,
dataset manifest, original real-model files and in-memory model weights remained
unchanged.

The evidence sheet shows the worst-scoring saved image in each test episode,
including image 1179 at the crossroads (IoU approximately 96.015%). Inspection
shows residual errors near road/junction edges. Magenta marks ignored pixels;
red/blue mark valid-pixel false positives/negatives. The summary score should be
read together with these local errors and coverage.

Worst image in each test episode: `runs/simulation_test_20261022_01/test_examples.png` (local artifact, supplied separately).

## Saved evidence

- Test run: `runs/simulation_test_20261022_01/`.
- Fixed selection/protocol: `protocol.json` in that directory.
- Scores/counts/group breakdowns: `test_results.json`.
- Independent recount: `independent_verification.json`.
- All 500 prediction masks: `predictions/Prediction_{id}.png`, uint8 0/1.
- Per-image counts, loss and prediction hashes: `per_image.csv`.
- Visual evidence: `test_examples.png`.
- Portable result: [results/simulation_test.json](results/simulation_test.json).
- Selected model: `runs/simulation_frozen_20261022_01/best.keras`.
- Checkpoint SHA-256:
  `8d8c9fd7a3880bb69643c4973db8f14c46242c178fae6927d06b5f9d6aaaf8af`.
- Split manifest SHA-256:
  `c1b9ebbc1ba4768de91d413cb03170bdd5ab8f0a5de1dd3bc6ef344e98675e92`.

The initial phase stopped automatically after 11 epochs; the best exact validation
IoU came from epoch 10 (99.8638%, Dice 99.9318%). Early stopping used a minimum
improvement of 0.001 with patience 5; checkpoint saving tracked the exact maximum.
No optional fine-tuning was performed before this fixed-checkpoint final evaluation.

The 500-image test split is now inspected. Future model choices should use
training/validation and a new untouched confirmation set for new final claims.
Fresh simulation scenarios with changed assets/geometry were subsequently
checked on 23 October: [SIMULATION_CONFIRMATION.md](SIMULATION_CONFIRMATION.md).
Neither evaluation changes deployment or connects segmentation to vehicle control.
