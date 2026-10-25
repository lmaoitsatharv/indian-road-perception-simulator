# Separate simulation model training — 22 October 2026

The new simulation-only model completed training on the prepared
`dataset/simulation_india_v5/` dataset. This starts a fresh decoder with the
cached ImageNet MobileNetV2 encoder, matching the established U-Net architecture.
No real-data checkpoint initializes this run. The real model, inference files,
deployment lock, original splits and retained checkpoints remain unchanged.

Training stopped automatically after **11 epochs** (about 25.8 minutes).
The best exact validation road IoU came from **epoch 10**: **99.8638%**, with
Dice **99.9318%**. This checkpoint was then fixed and scored on all 500 reserved simulation test images: road IoU **99.9288%**,
Dice **99.9644%**, valid-pixel coverage **98.2174%**. An independent recount of
all 500 saved predictions passed. Read [SIMULATION_TEST.md](SIMULATION_TEST.md).

## Current run and evidence

- Run: `runs/simulation_frozen_20261022_01/`.
- Live phase, epoch, batch and metrics: `progress.json` inside that run.
- Overall running/completed/failed state: `status.json`; final details:
  `run_config.json`.
- Epoch metrics: `history.csv`; validation-selected model: `best.keras`.
  `last.keras` and `history.json` are written after the initial phase ends.
- Persistent process ownership and console log:
  `runs/simulation_frozen_20261022_01_launcher/{owner.json,training.log}`.
- Passed verification:
  `runs/simulation_training_verification_20261022_02/verification.json`.
- Portable status snapshot: [results/simulation_training.json](results/simulation_training.json).
  The live run files take precedence over a dated snapshot.

The Linux training process owns its own session and continues after the launch
command returns. A hidden Windows helper temporarily prevents automatic system
sleep while the saved state is starting/running, with a four-hour limit. It
releases its wake request on completion/failure; the display may still turn off.
Its evidence is `keep_awake.log` inside the run directory.

## Settings and isolation

| Setting | Initial phase |
|---|---|
| Dataset | 4,000 train / 500 validation / 500 reserved test, whole episodes |
| Input | 512 × 512 RGB, MobileNetV2 normalization once to [-1, 1] |
| Initialization | Fresh decoder, cached ImageNet encoder; encoder frozen |
| Precision | Float32, matching the real model's initial frozen phase |
| Optimizer | Adam, learning rate 0.001, clipnorm 1.0 |
| Batch / seed | 4 / 42 |
| Augmentation | Existing paired horizontal flip; image, road and validity aligned |
| Loss | Per-image valid-pixel BCE + valid-pixel soft Dice, equal weights |
| Scores | Global valid-pixel TP/FP/FN road IoU and Dice, threshold 0.5 |
| Duration | Up to 20 epochs |
| Best checkpoint | Maximum validation road IoU |
| Early stopping | Validation road IoU, patience 5, minimum change 0.001 |
| Learning-rate reduction | Validation loss, factor 0.5, patience 2, minimum 1e-6 |

`simulation_data.py` retains the audited `(image, road, valid)` interface. The
separate `simulation_model.py` adapter packs road/valid into training targets.
The network still has an ordinary one-channel probability output. Validity is
never an input feature. Invalid target zeros are placeholders: both terms of
the objective and both scores explicitly exclude them, including the Dice
denominator. Standard unmasked real-model training would treat these placeholders
incorrectly and is not used here.

During training only train and validation image contents were opened.
Manifest/file-name checks confirmed the reserved test split existed. Test image
contents were subsequently opened only by the separate, fixed-checkpoint test
evaluation after training completed.
The trainer checks the fixed simulation manifest and all protected real-file
hashes before starting and after it ends. Each run saves its source snapshot,
recipe, verification report, runtime details and initialization hash. Existing
run directories are never overwritten.

## Completed checks before starting

The GPU verification passed on TensorFlow 2.20 / Keras 3.15.1 with the RTX 4060
Laptop GPU and the actual batch size of four:

1. Compare masked BCE/Dice with hand-calculated fixture values. Ignored
   predictions and labels cannot change loss; ignored pixels have exactly zero
   gradient. With all pixels valid, results match the original real objective.
2. Recount TP/FP/FN across separate batches, check IoU/Dice and the empty-road
   case, and verify ignored pixels cannot change either score.
3. Read actual train/validation images through the existing preprocessing;
   retain a smaller final batch and keep mask/validity aligned during flips.
4. Perform one discarded GPU learning step: finite loss/scores, decoder weights
   change, all frozen encoder/BatchNorm weights remain byte-identical.
5. Save and reload a compiled checkpoint including optimizer state, reproduce
   probabilities and validation scores, and load for ordinary inference too.
6. Verify protected real-model files remain unchanged. No test images were read.

The first verification attempt completed its smoke work but failed while writing
a NumPy integer to JSON. That attempt is recorded as failed. The scalar reporting
was corrected and the full independent second attempt passed. Neither discarded
smoke checkpoint is used as training initialization.

## Completed selection and following work

Epoch 10 was selected from validation only; checkpoint SHA-256 and threshold 0.5 were fixed
before scoring the test split. Optional fine-tuning was not performed. The saved
fine recipe remains a future experiment, requiring validation-based selection
and a new untouched confirmation set for any new final performance claim.

The following fresh Indian-style confirmation check was completed on 23 October
with changed geometry/assets and explicit simulator-only routing. On all 480
frames, the new model achieved road IoU 99.7478% (locked baseline 79.9195%).
Read [SIMULATION_CONFIRMATION.md](SIMULATION_CONFIRMATION.md). Segmentation does
not control vehicles. Bounded live observation was then implemented and checked;
read [LIVE_SIMULATION_OBSERVATION.md](LIVE_SIMULATION_OBSERVATION.md) for timing,
fault recovery and the measured display expiry limitation. No RL policy has been trained.

The dataset's preparation reports and original saved recipe are historical
artifacts from 21 October: their `training_started: false` means training had not
started when preparation completed. Current training state lives in the new run.

## Reproduction

From the project folder in the verified WSL environment, use fresh output/run
names to preserve previous evidence:

```bash
~/.venvs/idd-tf220/bin/python setup/verify_simulation_training.py --output runs/simulation_verification_NEW
~/.venvs/idd-tf220/bin/python setup/launch_simulation_training.py --run-name simulation_frozen_NEW --verification runs/simulation_verification_NEW/verification.json
```

The launcher sets deterministic GPU options. The trainer requires the successful
verification report to match its current source hashes and fixed dataset manifest.
The original `train_idd.py`, model files and real training routines are unchanged.
