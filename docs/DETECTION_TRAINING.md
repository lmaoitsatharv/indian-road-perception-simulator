# IDD object-detector training - 24 October 2026

This is the first completed 640-input training stage. Later clean-view refinement and the matched 1280 pilot are summarized in [current results](RESULTS.md).

**All 20 epochs and full final validation completed; epoch 20 was selected.**
Final validation is **47.26% mAP50 / 31.24% mAP50–95**, on 17,675 images.
The worker then ended with SIGKILL / return code −9 after printing results; the
cause is unconfirmed. Checkpoints were independently reopened and verified,
including GPU inference for the best checkpoint. Read
[the complete report](DETECTION_TRAINING_REPORT.md) and
[machine-readable results](results/detection_training.json).
The raw trainer status/batch files are stale because cleanup was interrupted;
use the report together with the preserved launcher exit record.

## Historical startup snapshot

Training followed the completed preprocessing stage. The run is `runs/detection_train_20261024_02/`; its launcher is
`runs/detection_train_20261024_02_launcher/`. The exhaustive file-readiness check
and both framework caches passed. **At the startup check, epoch 1 was running.**
The first ten real batches completed six successful optimizer steps with finite
sampled losses. Measured startup peak GPU allocation was 1.16 GiB, with 1.29 GiB
reserved. These numbers describe startup, not the final run's peak memory.

The first launcher attempt (`detection_train_20261024_01_launcher`) was stopped
during its pre-training file audit, before a model/run directory existed. Repeated
ancestor resolution through WSL/DrvFS took 1.611 seconds for 32 sampled pairs.
The revised guard resolves the four fixed data folders once, checks every image's
regular-file identity/size/timestamp and every label's hash in six bounded workers,
rejects symlinks and incorrect paths and rechecks the parent folders afterward.
Seven Windows and nine WSL verification groups passed. The earlier attempt and
its restart reason remain preserved; no optimizer step happened in that attempt.

The framework loaded all 67,159 training images (151 annotated negatives) and
17,675 validation images (53 annotated negatives), with zero missing or corrupt
pairs. Its cached per-class box counts exactly match the audited labels for both
splits. The actual augmented training preview `train_batch0.jpg` was visually
inspected; boxes follow the resized/padded/mosaic image geometry. All 17 protected
fingerprints and the original YOLO11s initialization file remain unchanged.
Read [the verified startup snapshot](results/detection_training_start.json).
No validation accuracy or completed-epoch result was claimed at startup.

## Data and settings

The input is `dataset/idd117k_detection_v2/data.yaml`: 67,159 training entries,
17,675 validation entries and 12 classes. All 24,644 official unlabelled test
images remain outside training and local scoring. Retained within-training
repeated frames and annotation limitations are documented in
[the preprocessing report](IDD_DETECTION_IMPORT.md).

The saved recipe is unchanged: COCO-pretrained YOLO11s, 640-pixel input, batch 4,
AdamW, initial learning rate 0.0001, final-rate fraction 0.1, seed 42, four loader
workers, AMP, deterministic mode, no image cache and mosaic closing for the last
five epochs. The configured maximum was 20 epochs, with early stopping after five
validation epochs without improvement. Checkpoint selection uses the pinned
Ultralytics validation fitness.

The isolated WSL `idd-detect` runtime used
PyTorch 2.9.1+cu128 and Ultralytics 8.3.225 on the RTX 4060 Laptop GPU. Live
preflight confirmed CUDA and the expected runtime versions. Model/source
fingerprints are saved before training. No road model is loaded for training.

## Preserved run files

- `runs/detection_train_20261024_02_launcher/preflight.json`: startup checks.
- `runs/detection_train_20261024_02_launcher/owner.json`: persistent worker identity.
- `runs/detection_train_20261024_02_launcher/training.log`: full startup/training log.
- `runs/detection_train_20261024_02_launcher/exit.json`: actual failed worker exit.
- `runs/detection_train_20261024_02/training_report.json`: reconciled result report.
- `runs/detection_train_20261024_02/status.json`: last trainer snapshot, stale status.
- `runs/detection_train_20261024_02/batch_progress.json`: last periodic batch snapshot,
  sampled finite losses, actual optimizer-step count and GPU memory.
- `runs/detection_train_20261024_02/first_training_batches.json`: first ten or more
  real batches with successful optimizer steps and measured backward-pass memory.
- `runs/detection_train_20261024_02/progress.json`: final saved-best validation metrics.
- `runs/detection_train_20261024_02/results.csv`: all 20 epoch-validation rows.
- `runs/detection_train_20261024_02/weights/last.pt` and `best.pt`: verified weights.
- `runs/detection_train_20261024_02/report_verification_02/checkpoint_verification.json`:
  post-run checkpoint integrity, class order and GPU/native-image inference checks.
- `runs/detection_train_20261024_02/completion_protection.json`: independent
  post-run protected-file and original initialization/recipe hashes.

Batch monitoring observes the optimizer through a post-step hook; it does not
change gradients, losses or learning settings. A successful startup proves real
backward/optimizer execution and finite sampled losses, not detector accuracy.
The training process has ended. The temporary Windows system-wake request was
released at 09:44:34 IST; the power plan was unchanged. The original startup
helper followed the worker's identity and left the display free to sleep.

## Model scope and further integration

Existing real-road and simulation-road checkpoints, real inference and the rule
driver remain separate. The new detector is not connected to vehicle control.
Tracking, distance estimation and RL remain future work. The report and
checkpoint verification are complete; subsequent error analysis and refinement
are documented in [results](RESULTS.md). The official test images have no released labels for local
accuracy measurement.
