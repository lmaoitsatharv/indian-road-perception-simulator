# Detector checkpoint recovery

This records interruptions and recovery of `runs/detection_train_20261024_04/` on 24-25 October 2026. That parent later completed 30 epochs, followed by the separate three-epoch 1280 pilot. See [results](RESULTS.md) and the [matched pilot report](DETECTION_RESOLUTION_PILOT_RESULT.md). Process snapshots below are historical, not a running-job announcement.

## Epoch-2 interruption and optimizer reset

The training environment disappeared shortly after epoch 2 at about 12:30 pm IST on 24 October. The original Linux trainer/supervisor were absent, the Linux environment had restarted and no normal completion/exit record existed. The initiating cause was not established; Windows had not rebooted.

Both `last.pt` and `best.pt` reopened with finite model tensors and epoch-2 EMA/optimizer state. Ordinary optimizer restoration was stopped after sharply rising classification loss. Saved FP16 AdamW statistics contained 1,755,729 entries with nonzero momentum but zero variance. A discarded training-only smoke check retained epoch-2 model/EMA and tested fresh AdamW statistics and scaler. Recovery replayed epoch 3 with the original 960-input, batch-4 schedule, weighted sampler, validation split, maximum 30 total epochs and patience 8.

The checkpoint SHA-256 was `f51dc09f64449c20d7f5a17e8b9e688b2245f712613647e6fd73a15bc03d838f`. Earlier process records, checkpoints, recipe, scores and progress were preserved in run/launcher recovery directories. The full 74,404-pair audit and protected-file checks were repeated. At least ten new batches and an optimizer update were required to establish resumed learning. Fourteen checks passed. Sampling/augmentation were reinitialized, so this was not bit-for-bit replay of an uninterrupted run.

The verification snapshot contained 295 new batches with normal sampled classification loss, restored epoch/EMA and fresh optimizer state. It establishes recovery at that stage, not a new validation score. [Recorded checks](results/detection_resume_20261024.json).

## Epoch-22 CUDA interruption

On 25 October the run failed during epoch 22 after 21 completed epochs with `CUDA error: unknown error`. A verified `last_resume_fp32.pt` restored raw model, EMA, scaler and all 255 optimizer state entries. No optimizer reset was used; epoch 22 was replayed within the original 30-epoch maximum. Fourteen checks passed with finite losses and new optimizer updates. [Recorded checks](results/detection_resume_20261025.json) and [inspection](DETECTION_PROGRESS_20261025.md).

The preserved error and successful short GPU probes do not establish the root cause or guarantee long-run stability. Failed attempts are not rewritten as clean exits.

## Full-precision recovery format

The custom trainer writes `weights/last_resume_fp32.pt` atomically at saved epochs. It contains full-precision raw weights, EMA, optimizer and scaler state. The training-only probe verifies serialization/reload; probe weights do not initialize the actual candidate. Normal `best.pt` and `last.pt` remain available for inference/model selection.

Restoring learned weights with a fresh optimizer differs from exact continuation. A stripped inference checkpoint may not contain continuation state. Recipes, class order, finite tensors, hashes and completed epoch must be checked before resuming.

## Runtime support

The launcher records Linux process identity and refuses to start beside another detector trainer. Earlier attempts and failures are preserved. A bounded keep-awake helper follows the owned supervisor and releases on completion/disappearance without changing power or GPU limits. A local read-only dashboard displays saved progress; opening it does not train a model.

Framework cache verification retains the same path-and-file-size digest with concurrent file-stat requests. A temporary cache replacement in the small probe was repaired and the complete 56,729-image cache reverified before recovery; no subset cache was accepted as complete data.

Implementation: `detection_refinement.py`, `train_detection.py`, `setup/probe_detection_recovery.py`, `setup/verify_detection_resume.py` and `setup/launch_detection_training.py`. Full run directories are local artifacts outside the source repository.
