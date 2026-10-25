# Detector interruption inspection - 25 October 2026

This is the historical inspection at approximately 04:01 IST of the 960-input parent, not its current completion state. The parent subsequently resumed, completed all 30 epochs and initialized a separate three-epoch 1280 pilot. See [current results](RESULTS.md), [recovery](DETECTION_RECOVERY.md) and [matched pilot result](DETECTION_RESOLUTION_PILOT_RESULT.md).

## Saved state at inspection

The run had stopped after a CUDA error during epoch 22 at about 01:47 IST. Twenty-one epochs were complete; best epoch was 21. The last snapshot showed 12,361 / 14,183 batches in the unfinished epoch, which had no completed validation score and needed replaying.

| Validation metric | Starting reference, 960 | Epoch 21 | Gain, percentage points |
| --- | ---: | ---: | ---: |
| mAP50 | 55.87% | 57.68% | +1.81 |
| mAP50-95 | 37.17% | 38.39% | +1.23 |
| Precision | 70.96% | 72.37% | +1.41 |
| Recall | 50.29% | 52.16% | +1.88 |

The 17,675-image validation membership was unchanged. The reference used standalone FP32 validation; framework validation used AMP. This was development evidence before final saved-checkpoint evaluation, not an official unlabelled-test score. Per-class gains were not established by this inspection.

Epoch 3 recorded 51.34% mAP50 / 33.44% mAP50-95. Both reference metrics were first exceeded at epoch 11. Gains continued through epoch 21, with smaller recent changes. Development goals of about 70% mAP50 and 45-50% mAP50-95 were not attained at this point.

## Integrity and fault evidence

`best.pt`, `last.pt` and `last_resume_fp32.pt` reopened with finite model/optimizer tensors, correct twelve classes and completed-epoch metadata matching saved results. The FP32 file retained raw model and AdamW moments for all 255 optimizer entries, so the earlier FP16-moment reset was unnecessary for the later continuation.

The trainer was absent and its wake request released. The exception was `torch.AcceleratorError: CUDA error: unknown error`, not early stopping. A fresh CUDA-context image forward pass and scalar gradient probe passed. They did not establish the cause or long-run stability. GPU temperature at inspection was 58 degrees C without a thermal-slowdown flag; this was not a measurement at failure. A nearby display event was not a confirmed cause.

All 17 protected model/inference/driver fingerprints matched. The inspection itself did not restart, deploy or change control. Subsequent continuation restored the verified FP32 checkpoint, replayed epoch 22 and retained the 30-epoch maximum. [Resume verification](results/detection_resume_20261025.json).

Portable evidence: [inspection result](results/detection_progress_20261025.json). Full verification, score CSV, traceback and exit records remain under `runs/detection_progress_check_20261025_01/` and original run/launcher directories, outside this source checkout.
