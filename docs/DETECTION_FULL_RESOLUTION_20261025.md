# Full detector resolution check - 25 October 2026

The full paired check confirms that higher input resolution improves the average score and people/vehicle detection, with an animal regression. These results cover all **17,675 unchanged validation images / 319,438 labelled objects** using the **same frozen epoch-23 weights**, FP32 inference and identical detection settings. Resolution is the only changed setting.

| Resolution | mAP50 | mAP50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 960 | 57.807% | 38.511% | 71.962% | 52.400% |
| 1280 | 61.477% | 40.956% | 73.397% | 55.701% |

The paired improvement is **+3.671 percentage points mAP50 / +2.445 points mAP50-95**. Person AP50 improves **56.988% to 62.730%**. Eleven of twelve class AP50 values improve. Animal AP50 falls **14.630% to 9.299%**, so no all-class inference default is changed. This is a development validation result, not a held-out independent test or proof of safe driving.

| Class | Epoch 23 at 960 | Epoch 23 at 1280 | AP50 change, points |
| --- | ---: | ---: | ---: |
| person | 56.99% | 62.73% | +5.74 |
| rider | 65.47% | 70.49% | +5.03 |
| car | 78.75% | 80.53% | +1.78 |
| motorcycle | 75.58% | 79.80% | +4.22 |
| bicycle | 47.30% | 51.22% | +3.92 |
| autorickshaw | 80.03% | 83.72% | +3.69 |
| bus | 77.62% | 78.72% | +1.09 |
| truck | 73.96% | 75.34% | +1.37 |
| animal | 14.63% | 9.30% | -5.33 |
| traffic light | 44.51% | 53.54% | +9.03 |
| traffic sign | 52.70% | 59.03% | +6.33 |
| other vehicle | 26.14% | 33.31% | +7.17 |

The CPU comparison measured 186.7 ms versus 295.7 ms of inference per image while the GPU trained. This is approximately 1.58 times the CPU inference cost; GPU latency needs its own measurement. It is a quicker route to accuracy improvement than another long training run, not faster inference.

## Completed parent and subsequent pilot

The separate 30-epoch parent run completed and exited cleanly. Its best saved checkpoint scores **58.054% mAP50 / 38.638% mAP50-95** at 960 input. The epoch-3 history was **51.340% / 33.442%**; epoch 3 weights were not retained. The original 960 reference scored **55.870% / 37.165%**. The final checkpoint therefore gains 2.184 / 1.472 points against the original reference. The per-epoch CSV differs slightly from final saved-checkpoint validation; use the final report for the saved model's score.

The subsequent three-epoch 1280 trial measured a full GPU FP32 baseline before learning and repeated identical evaluation afterward. It used learning rate 0.00003, existing rare-class sampling, no mosaic and saved each epoch plus FP32 recovery. The parent checkpoint was preserved. Final matched scores were 61.688% to 62.245% mAP50 and 41.125% to 41.525% mAP50-95; the gain is modest and three classes regress slightly. See [the final pilot result](DETECTION_RESOLUTION_PILOT_RESULT.md).

The initial AMP/batch-4 attempt failed with an unknown CUDA error after 238 sampled batches and nine optimizer updates, before any completed epoch. Its evidence remains under `detection_train_20261025_01`. A changed diagnostic attempt, `detection_train_20261025_02`, completed using FP32, batch 2, zero workers and `CUDA_LAUNCH_BLOCKING=1`. These changes reduced concurrency factors and localized errors; they did not establish AMP as the cause. No checkpoint was automatically deployed.

Evidence:

- Full paired results and integrity checks: `runs/detection_full_resolution_20261025_01/comparison.json` (local artifact, supplied separately).
- Completed parent model and checkpoint checks: `runs/detection_train_20261024_04/refinement_result.json` (local artifact, supplied separately).
- New trial preparation and matched baseline: `runs/detection_pilot_prepare_20261025_01/` (local artifact, supplied separately).
