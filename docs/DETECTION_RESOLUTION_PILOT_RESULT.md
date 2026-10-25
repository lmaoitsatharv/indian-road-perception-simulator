# Three-epoch resolution adaptation result - 25 October 2026

The trial completed all three epochs and all matched validation checks. Epoch 3 is selected. Both best/last checkpoint hashes were rechecked against the completion evidence. The parent best remains preserved.

Same 17,675 validation images / 319,438 objects, GPU FP32 inference and 1280 input, using the exact same standalone evaluation settings before and after learning.

| Metric | Starting model | Final model | Change, points |
| --- | ---: | ---: | ---: |
| mAP50 | 61.688% | 62.245% | +0.557 |
| mAP50-95 | 41.125% | 41.525% | +0.400 |
| Precision | 74.767% | 74.229% | -0.539 |
| Recall | 55.362% | 55.784% | +0.423 |

| Class | Starting AP50 | Final AP50 | Change, points |
| --- | ---: | ---: | ---: |
| person | 62.876% | 63.246% | +0.370 |
| rider | 70.626% | 70.768% | +0.142 |
| car | 80.585% | 81.506% | +0.921 |
| motorcycle | 79.845% | 79.923% | +0.078 |
| bicycle | 51.493% | 51.381% | -0.112 |
| autorickshaw | 83.860% | 83.597% | -0.264 |
| bus | 79.270% | 80.687% | +1.417 |
| truck | 75.554% | 76.801% | +1.247 |
| animal | 9.058% | 11.487% | +2.430 |
| traffic light | 54.228% | 53.853% | -0.375 |
| traffic sign | 59.283% | 59.700% | +0.416 |
| other vehicle | 33.578% | 33.985% | +0.408 |

The gain is modest. Bicycle, autorickshaw and traffic-light AP50 regress slightly; the candidate is not an all-class improvement. No checkpoint was automatically deployed. These are development validation results, not an independent test or a collision-avoidance evaluation.

The +25% learning-rate screen did not justify changing the rate. Further substantial improvements need a new evidence-backed hypothesis, particularly more varied examples of weak classes and reviewed training-only misses, rather than repeating identical long runs.

[Matched raw evidence](results/detection_resolution_pilot.json) · Best checkpoint: `runs/detection_train_20261025_02/weights/best.pt` (local artifact, supplied separately). · FP32 recovery checkpoint: `runs/detection_train_20261025_02/weights/last_resume_fp32.pt` (local artifact, supplied separately)..
