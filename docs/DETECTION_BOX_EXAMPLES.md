# Actual validation boxes and predictions

A local gallery under `runs/detection_box_examples_20261025_01/` contains four selected source-label/prediction comparisons. The provider photographs and local gallery are not distributed in this public checkout. These examples use an earlier epoch-1 checkpoint; [current aggregate results](DETECTION_RESOLUTION_PILOT_RESULT.md) use the final selected checkpoint.

The frozen checkpoint is the full-precision model after completed epoch 1 of detection_train_20261025_02. Inference uses CPU, 1280 input, confidence floor 0.001 and NMS IoU 0.7. The default displayed threshold is 0.25, with class-matched IoU 0.5 for the visual comparison. Green boxes are supplied labels, blue predictions are matched, and red predictions are unmatched to supplied labels. An unmatched prediction is not automatically a nonexistent real object.

## Reviewed examples

1. A large bus is recognized, with an extra overlapping bus detection.
2. People are detected, but a broad overlapping supplied label does not obtain a box match. Several shop mannequins also get person predictions.
3. Crowded traffic mixes larger objects and many small labels, with numerous unmatched labels.
4. A large foreground cow is labelled animal, but predicted as other_vehicle at about 0.50 confidence.

Detailed figures and metadata: `runs/detection_box_examples_20261025_01/REPORT.md` (local artifact, supplied separately).. Selected photos and label files were checked against their source hashes; the frozen checkpoint and all protected fingerprints remained unchanged. These four illustrations were deliberately selected from 80 candidates and do not establish an aggregate score.

## Nearby objects and collision risk

Nearby objects that may intersect the car's path deserve priority. Farther objects can still be relevant within the stopping/look-ahead window; at 50 km/h the car travels about 14 metres per second. Box size alone is not distance: a farther bus can look larger than a closer bicycle. Current labels contain 2D boxes, not measured range. A true nearby-risk score requires distance/closing-motion and path information. A separate large/foreground-object analysis can be a proxy, but must not be labelled a measured near-distance evaluation.

[Stopping-distance principle](https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158#rule126) · [Forward collision warning and braking](https://www.nhtsa.gov/vehicle-safety/driver-assistance-technologies). These figures assess perception; they do not validate collision-avoidance control.
