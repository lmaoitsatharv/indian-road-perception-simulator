# Why bicycle and traffic light got slightly worse in the v4 detector (29 Oct 2026)

**Question.** The overnight v4 checkpoint (`806a7163`, one epoch on the land-animal view) beats the old checkpoint (`f7bb7e4c`) overall,
but bicycle AP50 fell by about 0.9 points and traffic light AP50 by 0.5–0.7 points. Why?

**Short answer.** It is **not** a new confusion with another class, **not** dark images, and **not** a handful of bad images. The new
model makes **fewer, more confident predictions for most classes** (fewer false alarms, slightly fewer hits above each confidence
threshold). For cars and motorcycles the fewer false alarms win, so AP goes up. For bicycles, which are rare and mostly small, real
bicycles lost confidence, so AP goes down. Traffic lights behave a little differently: about the same lights are found, but there are
about 50 extra false "traffic light" boxes at confidence 0.25, all small. The drops are small, spread thinly over many images
(one box here, one box there), and sit near the edge of what random variation could produce. **Confidence in this explanation: moderate.**

Everything below is measured on the **project validation set** (17,675 images; same images, same labels for both models) unless it is
marked *inference*. No model was run for this review: it only re-reads saved evaluation files.

---

## 1. What was measured

| Item | Where it came from |
|---|---|
| AP50 per class | `analysis.json` → `per_class_ap` (old 1280 numbers from `runs/detection_pilot_recovery_20261025_01/matched_final_1280.json`, same checkpoint) |
| Recall / precision / hits (TP) / false alarms (FP) at confidence 0.10, 0.25, 0.50 | `per_image_counts.npz` from `setup/analyse_detection_size_brightness.py` (a box counts as found if a same-class prediction overlaps it with IoU ≥ 0.5, one-to-one) |
| Size bins | box size scaled to a 640 px image side: small < 32 px, medium 32–96, large ≥ 96 |
| Brightness bins | average brightness of each image, split into darker / middle / brighter thirds (same cut points for both models) |
| "What did a missed box become?" | the saved confusion table at confidence 0.25 |
| Noise check | paired bootstrap over images (2,000 resamples) of the recall / precision difference |

Matched pairs used: 1536 px / NMS 0.6 (main), 1280 px / NMS 0.7 (repeat), and the land-only validation copy at 1536 / 0.6
(results there are almost identical to the full set, so they are only in the JSON).

Helper script: `setup/analyse_v4_class_drop.py` (CPU only). Output: `runs/detection_v4_class_drop_20261029/class_drop_analysis.json`
and `train_label_diff_v3_v4.json`.

---

## 2. Headline numbers (validation)

**AP50, old → new (points)**

| Class | 1536 / NMS 0.6 | 1280 / NMS 0.7 |
|---|---|---|
| bicycle (764 boxes) | 53.67 → 52.73 (**−0.94**) | 51.38 → 50.50 (**−0.88**) |
| traffic light (2,657 boxes) | 57.80 → 57.09 (**−0.71**) | 53.85 → 53.36 (**−0.49**) |
| car (control) | 83.59 → 83.81 (+0.22) | 81.51 → 81.74 (+0.24) |
| motorcycle (control) | 82.86 → 83.12 (+0.26) | 79.92 → 80.17 (+0.25) |
| animal | +8.61 | +7.64 |

### Bicycle at 1536 / NMS 0.6 (764 boxes in 652 images)

| Confidence | Recall old → new | Precision old → new | Hits (TP) | False alarms (FP) | All bicycle predictions |
|---|---|---|---|---|---|
| 0.10 | 57.3 → 55.6 | 42.0 → 49.0 | 438 → 425 | 604 → 443 | 1,042 → 868 |
| 0.25 | 52.0 → 50.9 | 64.1 → 69.2 | 397 → 389 | 222 → 173 | 619 → 562 |
| 0.50 | 45.6 → 43.6 | 82.1 → 82.4 | 348 → 333 | 76 → 71 | 424 → 404 |

Reading it: the new model draws 17% fewer bicycle boxes. Most of the removed boxes were false alarms (good), but about 10–15 real
bicycles also dropped below each threshold. At confidence 0.50 recall fell 2 points with **no** precision gain: real bicycles now get
lower scores. That loss at the high-confidence end is what lowers AP.

At 1280 / NMS 0.7 the pattern is the same: at 0.25, recall 50.4 → 48.0 (385 → 367 hits), precision 65.8 → 69.9.

### Traffic light at 1536 / NMS 0.6 (2,657 boxes in 1,053 images)

| Confidence | Recall old → new | Precision old → new | Hits (TP) | False alarms (FP) |
|---|---|---|---|---|
| 0.10 | 62.4 → 61.5 | 51.6 → 51.2 | 1,659 → 1,634 | 1,555 → 1,555 |
| 0.25 | 52.4 → 52.4 | 69.7 → 68.0 | 1,392 → 1,392 | 606 → **655** |
| 0.50 | 38.3 → 37.6 | 87.0 → 85.7 | 1,017 → 998 | 152 → 166 |

Reading it: the same number of lights is found at 0.25, but there are 49 more false "traffic light" boxes (+8%). At 0.50 there are
19 fewer hits and 14 more false alarms. So the ranking of light boxes got slightly worse: some false alarms moved up, some real lights
moved down. At 1280 / NMS 0.7: recall flat (49.8 → 49.9 at 0.25), false alarms 658 → 690.

### Controls: car and motorcycle at 1536 / NMS 0.6, confidence 0.25

| Class | Recall old → new | Precision old → new | False alarms |
|---|---|---|---|
| car | 75.9 → 75.1 | 83.8 → 85.7 | 6,831 → 5,846 |
| motorcycle | 76.8 → 76.5 | 84.0 → 84.8 | 8,290 → 7,774 |

The improved classes show **the same shape** as bicycle: fewer predictions, fewer false alarms, slightly lower recall at a fixed
threshold. Across all 12 classes at confidence 0.10, person, car, motorcycle, bicycle, traffic sign and other_vehicle all lose
predictions (−170 to −2,070 each); truck, rider, bus, autorickshaw and animal gain them. This is a model-wide change in confidence,
not something special to bicycles.

---

## 3. Is it small objects?

Partly, but only because most of these objects are small. Bicycle: 504 of 764 boxes are small; traffic light: 2,594 of 2,657.

| Bicycle hits by size, conf 0.25 | small (504) | medium (223) | large (37) |
|---|---|---|---|
| 1536 / 0.6 | 198 → 192 | 175 → 172 | 24 → 25 |
| 1280 / 0.7 | 182 → 168 | 176 → 172 | 27 → 27 |

At 0.10 (1536) small bicycles lost 13 hits (231 → 218) while medium lost 1. So the bicycle loss leans to small boxes. For traffic
lights the extra false alarms are all small boxes (593 → 642 at 0.25). Small-box precision for cars and motorcycles went **up**, so
"the model got worse at small objects" in general is not true.

## 4. Is it dark images?

No. Bicycle hits at 0.25 (1536): darker third 177 → 178, middle third 151 → 144, brighter third 69 → 67. The darkest 10% of images
lost 1 bicycle. Traffic-light false alarms grew in all three brightness thirds (257 → 269, 257 → 286, 92 → 100). There is no dark-image
pattern at either resolution.

## 5. Is it confusion with another class?

No. What a missed bicycle overlapped instead, at confidence 0.25 (1536):

| Missed bicycle became… | old | new |
|---|---|---|
| nothing (background miss) | 239 | 244 |
| motorcycle | 89 | 90 |
| person | 25 | 24 |
| rider | 2 | 4 |
| a bicycle box already used for another bicycle | 7 | 8 |

Bicycle ↔ motorcycle confusion is unchanged (89 → 90); in the other direction, motorcycles whose best overlapping box was a "bicycle"
went **down** (42 → 33). The extra misses are plain background misses: the model was not confident enough to draw a box.

Traffic lights: misses that became "traffic sign" 14 → 17; signs taken as lights 12 → 14; background misses 1,237 → 1,237. Light ↔ sign
confusion is a few boxes, not 50.

## 6. Is it a few images, and could it be noise?

**Spread out, not concentrated.** Bicycle at 0.25 (1536): 17 images each lost exactly one found bicycle and 9 images each gained one;
net −8. No single image lost more than one. Traffic light at 0.25: 48 images lost one, 48 gained one (net 0). At 0.50: 56 images
lost, 38 gained (net −19).

**Bootstrap over images** (2,000 resamples, 95% range of new − old, in points):

| | Recall at 0.10 | Recall at 0.25 | Recall at 0.50 | Precision at 0.25 | Precision at 0.50 |
|---|---|---|---|---|---|
| bicycle 1536 | −3.2 to −0.1 | −2.3 to +0.3 | −3.4 to −0.6 | +2.8 to +7.4 | −1.5 to +2.3 |
| bicycle 1280 | −2.8 to −0.3 | −3.8 to −1.1 | −2.3 to +0.1 | +1.2 to +6.8 | +1.1 to +5.6 |
| traffic light 1536 | −1.7 to −0.2 | −0.7 to +0.7 | −1.4 to 0.0 | −2.8 to −0.6 | −2.3 to −0.3 |
| car 1536 (control) | −0.5 to −0.2 | −1.0 to −0.7 | −1.4 to −1.1 | +1.7 to +2.1 | +0.8 to +1.1 |

What this says: the bicycle recall loss and the traffic-light precision loss are usually just outside the "could be zero" range, so they
are probably real changes in the model, but small ones (8–18 bicycles; about 50 light false alarms).

What could **not** be computed: a bootstrap of AP itself, because per-box scores were not saved. The bootstrap also treats images as
independent, while many frames come from the same drive; true ranges are likely somewhat wider (*inference*). The 1536, 1280 and
land-only results agree, but they are the same images and the same two models, so they are not independent confirmations.

**Useful reference for normal wobble** (validation, 1280 / NMS 0.7, AP50): in the earlier 3-epoch fine-tune (epoch-30 parent →
`f7bb7e4c`), bus moved +1.4, truck +1.25, car +0.9, traffic light −0.38 and bicycle −0.11. Per-class moves of about 1 point between
gentle fine-tune steps are normal. Traffic light is now down two steps in a row: 54.23 → 53.85 → 53.36.

---

## 7. Did the training sampler or the training labels change?

| | old run (25 Oct) | new run (29 Oct) |
|---|---|---|
| Sampler settings | rare-class boosts bicycle ×2, animal ×4, traffic light ×2 | **identical** |
| Training images | 56,729 (v3) | 56,896 (v4) |
| Bicycle: images / expected draws per epoch | 2,758 / 4,981 | 2,771 / 4,997 (share +0.04%) |
| Traffic light: images / expected draws per epoch | 3,141 / 5,672 | 3,143 / 5,645 (share −0.8%) |
| Animal: images / expected draws per epoch | 120 / 433 | 254 / 912 (share +110%) |
| Bicycle boxes in training labels | 3,236 | 3,252 (+16) |
| Traffic light boxes in training labels | 7,208 | 7,211 (+3) |
| Learning rate (per epoch, from `results.csv`) | 3 epochs: 3e-5, 2.6e-5, 1.9e-5 | 1 epoch, **all of it at 3e-5** |

A file-by-file comparison of the v3 and v4 training labels shows that the 34 inherited images whose labels changed lost only animal
boxes. The +16 bicycles come from new images: 14 in the 105 recovered IDD images and 2 in the 82 DATS images. The +3 traffic lights
are 5 added in DATS images minus 2 in the 20 removed images. So bicycle and traffic-light training data and sampling are essentially
unchanged, and no bicycle or traffic-light label was edited or removed. The two real differences are (a) about 480 extra animal draws per epoch, which take roughly 0.8% of draws away from everything
else, and (b) the learning rate jumped back up from 1.9e-5 (the old run's last epoch) to 3e-5 for the whole new epoch, on top of an
already fine-tuned model.

---

## 8. Most likely explanation

1. **One more epoch at a re-raised learning rate made the whole model more conservative** (measured: fewer predictions and fewer false
   alarms for most classes, at both resolutions). *Inference:* the LR restart is the most likely driver, since sampling and labels for
   these classes did not change.
2. **Bicycle** is rare, mostly small, and often looks like a motorcycle or background. Under that conservative shift, a dozen or so real
   bicycles fell below each confidence threshold (measured), mostly small ones (measured). Its false-alarm drop came mostly at low
   confidence, where it helps AP less (*inference*), so AP50 fell.
3. **Traffic light** got about 50 extra small false alarms at confidence ≥ 0.25 with no recall gain (measured), so its precision and AP
   fell. This continues a slow downward drift already seen in the 25 Oct fine-tune (measured). Why extra false alarms appear is not
   known from saved data (no FP images were saved by class); *inference:* small lamp-like background objects.
4. **Not** caused by: dark images, bicycle ↔ motorcycle or light ↔ sign confusion, a few bad images, or changed labels/sampler (all
   measured).

How sure: moderate. The direction is consistent at both settings and the bootstrap ranges mostly exclude zero, but the size (about 1
AP point on 764 boxes) is within normal fine-tuning wobble, and AP itself could not be bootstrapped.

---

## 9. Options

1. **Accept v4 as it is (recommended for now).** mAP50 is up (64.81 → 65.41 at 1536 / 0.6, validation) and animal is up by about 8.6 AP50;
   the bicycle / light losses are about 8–18 bicycles and about 50 light false alarms on 17,675 images. Record the drop and keep
   watching those two classes. Because the new model's bicycle boxes are more precise at every threshold, a slightly lower bicycle
   confidence threshold in the driving stack could win back the lost recall (*inference*; it needs one more evaluation to confirm).
2. **Short probe with a stronger sampler for the two classes.** From `806a7163`, about a quarter of an epoch, bicycle ×3 and traffic
   light ×3 (animal ×4 kept), constant low learning rate (1.5e-5, no restart). Pass if bicycle and traffic light return to old AP50 at
   both matched settings while animal keeps at least +6 AP50 and overall mAP50 does not fall.
3. **Lower-LR redo to test the learning-rate explanation.** From `f7bb7e4c`, one v4 epoch at lr 1.5e-5 (where the old schedule ended)
   instead of 3e-5. If the model-wide "fewer predictions" shift and the bicycle / light losses mostly disappear while animal still
   improves, the LR restart was the cause. (A cheaper, inference-only check first: evaluate the saved `epoch1.pt` / `epoch2.pt` of the
   25 Oct run with the same script to measure how much bicycle and traffic light AP normally move between epochs.)

All options keep the official test set untouched and need a project decision before any GPU run.
