# Filtered animal training view and land-only re-score — plan and results (29 October 2026)

**Status (morning of 29 Oct):** Parts A and B were carried out as project decisions, and one training epoch was run; see the result
sections near the end. The text below was written as the original plan: **at the time, nothing here had been run.** No dataset is changed, no label is deleted, no training starts. Every step that
creates files, uses the GPU or changes what the model sees needs an explicit project decision.

Rule (project decision, 29 October): **`animal` = land animals and land birds** (dogs, cattle, buffalo, goats, horses, camels,
chickens on the ground). **Not animal = birds in flight and birds perched on wires or poles.**
Background: [animal label review](ANIMAL_DATA_REVIEW_20261028.md), [animal data plan](ANIMAL_DATA_PLAN_20261029.md).

DATS_2022 attribution: Paranjape, *DATS_2022*, Mendeley Data, Version 2, DOI [10.17632/nfc34n8svj.2](https://doi.org/10.17632/nfc34n8svj.2), licensed CC BY 4.0. Only counts, notes and scripts are published here; the images and annotations are not redistributed, and any local use changes them (EXIF rotation baked in, resized, relabelled to the project classes).

## Why this has to come first

- Training: 860 animal boxes in **120 images**; **692 tiny boxes (under 20 px) in 55 images** look like wire birds;
  only **168 boxes in 83 images** are 20 px or larger.
- Validation: 963 animal boxes; about 6–10% sit high in the frame and some are probably wire or sky birds.
- Under the new rule the current labels teach the wrong thing, and the current animal score measures a mix of
  two different tasks. Any new training or "improvement" would be compared against a muddled baseline.

## Part A — Build a filtered training view (new folder, `idd117k_detection_v3` untouched)

**A1. Decide every animal box in training (all 860 boxes, 120 images).**
- 120 images is small enough to review *all* of them: about 6 contact sheets of 20 images, drawing every animal
  box, using the same sheet tool as before (`setup/make_animal_review_sheets.py`, adapted).
- Proposed starting rule, to be checked rather than trusted: *drop* a box if it is under 20 px **or** its centre
  is in the top 25% of the image. Then review the sheets to measure how often the rule is wrong in both directions
  (a real small animal dropped; a large bird kept, e.g. a bird at 25 px on a wire).
- Output: one decision per box: `keep`, `drop`, or `unsure`. Record reviewer (model or person) and a note.
- Two model reviewers plus a human spot check on about 30 images; `unsure` boxes are kept out of training.

**A2. What "drop" means.** Delete the box from that image's label file, which makes the bird *background*.
That is the intended meaning (the model should stop detecting wire birds as animals). Do not move or delete images.
One exception to consider: an image whose only animals were dropped could stay as a normal negative image.

**A3. Build `dataset/idd117k_detection_v4_landanimals/` with the same layout as v3.**
- Images: hard links to the existing files, as v3 did (no extra disk).
- Labels: copy v3, rewriting only the labels listed in A1. **Every other label file stays byte-identical** (check by hash).
- Validation: **unchanged and identical to v3** (images and labels). Changing validation is Part B and is separate.
- `pairs.jsonl`, `preparation.json` and `provenance/` copied, with a new file listing every changed label and the reason.
  The sampler reads `pairs.jsonl` and the class boost for `animal` (currently 4×) is applied per image; with fewer animal
  images the boost is computed on the filtered labels, so check the resulting weights.
- Framework caches (`labels/*.cache`): regenerate for the new view, or reuse via `setup/cache_clean_detection_view.py`
  after the label audit; do not copy v3's cache blindly.

**A4. Independent checks before the view is allowed near training.**
- File counts and hashes: all unchanged files match v3; changed files match the decision list exactly.
- Per-class box counts: only `animal` differs, and by exactly the dropped boxes.
- Re-run the box-position/size statistics from this note: median animal size should rise well above 9 px and the
  top-quarter share should fall sharply.
- Look at a random 40-image sheet from the new view.

## Part B — Re-score validation animal recall on land animals only

**B1. Classify the 963 validation animal boxes.** Review the suspicious subset by eye (about 330 boxes:
the 267 under 20 px plus those in the top 35% of the image, with overlap), plus a random 60 of the remaining
boxes to estimate how many other birds hide there. Same labels as A1: `land animal`, `bird on wire or in flight`, `unsure`.
Results go in a file such as `validation_animal_classes.json`; validation images and labels are never edited.

**B2. Re-score without touching the validation set.** Extend `setup/analyse_detection_size_brightness.py`
(or add a sibling script, leaving the old one untouched) to read B1 and report, on the same 17,675 images:
- animal recall and AP50 on **all** boxes (the old number, for continuity),
- animal recall and AP50 on **land animals only**, with `unsure` and wire/sky-bird boxes excluded from the targets,
- how many *predicted* animals land on excluded boxes, so we can see whether the model's "animal" predictions
  are really wire birds.
- Run at the recorded setting (1536 input, NMS 0.6, once that is written into `models/perception_models.json`).
  Inference only; the GPU has had two CUDA errors, so check the card is idle first and keep FP32.

**B3. How to report it honestly.** This is a **different evaluation protocol**. It will probably look better than
11.5% because the hardest, least relevant boxes are removed. That is a fairer baseline, not progress. Report the
old and new numbers side by side and never compare a land-only score with an old all-box score as if they were the
same measurement.

## Part C — Order of work and approvals

1. Review sheets for A1 and B1 (no GPU, no dataset change). *Needs: approval to generate sheets and run reviewers.*
2. Write the 1536 / NMS 0.6 setting into `models/perception_models.json` (already approved for "before next training").
3. Build and audit the filtered view (A2–A4). *Needs approval: creates a new dataset folder.*
4. Land-only re-score (B2). *Needs approval: one GPU inference pass, about 10 minutes at 1536.*
5. Only then, the animal-data additions (63 accepted recovered-IDD images, DATS animals) and the short probe from
   the earlier plan, on the filtered view. *Needs approval and a class-mapping decision.*

## Results of the review sheets (29 October; model reviewers, not human verification)

Eight reviewers covered every training animal box and 330 validation boxes
(`runs/animal_filter_review_20261029_01/`, merged in `verdicts_merged.json`; sheet builder
`setup/make_animal_filter_sheets.py`). Rule 2 (project decision): boxes too small to identify are left out.

**Training view (860 boxes in 120 images)**

| Verdict | Boxes |
| --- | ---: |
| Birds on wires or in flight (do not count) | **627** (73%) |
| Land animals (count) | **148** (17%), in 67 images |
| Too small to identify (left out) | 83 (10%) |
| Not an animal | 2 |

- Only **148 boxes in 67 images** are real, identifiable land animals; median size 36 px.
- Test of the simple rule "drop if under 20 px or in the top quarter": it removes **625 of 627 bird boxes** (good) but also
  **24 of 148 real land animals** (16%, all small ones) and 50 of 83 unsure boxes. So use the per-box verdicts, not the rule.
- Caveat on "left out": deleting a label turns that object into background. For unsure boxes that may really be animals,
  prefer dropping the whole image or leaving the box out of the filtered view with a recorded reason; decide in Part A.

**Validation (963 animal boxes; 270 suspicious + 60 random others reviewed)**

| Group | Land animal | Bird (wire/flight) | Too small | Not animal |
| --- | ---: | ---: | ---: | ---: |
| 270 suspicious (under 20 px or high in frame) | 82 | 103 | 81 | 4 |
| 60 random others | 45 | 0 | 13 | 2 |

- Estimated (extrapolating the random 60 to the other 693 boxes): about **600 land animals, about 245 too small to identify,
  about 100 birds, about 25 not animals** out of 963. The extrapolation assumes the 60 are representative.
- So validation is mostly land animals, while training is mostly wire birds: the mismatch is now measured, not guessed.
- **Decision (project decision, 29 October): birds standing on walls, rooftops or other structures are excluded too.** Only birds on the
  ground count. This moves the egrets on a wall (validation ids 32–35) from land animal to excluded, which lowers the
  validation land-animal estimate by about 4 boxes. Any other wall- or roof-standing bird the reviewers counted as a
  land bird (check the notes in `verdicts_merged.json` for "wall", "roof", "fence") must be re-labelled before the
  filtered view is built.

**Final rule, as of 29 October:** `animal` = land animals, plus birds **on the ground**. Excluded: birds in flight, on wires or poles,
and on walls, rooftops or other structures; anything too small to identify.

## Part A result: the new view was built (29 October, project decision)

`dataset/idd117k_detection_v4_landanimals/` exists. v3, v2, the raw data and the DATS archive were not touched. No training was run.

| | v3 (before) | v4 (now) |
| --- | ---: | ---: |
| Train images | 56,729 | 56,896 |
| Animal boxes in train | 860 | **528** |
| Train images with animals | 120 | **254** |
| Share of animal boxes that are wire or flying birds | 73% | about 0 (removed) |
| Median animal box (px, at 1920×1080) | 9 | **59** |
| Animal boxes under 20 px | 80% | 15% |
| Animal box centre in top quarter | 29% | 0.9% |
| Validation images | 17,675 | 17,675 (byte-identical) |

- Changes: 20 images left out (unidentifiable animal boxes); 100 existing animal images processed (34 labels actually changed, mainly wire-bird boxes
  deleted; the rest were already all land animals); **105 recovered IDD images and 82 DATS images added**; 20 DATS images held out in `holdout_dats/`.
- Fewer than the earlier estimate (about 279 images / 620 boxes): images with poorly fitting animal boxes were **not added** (9 recovered IDD, 16 DATS),
  because there was no human to fix them.
- 15 independent checks all passed: file pairing, validation identical to v3, label format, only animal lines changed in inherited labels,
  counts add up, excluded and held-out images absent, no recording overlap with validation (`verification.json`).
- Still not done: the land-only validation re-score (Part B), the inference-setting change in `models/perception_models.json`, and any training.

## Part B progress: land-only validation copy built (29 October)

All 963 validation animal boxes are now reviewed (330 earlier + 633 in `runs/animal_val_review_20261029_01/`). Verdicts: 667 land animal,
111 wire/flying/wall bird, 176 too small, 9 not animal. Built `dataset/idd117k_detection_v4_landanimals/val_land/` and `data_land.yaml`:
17,603 images (72 left out), 487 animal boxes; the original `labels/val` is untouched (`setup/build_land_only_validation.py`,
`setup/verify_land_only_validation.py`, 6 checks passed). The GPU re-score on `val_land` has not been run.

## Overnight training (29 October, project decision, scheduled to finish by 08:00)

`runs/detection_train_20261029_01`: one epoch on the v4 view from the epoch-3 checkpoint, 1280 input, FP32, batch 2, lr 3e-5,
28,448 batches (about 3¼ h). Afterwards the same script scores the new best checkpoint at 1536/NMS 0.6 and 1280/NMS 0.7 on the original
validation (`runs/detection_v4_eval_20261029_*`). Live dashboard: http://127.0.0.1:8766. No automatic deployment.

### Result (measured on validation, 29 October; not a test result, not deployed)

Training finished normally at 06:34 IST (exit 0, losses finite). New checkpoint: `runs/detection_train_20261029_01/weights/best.pt`
(sha256 starts `806a7163`). Old = epoch-3 checkpoint `f7bb7e4c`. Same 17,675 original validation images, same settings on both sides.
Validation is reused for model choice, so these are development numbers, not an independent test.

| Setting | Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1536 / NMS 0.6 (recorded setting) | old | 0.7444 | 0.5837 | 0.6481 | 0.4268 |
| 1536 / NMS 0.6 (recorded setting) | **new** | 0.7462 | 0.5953 | **0.6541** | **0.4300** |
| 1280 / NMS 0.7 (old default) | old | 0.7423 | 0.5578 | 0.6224 | 0.4152 |
| 1280 / NMS 0.7 (old default) | **new** | 0.7426 | 0.5654 | **0.6271** | **0.4179** |

- **Animal (all 963 original boxes, including the wire birds):** AP50 0.114 → **0.200** at 1536/0.6 (+8.6 points), recall 4.1% → 12.1% at 1280.
  This is the old protocol; the land-only score is below once it finishes.
- **Classes that dropped by more than 0.5 AP50 points:** bicycle −0.94 (1536) / −0.89 (1280); traffic light −0.71 (1536); bus −0.52 (1280 only).
  The other classes moved by less than ±0.3. These drops are small, but they look real rather than random (bicycle drops at both settings), so they need watching.
- **20 held-out DATS images (68 boxes, 47 of them animals), 1536/0.6:** mAP50 0.546 → 0.579, animal AP50 0.056 → 0.291, but precision 0.68 → 0.50 and
  person AP50 0.49 → 0.20 (5 person boxes). Far too small to judge on; read it as a sanity check only.
- **Land-only validation (`val_land`: 17,603 images, 487 land-animal boxes), 1536/0.6. A DIFFERENT PROTOCOL; do not compare with the
  rows above.** Old → new: mAP50 0.6521 → **0.6643**, mAP50-95 0.4286 → 0.4354, precision 0.738 → 0.743, recall 0.589 → 0.605.
  **Animal AP50 0.159 → 0.320** (+16.1 points), AP50-95 0.069 → 0.159. The other classes match the original-validation changes
  (bicycle −0.90, traffic light −0.72; everything else within ±0.3). Same checks passed (protected files and checkpoint unchanged).
- Nothing was deployed: `models/perception_models.json` still points at the old checkpoint.

### Queued follow-up scoring (runs after the overnight chain, inference only, 1536 / NMS 0.6)

`runs/detection_v4_animal_prep_20261029_01/run_followup.sh` waits for the two overnight evaluations, then scores the old (f7bb7e4c) and new
checkpoints on the 20 held-out DATS images and on `val_land`:
`runs/detection_v4_holdout_20261029_{old,new}/analysis.json` and `runs/detection_v4_valland_20261029_{old,new}/analysis.json`
(logs in the prep folder). `val_land` and the holdout are different protocols from the original validation; report them separately.

### Training outcome, re-checked at the 07:05 morning check (29 October)

All numbers above were re-derived from the `analysis.json` files by `setup/compare_v4_overnight.py`
(`runs/detection_v4_comparison_20261029/comparison.json`) and match. The follow-up scoring finished at 07:21 IST (exit 0); nothing is running.

**Land-only validation (`val_land`, 17,603 images, 487 animal boxes, 1536 / NMS 0.6) — a different protocol from the table above:**

| Model | Precision | Recall | mAP50 | mAP50-95 | Animal AP50 | Animal recall at conf 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old (f7bb7e4c) | 0.7376 | 0.5886 | 0.6521 | 0.4286 | 0.159 | 29 / 487 |
| **new (806a7163)** | 0.7432 | 0.6049 | **0.6643** | **0.4354** | **0.320** | **105 / 487** |

Drops over 0.5 AP50 points on `val_land`: bicycle −0.90, traffic light −0.72 (same as the original validation). Both models are scored on the same
land-only boxes, so the animal gain is not caused by the easier protocol. **Outcome:** Part A and Part B are done and the first training on the
filtered view helped animals without a large regression elsewhere. The new checkpoint is **not deployed**; deployment is a separate, still-open decision. Matched comparison record:
`runs/detection_v4_comparison_20261029/comparison.json` (local output, not included).

## Risks and open questions

- **Rule errors.** A size and position rule will drop some real distant animals and keep some large birds. The full
  review in A1 exists to measure that; do not skip it.
- **Reviewer quality.** Earlier model reviews marked 122 of 194 images "uncertain" because boxes were tiny. Expect many
  `unsure` calls; they are excluded rather than guessed.
- **Chickens and other land birds.** They count, but they are small. Make sure they are not dropped by the size rule:
  review any `keep`/`drop` that depends on a bird being on the ground.
- **Evaluation drift.** Changing what validation counts as `animal` breaks comparison with every earlier animal number.
  Keep both versions in every report.
- **Disk and time.** Hard links make the new view nearly free; the review is the main cost (roughly 1–2 hours of reading).
- **Public repository.** Data and review sheets stay local; only scripts and written summaries are candidates for release.

## Decisions still open (as written in the original plan)

1. OK to generate the A1 and B1 review sheets and run reviewers (no dataset changes)?
2. OK to treat `unsure` boxes as excluded from training and from the land-only score?
3. After the filtered view exists, approval for the land-only re-score as a separate step.
