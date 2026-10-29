# Animal-data plan — 29 October 2026 (PLAN ONLY, nothing below has been run)

**Goal:** raise the detector's `animal` class (AP50 11.5%, recall 4.7%, 860 training boxes) without
hurting the other eleven classes. **Status:** proposal. No training, no training-set change, no default
change. Every step that touches GPU, training data or a model needs an explicit project decision.

Related evidence: [animal label review](ANIMAL_DATA_REVIEW_20261028.md), [detector error review](DETECTION_ERROR_REVIEW_20261028.md).

DATS_2022 attribution: Paranjape, *DATS_2022*, Mendeley Data, Version 2, DOI [10.17632/nfc34n8svj.2](https://doi.org/10.17632/nfc34n8svj.2), licensed CC BY 4.0. Only counts, notes and scripts are published here; the images and annotations are not redistributed, and any local use changes them (EXIF rotation baked in, resized, relabelled to the project classes).
Pending decision recorded earlier: 1536 input / NMS 0.6 becomes the recorded inference setting **before**
any new training session.

## What we have (counts from files, not accuracy claims)

| Source | Animal boxes | State |
| --- | ---: | --- |
| Current training view (v3) | 860 | in use |
| DATS_2022 subset (`dataset/dats2022_raw/animal_subset/`) | 306 in 146 images | extracted; upright sample looks clean |
| IDD images removed for conflicting duplicate labels | up to 918 in 276 images | 194 groups where the animal is the only conflict; sample mostly real, many tiny |
| Validation (must stay untouched) | 963 | IDD only |

## Decision on birds (project decision, 29 October)

**Birds and chickens on land count as `animal`; flying birds do not.** Applied to the recovered-IDD review:

- Land (count): id 59 (two chickens on the ground), id 125 (chickens on the roadside), id 154 (bird standing at a wall).
- Flying (do not count): ids 39, 43, 52, 122, 144 (birds or specks in open sky). Their boxes must be removed, not trained as animals.
- Perched on a wire or pole (**decided 29 October (project decision): exclude**): id 150 and one box of id 153, and the
  wire-bird flocks that make up most of the existing training animal labels.

**Final rule: animal = land animals and land birds (dogs, cattle, buffalo, goats, horses, camels, chickens on the
ground). Not animal = birds in flight, birds perched on wires or poles.**

**Measured (29 Oct, label files only, no images viewed):** the existing `animal` labels look very different
in training and validation.

| | Validation (963 boxes) | Training (860 boxes) |
| --- | ---: | ---: |
| Box centre in the top quarter of the image | 6.0% | **29.2%** |
| Box centre in the top 35% | 10.4% | **66.9%** |
| Median box size (native 1920×1080) | 31 px | **9 px** |
| Boxes under 20 px | 27.7% | **80.5%** |

Training animals are mostly tiny specks high in the frame, which fits flying birds, while validation animals
are larger and mostly on the ground. So the real number of *land* animals in training may be only
about 100–300 of the 860 (upper-bound guess if everything in the top 35% is a sky bird). This is a position-and-size
heuristic, not a verified count: a cow on a hill would also sit high. Consequences: (1) the recovered-IDD and DATS
images are a larger share of usable land-animal data than first thought; (2) the 860 training boxes may need
filtering or relabelling under the "land birds yes, flying birds no" rule; (3) validation animal recall should be
re-measured on land animals only. None of this has been changed or run.

**Confirmed by looking (29 Oct, `runs/animal_label_review_20261028_03/`, two 24-box samples):**

- All 24 sampled top-quarter boxes are **small birds perched on wires and poles**, in near-identical scenes (the
  same pole and flock, again and again). Another 12 of the 24 "rest" boxes are also wire birds.
- The other half of the "rest" sample are real land animals: cattle, buffalo, goats and dogs.
- Only **120 training images contain any animal box at all.** Of the 860 boxes, **692 are tiny (under 20 px)
  and sit in just 55 images**; only **168 boxes in 83 images** are 20 px or larger. So the model has effectively
  seen on the order of **80 images of land animals**.
- That makes the recovered-IDD (63 accepted images) and DATS (146 images) additions a large relative gain in
  distinct animal scenes, roughly 80 → 290, though still small in absolute terms.
- **Decision made (project decision, 29 October): birds perched on wires/poles are excluded.** Consequences, none done yet:
  1. **New Step 0 — filter the existing training animal labels.** About 692 tiny boxes (under 20 px) in 55 images
     are almost all wire birds. Each image needs a decision: delete the box, or turn the label into background
     (note: a deleted label in a photo still containing the bird makes the bird "background", which is the intended
     meaning here). Do it in a *new* training view, never by editing `idd117k_detection_v3`.
  2. **Re-baseline validation animal recall on land animals only.** The 963 validation animal boxes include some wire
     and sky birds (about 6–10% sit high in the frame). Re-score recall and AP with those removed or reported
     separately, so before and after are comparable. Changing the evaluation is a different protocol and must be reported as such.
  3. The animal-only probe (Step 5) must run on the filtered view.

## Step 1 — Class mapping (decide before converting anything)

**Biggest trap:** YOLO training treats anything unlabelled as background. If we convert only the animal
boxes, the cars, people and motorcycles in the same photos become "background" and the model learns to
ignore them. So every DATS image used must have *all* relevant objects mapped, or be excluded.

Proposed DATS → project class (to be confirmed by the team):

| DATS name | Project class | Note |
| --- | --- | --- |
| cattle, goat, dog, camel, horse | animal | the point of this work |
| person, pedestrian-type labels | person | |
| car | car | |
| bike | motorcycle | **DECIDED (project decision, 29 Oct): bike = motorcycle.** Risk: DATS has no `rider` class, so a person sitting on a bike may be labelled only `person`; our model separates rider from motorcycle. Check 30 images before training, and consider treating DATS riders as unlabelled risk |
| cycle | bicycle | |
| rikshaw | autorickshaw | |
| bus, truck | bus, truck | |
| tempo, tractor, ambulance, crane, road roller | other_vehicle or truck | needs a rule |
| traffic signal | traffic_light | |
| traffic sign board, road sign board | traffic_sign | |
| bullock cart | **not animal** | a vehicle; decide: other_vehicle or drop the image |
| lamp post, tree, building, road, board, ... | ignore | IDD does not label these either |

Rule: an image is used only if every non-ignored object in it maps cleanly; otherwise drop it. Count how
many of the 146 survive before going further.

## Step 2 — Convert DATS safely

1. Open each image with EXIF rotation applied (73 of 146 are stored sideways; boxes are for the upright view).
2. Save a new JPEG with the rotation baked into the pixels and the orientation tag removed, so the training
   loader cannot rotate it a second time.
3. Scale boxes to the upright size (read real size from the file; 44 XMLs say 0×0). Resize the long side to
   about 1920 to match IDD scale; very large originals (3472×4624) would otherwise dominate memory.
4. Write YOLO text files, then draw boxes on a sample of 50 converted images and look at them.
5. Output goes to a new folder (for example `dataset/animal_extra_v1/`), never into `idd117k_detection_v3`.

## Step 3 — Choose which recovered IDD images to accept

1. Start from the 194 groups where the animal is the only difference between copies.
2. A person reviews each (a contact-sheet tool makes this about an hour): accept, reject, or fix the box.
   Reject boxes only a few pixels wide unless clearly an animal.
3. Check their recording IDs do not overlap validation recordings. By how v2 was built they should not,
   but verify from `provenance/source_manifest.csv` and record the result.
4. Keep the accepted copy of each image and its label; record every decision in a file.

## Step 4 — Keep evaluation honest

- **Validation stays exactly the same** (17,675 IDD images). Nothing from DATS or recovered IDD enters it.
- Hold out about 20 DATS animal images as a small **out-of-domain check**. Never train on them.
- Scoring uses the existing script `setup/analyse_detection_size_brightness.py` at the recorded setting
  (1536, NMS 0.6) so before and after are comparable. Report animal AP50/recall by size, all 12 class
  AP50 values, and the confusion row for animals.
- **Guard:** flag any other class that falls by more than 0.5 AP50 points, as in the 25 October trial.
  Do not call the result an improvement unless animal rises and nothing important falls.

## Step 5 — Smallest useful experiment first (needs approval)

Do not start with a multi-hour full-dataset run.

- **Probe:** fine-tune from the epoch-3 checkpoint for one short pass on a **mixed set**: all accepted animal
  images plus a random sample (about 3,000) of ordinary IDD training images as "replay", so the model does not
  forget other classes. Low learning rate (3e-5, as in the 1280 trial), FP32, save every epoch.
- Expected cost: roughly an hour or less (not measured; a tiny smoke test is to be timed first). A full
  3-epoch run at 1536 is estimated at 16–18 hours (extrapolated from the 1280 trial) and is a separate decision.
- Compare against the unchanged epoch-3 checkpoint under identical settings.
- Success means animal AP50 moves clearly beyond what was measured at 1280 and 1536 (11.5% and 10.9–11.4%) with
  no regressions above the guard. If it does not, record that result and stop; do not repeat the same run.

## Risks to watch

- Very few distinct animals: DATS frames come from short videos, so near-identical frames can inflate counts.
  Count distinct scenes, not just boxes.
- Camera and style differences: DATS are phone photos with close-ups and dashboards; IDD is a windshield camera.
- Over-correcting: more animal examples can raise false animal detections (precision was already 53.6%).
- The GPU has had CUDA errors twice. Keep FP32, per-epoch saving, and a recovery checkpoint.
- Licence: DATS is CC BY 4.0, so credit the source and say we changed it (resized, relabelled, rotated).
  The IDD terms apply to the recovered images as before.
- Do not mix any of this into the public repository: data stays local.

## Decisions still open (as written in the original plan)

1. Approve the class mapping table above, especially bike, tempo and bullock cart.
2. Approve a person reviewing the 194 recovered-IDD candidates.
3. Approve the one-hour probe after Steps 1–4 are done, and the order relative to the detector setting change.

No other DriveIndia or outside data is included here; it can be added later once its licence and class list
are confirmed.
