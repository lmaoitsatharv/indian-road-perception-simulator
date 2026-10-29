# Animal-class data review — 28 October 2026

Why: the detector finds only 4.7% of validation animals (AP50 11.5%) and has only **860 animal boxes** in
its training view. This note records what extra animal data exists. **Nothing was trained, nothing was
added to any training set, and no image was moved.** Counts are from files; the visual check is a
24-image sample, not a full review.

## 1. Animals hidden in the cleaned-out IDD images (already on disk)

When v3 was built, 5,045 training images were removed because the same pixels had conflicting labels.

- **276 of those images contain 918 animal boxes** (more than the 860 now in training), in 275
  duplicate-pixel groups from 174 different recordings.
- In **274 of 275** groups, exactly **one** copy has the animal and the other copy does not. So the
  "conflict" is usually that one annotator labelled an animal and the other missed it.
- In **194 of 275** groups, the animal is the *only* difference between the copies. In the other 81, other
  boxes also differ.
- A random sample of 24 from those 194 is in `runs/animal_label_review_20261028_01/contact_sheet.jpg`
  (red box = animal label). Reading it by eye:
  - **About 8 look clearly right:** dogs, goats and cattle you can see plainly.
  - **About 10 are tiny boxes** (a few pixels across, far away) that cannot be confirmed either way.
  - A few are odd (for example, one thin horizontal sliver, one large dark box under trees).
- So these labels look **mostly genuine but not clean**. The 194-group set could add up to roughly 300
  boxes after a quick human check, but that figure has **not** been verified. Nothing should be added
  without that check, and the animals' recordings must not overlap validation.

## 2. DATS_2022 (Mendeley Data, DOI 10.17632/nfc34n8svj.2, CC BY 4.0)

DATS_2022 attribution: Paranjape, *DATS_2022*, Mendeley Data, Version 2, DOI [10.17632/nfc34n8svj.2](https://doi.org/10.17632/nfc34n8svj.2), licensed CC BY 4.0. Only counts, notes and scripts are published here; the images and annotations are not redistributed, and any local use changes them (EXIF rotation baked in, resized, relabelled to the project classes).

- Full dataset: 10,245 files, **10.7 GB** (images about 8 MB each, phone photos, mostly 3472×4624).
  The paper's "1,000-image subset under 1 GB" is not what the Mendeley page serves.
- **Only the 2,941 XML annotation files (4.9 MB) were downloaded**, all matching their published SHA-256,
  into `dataset/dats2022_raw/annotations_xml/`. No images yet.
- 61 raw class names (some are typos), 17,194 boxes. Animal-like classes:

| DATS class | Boxes | Images |
| --- | ---: | ---: |
| cattle | 185 | 78 |
| dog | 66 | 50 |
| goat | 49 | 14 |
| camel | 13 | 7 |
| horse | 4 | 3 |
| bullock cart (vehicle, not animal) | 6 | 6 |

- That is **about 317 animal boxes in roughly 150 images**: a useful top-up (+37% on 860), not a fix.
- Caveats: frames come from 10–15 s videos, so many images are near-duplicates; 708 annotation files list
  image size 0×0; phone photos differ in style from IDD's cameras; the 12-class mapping is a judgement
  (cattle, dog, goat, camel and horse → `animal`).

### Update, 29 October: full DATS zip downloaded and checked

- The full zip was downloaded manually to `dataset/DATS_2022.zip` (10,701,624,652 bytes).
  `setup/verify_dats2022_download.py` checked **all 10,199 unique file contents** against the saved Mendeley
  SHA-256 values: **0 missing, 0 unexpected, zip CRC clean. Passed.** (Report: `dataset/dats2022_raw/download_verification.json`.)
- Only the animal images were unpacked, not all 10.7 GB: **146 image + XML pairs, 0.56 GB**, in
  `dataset/dats2022_raw/animal_subset/` (mapping in `SUBSET.json`). They hold **306 animal boxes**: cattle 179,
  dog 62, goat 49, camel 13, horse 3. Six animal XML files (11 boxes) could not be paired with an image.
- **Orientation trap:** 73 of the 146 photos are stored sideways (EXIF orientation 6). Their boxes are
  written for the *upright* picture, so the image must be rotated first (EXIF transpose) or every box will
  be in the wrong place. 44 XML files also report image size 0×0 and need the real size read from the file.
- With rotation applied, a random 24-image sample (`runs/animal_label_review_20261028_01/dats_contact_sheet_upright.jpg`)
  shows boxes that line up well on **clearly real animals** (cattle, buffalo, dogs, goats, camels), better
  quality than the recovered IDD set. Many are close-up phone or dashboard-camera views, unlike IDD's
  cameras. A few boxes are very small and one image shows bullock-cart oxen.

### Update, 29 October: the 194 recovered-IDD candidates were checked

Four reviewers looked at all 194 candidates on numbered contact sheets (`runs/animal_label_review_20261028_02/`,
merged verdicts in `verdicts_merged.json`, sheet builder `setup/make_animal_review_sheets.py`).
**These are model visual judgments of downscaled sheets, not human verification.** A spot check of the first
sheet found the calls on it fair.

| Verdict | Images | Animal boxes | Boxes with longest side ≥ 25 px |
| --- | ---: | ---: | ---: |
| **Accept** (every box on a real animal, reasonable fit) | **63** | **96** | 89 |
| Uncertain (tiny, dark or blurry; cannot confirm) | 122 | 430 | 225 |
| Reject (box on sky, wires, shadow, fixtures) | 9 | 27 | 2 |

- Accepted animals are mostly dogs (37) and goats (10), then cattle (10) and buffalo (3); three are other or unclear.
- **Most boxes are tiny** (many 5–20 px). Those are the hardest cases for the model, and also the hardest to
  label. They cannot be judged at this sheet resolution; a human would need native-size crops.
- Birds and chickens appear in several images (for example ids 39, 43, 52, 59, 122, 125, 144, 150, 153, 154).
  Whether poultry and birds count as `animal` is a policy decision; they are not counted as accepted here.
- Four images may contain an unlabelled animal (ids 11, 133, 166, 180); these are weak sightings.
- **Honest takeaway:** the safely usable recovery is about **63 images / 96 boxes**, far less than the 918 raw
  boxes. The other roughly 120 images need a human check at full size before use.

### FINAL COUNT, 29 October (all sources reviewed box by box; model reviewers only, by project decision not to wait for human review)

Rule applied: land animals and birds on the ground count; birds in flight or on wires, poles, walls or rooftops do not; boxes too small to identify
are not counted, and **any image containing such a box is left out** (leaving a real animal unlabelled would teach the model it is background).
944 boxes were re-reviewed at box level (`runs/animal_final_review_20261029_01/`, merged status in `image_status.json`).

| Source | Images with usable land animals | Land-animal boxes | Left out |
| --- | ---: | ---: | --- |
| Existing training (after removing wire/flying birds and unclear boxes) | **67** | 172 | 20 images with unclear boxes; 33 images with only birds |
| Recovered IDD images | **114** | 237 (10 poor fit) | 68 images with unclear boxes; 12 with no usable animal |
| DATS_2022 | **142** | 295 (23 poor fit) | 3 unclear, 1 none |
| **Total** | **323 images** | **about 704 boxes** | |

- **For training: about 303 images.** About 20 DATS images are held back as an out-of-domain check, leaving about 122 DATS images (about 253 boxes), so roughly **660 boxes**.
- Compared with the starting point of 67 land-animal images and 148 boxes, that is about **4.5× the images**.
- About 37 boxes have a poor fit (partial animal, two animals in one box, box offset) and need fixing or dropping before use.
- **DATS class mapping:** only 84 of the 142 DATS images have other objects that map cleanly. The rest contain "bike" (44 images in the
  146), "tempo", "cart", "board" or "tractor", which need a mapping rule. Without one, about 58 DATS images cannot be used safely.
- **Update (project decision, 29 October): DATS `bike` = `motorcycle`.** With that rule, **118 of the 142** usable DATS images have no
  other ambiguous class. The remaining 24 are blocked by `tempo` (9), `cart` (8), `board` (8) and `tractor` (2) and still need a rule
  or must be dropped. Check for riders labelled only as `person` before training.
- **Update (project decision, 29 October): the 24 DATS images with tempo, cart, board or tractor are dropped.** Final DATS selection:
  **118 images, 256 land-animal boxes** (list in `dataset/dats2022_raw/dats_final_selection.json`). The 24 dropped images held 39 land-animal boxes.
- **Final totals after both decisions:** 67 (existing) + 114 (recovered IDD) + 118 (DATS) = **299 images, about 665 boxes.** With about 20 DATS images
  held back as the out-of-domain check, **about 279 images and about 620 boxes** are available for training.
- All verdicts are model readings of zoomed crops, not human checks.

## Combined picture

About 860 (now) + ~300 (recovered IDD, after checking) + ~317 (DATS) is roughly 1,500 animal boxes at best,
under 2× today's number. That helps, but animal data will stay far smaller than the other classes
(person 114,695 boxes, other_vehicle 161,007). Realistic gain is uncertain; test it with a matched
validation comparison before drawing any conclusion.

## Next steps (planned, none started)

1. Decide which DATS images to fetch (only the ~150 with animals, instead of 10.7 GB).
2. Human check of the 194-group candidates; record which are accepted.
3. Write a class-mapping note, keep every new image out of validation, then plan a short matched fine-tune.
