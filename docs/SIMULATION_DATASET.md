# Separate Indian-style simulation dataset

This guide records preparation of a new **simulation-only** road segmentation
dataset. Preparation finished without training on 21 October. A separate training
run began on 22 October; see [SIMULATION_TRAINING.md](SIMULATION_TRAINING.md).
The locked model for real photos/videos remains unchanged.
The machine-readable completion evidence is local at
`dataset/simulation_india_v5/dataset_report.json` and
`dataset/simulation_india_v5/independent_verification.json`. Their existence and
successful status, rather than this implementation guide, establish completion.

## Work stages

| Step | Deliverable | Scope |
|---|---|---|
| 1 | Freeze the protocol and protect the real model | Saved hashes, seeds, whole-episode splits, label policy |
| 2 | Build different left-hand road layouts | Crossroads, north T-junction, south T-junction, straight street |
| 3 | Generate fresh traffic runs automatically | 32 episodes, each 80 simulated seconds, synchronized native camera streams |
| 4 | Select 5,000 useful images and generate labels | Reject repeated RGB images and low-change candidates; no model-generated labels |
| 5 | Split and audit | 4,000 train / 500 validation / 500 test; every episode/seed belongs to one split |
| 6 | Preprocess and inspect | 512x512 images, binary masks, valid-pixel masks, real TensorFlow batch checks and galleries |
| 7 | Train and evaluate a separate simulator model | Completed 22 October: epoch-10 best checkpoint, 500-image test IoU 99.9288%; see SIMULATION_TEST.md |

## Variation and what the maps mean

These are four different generated OpenDRIVE road layouts. They
share a small neighbourhood asset kit and two-way, left-hand lanes. They are
not four downloaded Indian cities, and do not represent every Indian road type.
No right-hand stock-town traffic is included in the prepared dataset. The stock
CARLA town is only an engine startup step before the custom map replaces it.

Each topology has six training episodes, one validation episode and one test
episode. Traffic cases include mixed actors, slow cycles and overtaking,
oncoming vehicles, pedestrian crossings, return-gap interactions and junction
traffic. Routes that do not exist on a T-junction/straight map are omitted.
The junction-only case is not assigned to the straight map.

Clear, overcast, wet and dusk conditions, sunlight azimuths, vehicle colours,
initial longitudinal positions, desired speeds and pedestrian timing vary.
Windscreen camera heights are 1.25/1.35/1.50 m, fields of view 90/95/100 degrees,
with modest recorded pitch/yaw changes. Calibration is saved for each episode;
all evidence cameras share exactly the same pose and settings. No image is
counted twice by augmentation.

The final capture uses explicit atmospheric scattering and the native cloudy
preset's cloudiness 60 for cloudy/wet scenes. A seven-condition native probe
showed that 85 produced invalid black daytime skies in this custom low-quality
world. Those pilot versions were rejected before training. Every final daytime
frame with visible sky must pass a sky-brightness gate; diagnostic weather probes
are explicitly barred from dataset preparation. The atmosphere parameters follow
the [CARLA 0.9.16 weather API](https://carla.readthedocs.io/en/0.9.16/python_api/#carlaweatherparameters).

The neighbourhood uses installed generic buildings, kiosks, food carts, palms,
raised paving, parking and left-hand traffic. Available vehicles include cars,
vans, buses, scooters, motorcycles and cycles. There is no authentic auto-rickshaw
asset. This is an Indian-style simulation, not photographic authenticity or a
claim of calibrated traffic behaviour. Sparse/dense actor counts are reported
from actual scenes rather than invented labels.

Every episode captures at 5 Hz. Training selects 166/167 changing views from its
approximately 400 native frames; validation/test select 125 each over the same
80-second window. The original candidate footage is retained. Exact duplicate
RGB pixels and low-change consecutive candidate views (48x27 RGB mean absolute
difference below 0.75 on a 0..255 scale) are excluded before uniform temporal
selection. If too few useful views remain, preparation fails rather than padding
the dataset. Independent verification checks resized-image uniqueness too and
reports cross-split visual similarity. Related maps and frames remain correlated;
whole-episode splitting reduces temporal leakage but does not establish city-level
generalisation or readiness for an actual vehicle.
Each held-out split has four whole episodes, so individual case families are not
all represented separately in both held-out splits. The final report exposes
case-by-split counts; mixed-traffic episodes include several interactions. Future
claims about every scenario family need broader independent evaluation.

## Label contract

Labels come from the synchronized semantic/instance/depth cameras and the actual
OpenDRIVE driving-lane footprint. The road model is never used as an annotator.
Known visible pedestrians, vehicles, buildings and footpaths are non-road.
Road markings belong to the road. Occluded road behind an actor is not labelled
as visible road. A geometric road mask is not a free-space or obstacle detector:
it does not declare every road pixel safe to drive through.

Custom asphalt has no native Road tag, so only its exact owned instance IDs and
unlabelled surfaces can be resolved using depth, ground height and map geometry.
A conservative 0.2 m boundary neighbourhood and an 80 m geometric range limit
leave uncertain pixels ignored. Semantic/instance class disagreements up to 0.1%
of a frame are retained in the source evidence but individually ignored. Larger
disagreements stop capture. This is necessary for occasional one-pixel renderer
edges at varied viewpoints; the default real/simulator evaluator is not loosened.

| Saved folder | Pixel values | Meaning |
|---|---|---|
| `images` | RGB uint8 | Bilinear resize from native 960x540 to 512x512 |
| `masks` | 0 / 1 | Binary visible road target; invalid positions are zero placeholders |
| `valid` | 0 / 1 | Whether a target pixel may enter loss/metrics |
| `reference` | 0 / 1 / 255 | Auditable non-road / road / ignore labels |

**A binary-mask zero where `valid == 0` is not a background annotation.** Training
and evaluation must exclude that pixel. This avoids silently changing uncertain
road boundaries into false non-road labels. Labels and validity use identical
nearest-neighbour resize coordinates. They are never divided by 255.
Raw 0/1 mask PNGs look nearly black in an ordinary photo viewer; the preview
gallery displays road as white/green and ignored pixels as magenta.

## Matching the established real-model method

`simulation_data.py` imports the unchanged real loader's normalization and the
existing paired/shadow augmentation helper. It returns `(image, mask, valid)`.
RGB becomes float32 and is normalized exactly once to `[-1, 1]`; target and valid
remain float32 0/1 with shape `[512, 512, 1]`. Training shuffles paths with seed 42
and flips image/target/valid together. Validation/test have neither shuffle nor
augmentation. The final smaller batch is retained; no full image cache or
pre-normalized float copies are created. Default batch size is 4.

The saved training recipe follows the real model's staged method:

1. Fresh decoder plus the existing cached ImageNet MobileNetV2 encoder, matching
   the U-Net architecture. Freeze the encoder for the initial phase; Adam 1e-3,
   batch 4, up to 20 epochs, checkpoint by validation road IoU, the same callback
   settings, seed 42 and threshold 0.5.
2. After validation review, a separately recorded fine-tuning run may use the
   winning real-run settings: blocks 13-16 eligible, all BatchNorm frozen,
   mixed precision with float32 output, batch 16, Adam 3e-6, existing shadow
   augmentation, up to 8 epochs. These are future settings, not performed runs.
3. Keep BCE + soft Dice and global TP/FP/FN metrics, explicitly masked by `valid`.
   The original real trainer assumes fully labelled masks and fixed real counts;
   **do not feed this dataset into `train_idd.py` or ordinary unmasked Dice**.
   Step 7 now uses the separate masked adapter in `simulation_model.py` and the
   protected training entrypoint `train_simulation.py`.
4. Select on simulation validation only. Freeze the selected checkpoint/protocol
   before simulation-test model scoring. Never mix real validation/test examples
   into simulation training. File/label QA of the new simulation test split is
   allowed during preparation and is not a model accuracy evaluation.

## Files and reproduction

The dataset is separate under `dataset/simulation_india_v5/`, excluded from Git
like the real dataset. `plan.json` fixes conditions/splits before capture;
`capture.json` links owned native sessions; `provenance.jsonl` connects every
selected image to sensor frame, calibration and hashes. `episode_audit.json`
preserves per-episode counts, candidate rejection and native health results.
`splits/pair_splits.csv` is the portable split manifest.

Run from the project folder, using a **new** dataset output path to preserve
previous attempts:

```powershell
.\.venv-carla\Scripts\python.exe setup\prepare_simulation_dataset.py plan --output dataset\simulation_india_v6
.\.venv-carla\Scripts\python.exe setup\check_simulation_maps.py --output dataset\simulation_india_v6
.\.venv-carla\Scripts\python.exe setup\prepare_simulation_dataset.py capture --output dataset\simulation_india_v6
.\.venv\Scripts\python.exe setup\build_simulation_dataset.py --output dataset\simulation_india_v6
.\.venv-carla\Scripts\python.exe setup\verify_simulation_dataset.py --output dataset\simulation_india_v6
.\.venv\Scripts\python.exe setup\check_simulation_preprocessing.py --output dataset\simulation_india_v6
.\.venv\Scripts\python.exe setup\finalize_simulation_dataset.py --output dataset\simulation_india_v6
```

The native simulator needs desktop/GPU access and its local RPC connection.
The paired encoder blocks rather than silently dropping evidence. Failed or
interrupted attempts remain saved. Original real checkpoints, deployment files,
real split and completed real-test evidence are outside all output paths.
Three bounded preprocessing workers can process completed episodes while later
ones capture. Pair IDs and split assignments are fixed by the plan, independent
of worker completion order.
