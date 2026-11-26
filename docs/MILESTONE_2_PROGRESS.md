# Milestone 2 progress report

**Project:** Indian road perception and traffic simulation, with adaptive driving integration as the next stage.
**Status date:** 26 October 2026.
**Team:** Atharv and Sarthak.
**Evaluation format:** progress review of completed work, measured results and remaining implementation. A complete autonomous-driving demonstration is not claimed.

## Problem and objective

Indian road scenes contain mixed vehicle types, pedestrians, irregular boundaries and crowded interactions. The project develops road and object perception alongside a controllable Indian-style simulation in which later driving decisions can be tested. The intended final pipeline is camera input, road/object perception, tracked obstacles, route and behavior decisions, and vehicle control with an independent emergency override.

The current deliverable has independently evaluated perception modules and a working rule-driven simulator. Camera perception has not yet replaced the simulator-position inputs used by the driver.

## Completed implementation

| Component | Implemented work | Evidence and limits |
| --- | --- | --- |
| Real road data | Paired image/mask audit, deterministic split, preserved source, cleaned manifest and checked input pipeline | 5,507 train / 687 validation / 689 test; original recording identities unavailable |
| Real road model | MobileNetV2 encoder with U-Net decoder, transfer learning, controlled fine-tuning and locked final evaluation | 91.6549% road IoU and 95.6457% Dice on 689 test images; only the locked model was tested |
| Real object data | IDD-117K archive verification, provider indexing, recording exclusions, box conversion and exact-duplicate filtering | Current training view: 56,729 unique retained training images; unchanged 17,675 validation images |
| Object detector | COCO-pretrained YOLO11s adapted to 12 project classes, 640-to-960 refinement and a matched 1280 trial | Latest 62.245% mAP50 / 41.525% mAP50-95 on project validation; no labelled independent detector test |
| Simulation data/model | 5,000 paired synthetic images with road/validity labels; whole-episode split and separate masked training | 4,000 train / 500 validation / 500 test; fixed simulation-test road IoU 99.9288%, 98.2174% valid coverage |
| Connected town | Procedural OpenDRIVE roads, five districts, nine streets, seven junctions and scenery | 25 vehicles and 21 pedestrians; generic assets and illustrative behavior |
| Driving baseline | Native throttle, brake and steering; lane following, following gaps, junction rules and pedestrian handling | Uses simulator ground truth, not learned camera control |
| Overtaking reference | Explicit pass/wait/return logic and selected scenario audits in the earlier neighbourhood | Combined-town overtaking refinement remains future work |
| Live perception preview | Matching-frame road predictions, bounded buffering, frame-age checks and disconnect recovery | Observation-only; measured redraw expiry limitation remains |
| Reproducibility | Source/configuration snapshots, checkpoint hashes, saved metrics, independent recounts and portable summaries | Large data, model weights and full run folders are supplied separately from Git |

## Data preparation and exploratory analysis

The road-segmentation copy originally contained 6,993 image/mask pairs. Audits found no exact duplicate images; the 110 all-zero masks had identical mask content. Those 110 pairs were excluded from split copies at the user's request, leaving 6,883 pairs. They were not removed because their input photographs were duplicate. Their original semantic interpretation remains a limitation. RGB inputs are float32 at 512x512 and normalized once to [-1,1]; masks remain 0/1.

The object-detection dataset is a separate collection. The provider supplied 96,897 train and 20,202 validation image identities plus 24,644 unlabelled test identities. Filtering preserved provider split assignments. Recording overlaps, unresolved uploads, invalid pairs and 6,767 exact train duplicates of validation were excluded. The resulting v2 view had 67,159 training entries. Further training-only cleaning excluded 5,045 entries from ambiguous repeated-pixel groups and 5,385 extra consistent duplicate entries, giving 56,729 unique training images. Validation remained 17,675 images / 319,438 objects.

Analysis included image sizes, class counts, object-size distributions, class-specific precision/recall/AP and selected error examples. Animals, other vehicles and small/occluded people remain important weaknesses. Exact byte/pixel checks do not establish that all near-duplicates or source annotation errors were removed.

## Model selection and technical justification

MobileNetV2 provides a relatively compact pretrained visual encoder. U-Net skip connections recover spatial detail for one road probability per pixel. BCE plus soft Dice combines probability learning with overlap learning. The initial encoder was frozen, followed by separate low-rate fine-tuning experiments. The selected real model used shadow augmentation and a 0.000003 learning rate with encoder/decoder BatchNorm held fixed.

YOLO11s is the separate bounding-box detector. COCO pretraining supplies initial visual features; training adapts the detector to the project's 12 categories. Resolution experiments isolated an important issue: larger inputs reveal smaller objects more clearly but cost additional computation. The final three-epoch trial at 1280 used AdamW, batch 2, FP32 and learning rate 0.00003. Its matched full validation improves mAP50 by 0.557 percentage points and mAP50-95 by 0.400 points. Three classes regress slightly; no universal improvement or automatic deployment is claimed.

The simulation road model was trained separately from ImageNet initialization and a fresh decoder. Invalid synthetic label pixels are masked out of both loss and metrics. Its very high synthetic IoU reflects the declared synthetic distribution and reference; it is not the real-road score.

## Current architecture

```text
Real image/video -> locked real road model -> road probabilities/mask
Real labelled images -> YOLO11s detector -> object boxes/classes/confidence
CARLA camera -> separate simulation road model -> live road preview

CARLA actor states -> rule driver -> throttle/brake/steering -> CARLA physics

NEXT: camera perception -> tracking/range -> behavior decision -> controller
NEXT: selected destination -> route planner -> route-following controller
```

These are separate current paths. Drawing a box or road mask does not itself make it an input to steering or braking.

## Progress evidence to show

1. Road preprocessing shapes and a source-versus-mask example from the owner's permitted local data.
2. Model architecture and explanation of which weights were frozen or fine-tuned.
3. Locked real-road test summary, clearly separated from detector validation and synthetic scores.
4. Detector before/after per-class chart; include animal weakness and the small regressions.
5. Connected-town map and selected simulator captures; describe the rule-based control inputs.
6. Live-road preview screenshot with matching source frame and freshness information.
7. Remaining-work checklist and responsibility split.

The [evidence gallery](MILESTONE_2_EVIDENCE.md) contains portable synthetic screenshots and aggregate charts. Raw dataset photos and third-party footage stay local.

## Remaining implementation

| Priority | Remaining work | Completion evidence |
| --- | --- | --- |
| 1 | Targeted detector error review and one justified short improvement trial | Matched validation, all-class comparison, saved failed outcomes |
| 2 | Add live object detection and tracking to the simulator | Matching frames, identity continuity, missed actors, latency and stale-track checks |
| 3 | Connect perception to bounded STOP/SLOW/PROCEED decisions | Trace showing the model-derived observation actually influences target speed, with independent braking |
| 4 | Add destination selection and route following | Reach several chosen destinations and stop; show route and goal state |
| 5 | Refine overtaking in the connected town | Clear pass, blocked pass, pedestrian conflict and return-gap scenarios |
| 6 | Broader evaluation and interface polish | Collisions/close passes, completion, waiting and comfort across additional seeds/weather |
| Later | Compare the bounded RL policy with the rule baseline | Explicit observations/reward/actions and matched held-out comparisons |

These are planned features, not completed screenshots or performance claims. Percentage readiness is a project-management estimate tied to agreed objectives; it is not the detector's accuracy and has not been independently scored as exactly 80%.

## Fit to the provided Milestone 2 rubric

| Criterion | Marks | Material to prepare |
| --- | --- | --- |
| Dataset collection, preprocessing and exploratory analysis | 1.5 | Data origin, three dataset contracts, split/cleaning table, class/object-size analysis, limitations |
| Implementation and technical progress; methodology/model justification | 1.5 | U-Net/MobileNetV2, YOLO11s, actual training settings, simulation/controller architecture and measured results |
| Progress report | 1 | This report plus evidence, responsibility plan and remaining milestones |
| Mid-term project viva | 4 | Both members explain the pipeline, their modules, metrics, failures and code entry points |
| Total | 8 | The viva contributes half the available Milestone 2 marks |

## Reading and rehearsal

Read the [beginner guide](MILESTONE_2_VIVA_GUIDE.md), use the [file guide](FILE_GUIDE.md) when a teacher points to a filename, and rehearse the [balanced ownership plan](TEAM_OWNERSHIP.md). Each member should complete the listed review exercises and accurately describe work they have personally reviewed or performed.

Source evidence: [real segmentation](results/segmentation.json), [detector matched result](results/detection_resolution_pilot.json), [simulation test](results/simulation_test.json), [connected town](results/combined_indian_town.json), [live observation](results/live_simulation_observation.json).
