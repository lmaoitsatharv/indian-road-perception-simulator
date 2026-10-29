# Documentation index

## Start here

- [Getting started](GETTING_STARTED.md): CARLA installation, runtimes and external artifacts.
- [Code map](CODE_MAP.md): modules, entry points and data flow.
- [Results](RESULTS.md): current measurements and scopes.
- [Roadmap](NEXT_STEPS.md): planned perception, routing and behavior integration.
- [Security](../SECURITY.md): local services, trusted model loading and publication boundaries.

## Simulation and control

- [Simulation controls](../INDIAN_SIMULATION.md).
- [Connected town](COMBINED_INDIAN_TOWN.md) and [individual road worlds](INDIAN_WORLDS.md).
- [Selected scenario results](results/carla/SCENARIO_REPORT.md).
- [Decision display](DECISION_DISPLAY.md) and [playback controls](PLAYBACK_CONTROLS.md).
- [Recorded camera observation](CAMERA_OBSERVATION.md) and [live road preview](LIVE_SIMULATION_OBSERVATION.md).
- [Scene quality](SIMULATION_QUALITY.md).
- [Connected-town gridlock diagnosis](TOWN_GRIDLOCK_DIAGNOSIS_20261029.md): late-run junction lock in a 900 s run; diagnosed, not fixed.
- [Planning/control references](../integration/README.md).

## Road segmentation

- [Preprocessing](../PREPROCESSING.md) and [architecture/training](../MODEL_AND_TRAINING.md).
- [Model catalog](../models/README.md) and [perception metadata](../models/perception_models.json).
- [Real-image inference](../INFERENCE_GUIDE.md), [video](../REAL_VIDEO_GUIDE.md) and [stream measurements](../STREAMING_GUIDE.md).
- [Synthetic dataset](SIMULATION_DATASET.md), [training](SIMULATION_TRAINING.md), [test](SIMULATION_TEST.md) and [fresh confirmation](SIMULATION_CONFIRMATION.md).
- [Real/simulation model separation](REAL_MODEL_RETENTION.md).

## Object detection

- [Import/conversion](IDD_DETECTION_IMPORT.md).
- [Cleaning, object-size analysis and refinement](DETECTION_IMPROVEMENT.md).
- [Training protocol](DETECTION_TRAINING.md) and [initial result](DETECTION_TRAINING_REPORT.md).
- [Full resolution comparison](DETECTION_FULL_RESOLUTION_20261025.md).
- [Latest matched result](DETECTION_RESOLUTION_PILOT_RESULT.md).
- [Box-error illustrations](DETECTION_BOX_EXAMPLES.md).
- [Checkpoint recovery](DETECTION_RECOVERY.md).
- [Error review, 28 Oct](DETECTION_ERROR_REVIEW_20261028.md): where the detector misses; 1536 / NMS 0.6 setting test (validation).
- [Animal label review](ANIMAL_DATA_REVIEW_20261028.md) and [animal data plan](ANIMAL_DATA_PLAN_20261029.md): why animal labels were inconsistent; extra data sources.
- [Filtered animal view and land-only re-score](ANIMAL_FILTER_RESCORE_PLAN_20261029.md): what was built and the overnight training result (validation, not deployed).
- [Bicycle and traffic-light drop after v4](DETECTION_V4_BICYCLE_LIGHT_DROP_20261029.md): why two classes dipped slightly (validation, CPU-only analysis).

## Portable result records

| Topic | Record |
| --- | --- |
| Real-road segmentation | [segmentation.json](results/segmentation.json) |
| Detector data | [idd_detection_import.json](results/idd_detection_import.json) |
| Latest detector comparison | [detection_resolution_pilot.json](results/detection_resolution_pilot.json) |
| Synthetic dataset | [simulation_dataset.json](results/simulation_dataset.json) |
| Synthetic test | [simulation_test.json](results/simulation_test.json) |
| Fresh synthetic confirmation | [simulation_confirmation.json](results/simulation_confirmation.json) |
| Connected town | [combined_indian_town.json](results/combined_indian_town.json) |
| Live observation | [live_simulation_observation.json](results/live_simulation_observation.json) |

Preparation/intermediate snapshots describe their recorded stage; they do not supersede final matched results. The current aggregate measurements are collected in [RESULTS.md](RESULTS.md).

## External artifacts

Dataset images, checkpoint binaries, engine installations and complete run directories are not included. Technical reports may reference those relative local artifacts as provenance. A source-only clone contains selected summaries, not every gallery or an immediately provisioned model environment. Follow [getting started](GETTING_STARTED.md) before running tasks that depend on external files.
