# Indian Road Perception Simulator

Road segmentation, object detection and a native CARLA traffic environment for studying Indian-style road scenes and driving behavior.

The project combines a locked real-road segmenter, a separate synthetic-road segmenter, a 12-class detector and a connected left-hand-traffic town. Vehicles currently use a rule-based controller with simulator actor states. Camera predictions are available for observation; perception-driven control and destination navigation are planned.

## Features

- MobileNetV2/U-Net road segmentation with fixed preprocessing and checkpoint verification.
- YOLO11s detection of people, riders, vehicles, animals, traffic lights and traffic signs under a documented 12-class policy.
- Separate synthetic-road data/training with validity-masked labels and metrics.
- One OpenDRIVE town with market, colony, divided main road, rural road and irregular junction districts.
- Native vehicle route following, following gaps, junction rules, pedestrian handling and selected overtaking scenarios.
- Synchronized camera capture, matching-frame road overlays and stale/offline handling.
- Inspectable model metadata, curated result records and independent audit utilities.

## Results

| Component | Recorded result | Evaluation scope |
| --- | --- | --- |
| Locked real-road segmentation | 91.6549% road IoU / 95.6457% Dice | 689 real test images; checkpoint/protocol fixed before test |
| Simulation-road segmentation | 99.9288% road IoU | 500 synthetic test images; 98.2174% valid coverage |
| YOLO11s detection | 62.245% mAP50 / 41.525% mAP50-95 | 17,675 project validation images, 1280 input, GPU FP32 |
| Connected town | About 2.21 km across five districts | One 650-second seeded native check |

The metrics measure different tasks and datasets. Detector results use a project class mapping and filtered validation protocol, not the official full IDD benchmark. Its latest matched trial gains 0.557 mAP50 percentage points; animals remain weak and three class AP50 values decline slightly. Read [results and limitations](docs/RESULTS.md) for conditions and underlying records.

## Quick start

Use Windows, official 64-bit Python 3.12 with the py launcher, PowerShell and a compatible graphics setup for CARLA 0.9.16. Run from the repository root:

    git clone https://github.com/lmaoitsatharv/indian-road-perception-simulator.git
    cd indian-road-perception-simulator
    .\setup\project\bootstrap_carla.ps1 -DownloadCarla
    .\Start_Combined_Indian_Traffic.ps1 -SimulationSeconds 120 -Quality Low

The bootstrap creates a separate .venv-carla environment and downloads/extracts the engine. Allow at least 30 GB free for the archive and installation. The rule-driven scene needs no datasets, checkpoints or MATLAB.

In the combined-town viewer, Tab switches camera, Space pauses/resumes, S saves a screenshot and Esc closes the owned session. The minimap shows districts and actors. The original smaller scene is available through Start_Indian_Traffic.ps1. See [getting started](docs/GETTING_STARTED.md) for model prerequisites and [simulation controls](INDIAN_SIMULATION.md) for options.

## Source and model distribution

The repository contains source, clean notebooks, dependency specifications, model metadata, original small scene assets and curated results. Datasets, downloaded footage, trained weights, engine binaries, environments and complete experiment directories are external artifacts.

[Perception model metadata](models/perception_models.json) identifies recorded checkpoints, hashes and scopes. Inference also requires the relevant files and provenance; a source-only clone is not a ready-to-train model package. Keep TensorFlow, detector and CARLA runtimes separate because their package requirements differ.

## Documentation

- [Getting started](docs/GETTING_STARTED.md): installation, launchers and external artifacts.
- [Code map](docs/CODE_MAP.md): primary modules and data flow.
- [Results](docs/RESULTS.md): measured outcomes and limits.
- [Roadmap](docs/NEXT_STEPS.md): perception integration, routing and behavior evaluation.
- [Documentation index](docs/INDEX.md): technical guides by topic.
- [Security](SECURITY.md): local services, checkpoint trust and repository safeguards.

## Current limits

The native driver uses simulator ground truth, not camera-derived obstacle states. The live road preview is observation-only. Tracking, range integration, selected destinations and a learned STOP/SLOW/PROCEED policy are planned. Earlier neighbourhood overtaking checks do not establish combined-town passing performance.

Road masks do not establish obstacle-free space or lane legality. Detector traffic-light/sign categories do not supply light state or sign meaning. Generic assets, correlated synthetic layouts, close clearances, transient jerk and runtime interruptions limit the existing evidence. Finite zero-collision checks do not establish real-road safety.

## Team

- [Atharv](https://github.com/lmaoitsatharv)
- [Sarthak](https://github.com/Tumgharag)

## Contributing

See the [contribution guide](CONTRIBUTING.md) for the branch, review and evaluation workflow.

Read the code map and relevant evaluation protocol before changing a module. Preserve model/split contracts, use fresh experiment directories, distinguish prediction from control and report failed/regressing cases. Include evaluation scope with performance claims. Keep datasets, weights, credentials and generated runs outside source commits. The staged-content guard at setup/project/check_repository.py checks common publication mistakes; it is not an exhaustive security scan.
