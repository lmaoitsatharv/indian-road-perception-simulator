# Simulation and planning integration

This directory contains the native CARLA traffic simulation and a separate set of MATLAB planning and tracking references. The current CARLA driver follows predefined routes using hand-coded rules and exact simulator actor states. Camera-model predictions are displayed for observation; they do not currently supply steering, throttle or brake commands.

Start with [Getting started](../docs/GETTING_STARTED.md), the [documentation index](../docs/INDEX.md) and the [results overview](../docs/RESULTS.md).

## Native CARLA simulation

The connected town contains market, residential, divided-road, rural and irregular-junction districts in one world. Procedural OpenDRIVE specifications define roads, lanes and junction connectors. Scenery uses installed generic meshes together with original project sign/shop textures. Vehicles receive native control commands and move through CARLA physics rather than pose teleports.

The documented connected-town configuration has seven junction nodes, nine streets, 36 directed lane connectors, 25 vehicles and 21 pedestrians. A selected 650-second run kept one world and persistent vehicle identities while the ego vehicle travelled about 2.21 km across all five districts. These are finite synthetic-scene results, not a broad traffic or real-road safety guarantee. See [the connected-town report](../docs/COMBINED_INDIAN_TOWN.md).

From the repository root, after installing the required runtime and CARLA engine:

```powershell
.\Start_Combined_Indian_Traffic.ps1
```

The [simulation guide](../INDIAN_SIMULATION.md) describes the launchers, camera controls, pause, capture and exit behavior. The earlier smaller neighbourhood is a separate scene used for the selected overtaking and observation checks.

## Main modules

| Module | Responsibility |
| --- | --- |
| [india_combined_map.py](india_combined_map.py) | Connected OpenDRIVE roads, legal lane directions and junction links |
| [india_combined_scenery.py](india_combined_scenery.py) | District-specific scenery in the connected world |
| [india_combined_traffic.py](india_combined_traffic.py) | Persistent actor routes, junction handling and native control commands |
| [india_combined_scene.py](india_combined_scene.py) | World setup, viewer, sensors, fixed-step episode loop and cleanup |
| [india_behavior.py](india_behavior.py) | Earlier-neighbourhood rule driver, following, overtaking and return-gap states |
| [carla_india_scene.py](carla_india_scene.py) | Earlier-neighbourhood scene and its capture/observation integration |
| [india_road_observer.py](india_road_observer.py) | Frame-aligned RGB, semantic, instance and depth evidence |
| [india_live_preview.py](india_live_preview.py) | Bounded camera transport, matching-frame display and stale/offline states |
| [india_evidence_io.py](india_evidence_io.py) | Lossless evidence writes with error propagation |
| [india_playback.py](india_playback.py) | Wall-clock playback pacing without changing the fixed physics step |

The launcher/session support in [setup/carla](../setup/carla) owns the processes and actors it creates. Runtime records are written to local `runs/` directories, which are not included in the source distribution.

## Perception boundaries

The project has three distinct model roles:

- A locked real-road U-Net/MobileNetV2 segmenter, evaluated on 689 real test images: **91.6549% road IoU / 95.6457% Dice**.
- A separate simulation-road segmenter, evaluated on 500 synthetic test images: **99.9288% road IoU**, with **98.2174% valid-pixel coverage**.
- A separate 12-class YOLO11s detector, evaluated on 17,675 project validation images at 1280 input and GPU FP32: **62.245% mAP50 / 41.525% mAP50-95**.

These measure different tasks and datasets. The synthetic score is not evidence of real-road transfer, and detector AP is not a percentage of safe driving. Model binaries, datasets and full experiment output must be obtained separately and verified against the [model metadata](../models/README.md).

The live road preview is available through `Start_Live_Road_Preview.cmd` after its runtime, weights and provenance prerequisites are supplied. It uses the smaller neighbourhood, displays the simulation model's mask on its matching source image and keeps the rule driver separate. The [live observation report](../docs/LIVE_SIMULATION_OBSERVATION.md) records inference/frame-age measurements, delay/disconnect tests and a measured redraw expiry limitation. No live object-tracking or camera-based driving result is implied by the road overlay.

Synthetic road labels use synchronized native sensor data and OpenDRIVE geometry. Uncertain pixels are excluded through a validity mask. A road mask is not a calibrated metric obstacle map or a declaration that every road pixel is free to drive through. See [the simulation dataset](../docs/SIMULATION_DATASET.md) and [camera observation protocol](../docs/CAMERA_OBSERVATION.md).

## Driving and overtaking scope

The current rules handle route following, speed/following gaps, junction reservations and pedestrian constraints. The earlier neighbourhood's overtaking logic considers slower leaders, approaching traffic, pedestrians and lateral body clearance. It checks the current return gap when the vehicle is ready to merge back; beginning a pass does not guarantee a future gap.

Selected clear-pass, blocked-oncoming, pedestrian, return-gap and junction cases have saved audits in [the scenario report](../docs/results/carla/SCENARIO_REPORT.md). Those checks do not automatically establish overtaking performance in the newer connected town, whose route and traffic implementation differs. Tight clearance, transient jerk and broader scenario coverage remain development concerns.

## Separate MATLAB references

The [matlab](matlab) directory contains static Hybrid A* planning, controller tracking, mask-reader and replay references. The static tracker uses Stanley steering, bounded speed feedback/feedforward and a kinematic bicycle model. These examples have their own software dependencies; they are not invoked as the connected CARLA town's live planner.

From the repository root in MATLAB, use fresh output directories:

```matlab
projectRoot = pwd;
addpath(fullfile(projectRoot, 'integration', 'matlab'));
referenceOutput = fullfile(projectRoot, 'runs', 'static_reference_new');
trackingOutput = fullfile(projectRoot, 'runs', 'static_tracking_new');
run_static_reference(referenceOutput);
run_static_tracking_reference(referenceOutput, trackingOutput);
```

The source-only repository does not contain the full saved run folders or their generated replay files. Replaying stored poses or controller traces is different from performing live perception-driven control. An image mask also requires calibrated projection, clock/frame conventions, obstacle observations and uncertainty handling before metric planning can use it.

## Planned integration

The remaining work includes live object detection/tracking, range and motion checks, perception-derived STOP/SLOW/PROCEED decisions, destination selection and routing, and overtaking refinement in the connected town. A future bounded reinforcement-learning policy may propose target speed above route control, with independent collision/emergency checks retained. No RL policy is currently trained.

Development results and limitations are collected in [Results](../docs/RESULTS.md); implementation priorities are in [Next steps](../docs/NEXT_STEPS.md).
