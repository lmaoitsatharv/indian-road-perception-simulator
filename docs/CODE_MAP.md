# Code map

`project_runtime.py` resolves configurable WSL interpreter/distribution paths, standard Windows-drive mounts and optional MATLAB executables. It does not load models or start work on import. Runtime settings are described in [getting started](GETTING_STARTED.md).

This map identifies primary entry points and distinguishes training, prediction, simulator control and saved-evidence tooling. See [getting started](GETTING_STARTED.md) for prerequisites and [results](RESULTS.md) for measurements.

## Current data flow

    Real RGB -> locked real road model -> probabilities / road mask
    Labelled road-scene RGB -> YOLO11s -> boxes / classes / confidence
    CARLA RGB -> simulation road model -> live observation

    CARLA actor states -> rule driver -> throttle / brake / steering -> physics   (the other traffic)

    CARLA front RGB -> simulation road model -> road mask --+
    CARLA front RGB -> YOLO11s -> boxes --------------------+-> camera driver -> throttle / brake / steering / gear
    rear / side RGB (when needed) -> YOLO11s -> boxes ------+        ^
    speedometer, gyro -> dead reckoning ---------------------+        |
    picked destination -> sat-nav (lane graph) -> route guide, junction and U-turn cues
    learned speed policy (optional) -> a fraction of the shield's speed

The camera driver sees only its own camera pictures through the two models, its speedometer and gyro, and the sat-nav's route cues; it has no access to the simulator's actors. The sat-nav uses the car's exact position and the town's lane graph, as a phone's map would. [Camera driving](CAMERA_DRIVING.md) describes both.

## Road segmentation

| File | Purpose |
| --- | --- |
| [idd_data.py](../idd_data.py) | Manifest-based pairing, binary-target checks, paired augmentation and one RGB normalization. |
| [check_preprocessing.py](../check_preprocessing.py) | Verifies actual batches; requires external prepared data. |
| [prepare_idd_splits.py](../prepare_idd_splits.py) | Initial split construction, not a replacement for the preserved cleaned manifest. |
| [idd_model.py](../idd_model.py) | MobileNetV2 U-Net, BCE/soft-Dice objective and global road metrics. |
| [train_idd.py](../train_idd.py) | Explicit training to a fresh run with settings, history and checkpoints. |
| [idd_experiments.py](../idd_experiments.py) | Controlled augmentation, trainable-block and precision helpers. |
| [idd_boundary.py](../idd_boundary.py) | Boundary-aware objective/metric helpers. |
| [idd_inference.py](../idd_inference.py) | Locked checkpoint/protocol/hash/manifest validation and fixed-weight prediction. |
| [predict_road.py](../predict_road.py) | Image CLI exporting probabilities, masks, previews and transforms. |
| [predict_road_video.py](../predict_road_video.py) | Recorded-video inference with timestamp diagnostics. |
| [benchmark_road_stream.py](../benchmark_road_stream.py) | File-stream timing, separate from physical-camera/control latency. |

Clean notebooks provide explanatory code cells. Weights and full experiment output remain external.

## Object detection

| File | Purpose |
| --- | --- |
| [detection_policy.json](../detection_policy.json) | Fixed class order, source mapping and annotation/coordinate policy. |
| [idd_detection_data.py](../idd_detection_data.py) | JSON/XML parsing, image/box checks, YOLO conversion and preparation audit. |
| [Provider index](../setup/index_idd117k_detection.py) | Provider identities, source splits and recording groups. |
| [Recording curation](../setup/curate_idd_detection_manifest.py) | Excludes recording overlap without moving retained split labels. |
| [Data preparation](../setup/prepare_idd_detection.py) | Builds an audited detector view from a reviewed manifest. |
| [Quality filtering](../setup/finalize_idd_detection_quality.py) | Recorded invalid-pair and cross-split duplicate exclusions. |
| [Clean training view](../setup/prepare_clean_detection_view.py) | Excludes ambiguous repeated pixels and extra consistent duplicates, preserving validation. |
| [train_detection.py](../train_detection.py) | Readiness by default; explicit training after provenance/runtime checks. |
| [detection_refinement.py](../detection_refinement.py) | Rare-image sampling and FP32 model/EMA/optimizer recovery. |
| [Initial recipe](../detection_training_recipe.json) | Recorded 640-input settings/provenance. |
| [Clean-view recipe](../detection_refinement_clean_recipe.json) | Recorded 960-input parent settings. |
| [1280 recipe](configs/detection_pilot_1280.json) | Settings for the matched three-epoch adaptation. |
| [Error analysis](../setup/analyse_detection_errors.py) | Fixed-weight resolution/miss analysis on labelled validation. |

Download/import, independent label audits, runtime checks and launch/report helpers are under setup/. Their recipes require external verified parents, data and provenance.

## Synthetic-road learning

| File | Purpose |
| --- | --- |
| [Road observer](../integration/india_road_observer.py) | Synchronized native sensor evidence and camera metadata. |
| [carla_road_reference.py](../carla_road_reference.py) | Visible-road reference from tags, instances, depth and lane geometry, with ignored uncertainty. |
| [Episode preparation](../setup/prepare_simulation_dataset.py) | Capture plan and episode/split protocol. |
| [Dataset builder](../setup/build_simulation_dataset.py) | Selects/audits image/road/validity pairs from saved captures. |
| [simulation_data.py](../simulation_data.py) | Episode-consistent input loading and aligned transforms. |
| [simulation_model.py](../simulation_model.py) | Validity-masked loss/metrics; one road-probability output. |
| [train_simulation.py](../train_simulation.py) | Separate verified training protecting real-model contracts. |
| [evaluate_simulation.py](../evaluate_simulation.py) | Fixed-checkpoint test exports for independent recounting. |
| [Recorded simulation inference](../idd_simulation_inference.py) | Explicit recorded-simulator manifest/scope. |
| [Live simulation inference](../idd_live_simulation_inference.py) | Distinct native live-observation scope. |

Validity is target metadata, not a visual input feature. Synthetic and real evaluations remain separate.

## Native CARLA runtime

| File/group | Purpose |
| --- | --- |
| [Combined launcher](../Start_Combined_Indian_Traffic.ps1) | Main town wrapper; its .cmd invokes this script. |
| [Neighbourhood launcher](../Start_Indian_Traffic.ps1) | Smaller scene and selected traffic cases. |
| [Session owner](../setup/carla/run_session.py) | Bounded engine/viewer startup, result records and cleanup. |
| [Windows jobs](../setup/carla/windows_job.py) | Owns only process trees created for the session. |
| [Combined map](../integration/india_combined_map.py) | Nodes, streets, lane geometry and junction connectors. |
| [Combined scenery](../integration/india_combined_scenery.py) | District scenery in one native world. |
| [Combined traffic](../integration/india_combined_traffic.py) | Persistent routes, following/junction rules and native commands. |
| [Combined scene](../integration/india_combined_scene.py) | World, actors, viewer, minimap, sensors and stepping. |
| [Town HUD](../integration/india_town_hud.py) | Display-only overlay: district, clock, minimap, speed panel, key hints and help. |
| [PNG worker](../integration/india_png_worker.py) | Encodes evidence PNGs in a separate process so the viewer keeps 20 Hz. |
| [Neighbourhood scene](../integration/carla_india_scene.py) | Smaller scenario/capture loop. |
| [Behavior](../integration/india_behavior.py) | Baseline controller, pedestrian rules and earlier passing state machine. |
| [Decision display](../integration/india_decisions.py) | Frame-linked explanatory text; no actuator commands. |
| [Camera configuration](../integration/india_camera.py) | Shared camera appearance. |
| [Evidence writing](../integration/india_evidence_io.py) | Lossless capture writes and error propagation. |
| [Playback](../integration/india_playback.py) | Wall-clock pacing without altering the physics step. |
| [Individual worlds](../integration/india_world_maps.py) | Separate road-profile specifications. |

Surface, streetscape, weather and signal modules provide specific scene components. Small original textures and source-only autorickshaw artwork are under integration/indian_world_assets/. Its manifest distinguishes source artwork from a registered physics vehicle.

## Camera-only driving

| File | Purpose |
| --- | --- |
| [Launcher](../Start_Camera_Driving.ps1) | Wrapper (and `.cmd`); `-Destination`, `-RLSpeed`, weather and length options. |
| [Session owner](../setup/run_camera_driving.py) | Checks the two approved checkpoints, starts the WSL model workers, then the town with the camera car. |
| [Model workers](../setup/camera_driving_service.py) | Authenticated local service; [road](../setup/camera_driving_road_worker.py) and [detector](../setup/camera_driving_detector_worker.py) workers answer one picture at a time. |
| [Road inference](../idd_driving_inference.py) | Fixed-checkpoint simulation road model for driving. |
| [Perception client](../integration/india_perception_client.py) | Sends each front picture to both workers, plus one rear or side picture to the detector, and waits for the answers. |
| [Camera geometry](../integration/india_camera_geometry.py) | Pinhole projection and flat-ground distances for each mounted camera. |
| [Camera driver](../integration/india_camera_driver.py) | Lane keeping, obstacles and phantom rejection, passing, junctions, arrival, U-turns, rear/side camera checks and stuck recovery. No simulator access. |
| [Route planner](../integration/india_route_planner.py) | The town's lane graph, snapping a clicked point, Dijkstra routes with either-side pins and U-turns. |
| [Sat-nav](../integration/india_satnav.py) | Destination state, re-planning after wrong turns or resets, and the cues it gives the driver. |
| [Hybrid A*](../integration/india_hybrid_astar.py) | Junction paths on the camera's top-down road grid. |
| [Learned speed](../integration/india_rl_speed.py) | The shield, observation and numpy policy used by the driver. |
| [Training and analysis](../rl/) | Stand-in town, PPO training, route export and CARLA run scoring for the learned speed. |
| [Offline checks](../setup/verify_camera_driving.py) | Synthetic camera checks of driving, passing, the protocol and isolation; [destinations and U-turns](../setup/verify_camera_destinations.py) and [learned speed](../setup/verify_rl_speed.py). |

The town scene ([combined scene](../integration/india_combined_scene.py)) spawns the camera car's sensors, steps the driver at 20 Hz and records its decisions, views and contacts; its HUD shows what the AI sees.

## Live observation

| File | Purpose |
| --- | --- |
| [Live launcher](../Start_Live_Road_Preview.ps1) | Wrapper requiring separately provisioned model/runtime artifacts. |
| [Live session](../setup/run_live_simulation.py) | Coordinates a bounded observer session and optional matched/fault checks. |
| [GPU worker](../setup/live_simulation_worker.py) | Fixed-model prediction for one authenticated native camera session. |
| [Protocol](../simulation_live_protocol.py) | Bounded messages, identities, latest-only buffering and freshness. |
| [Preview UI](../integration/india_live_preview.py) | Matching-frame overlays and transport; no driver/actuator interface. |

Existing timings measure observation, not complete camera-to-command latency.

## Planning references and support files

The integration/matlab/ directory contains separate planning, tracking, traffic and replay references. Static Hybrid A*/Stanley/PI examples are not the active combined-town route planner. Read [the integration guide](../integration/README.md) for their assumptions.

- models/: model selection locks and checkpoint metadata, not weights.
- docs/configs/: inspectable recorded experiment settings.
- docs/results/: curated metrics, audit records and synthetic captures.
- requirements-*.txt: task-specific package contracts.
- setup/prepare_*, verify_*, report_* and finalize_*: task-specific preparation/check/report utilities; inspect each CLI and its external requirements.
- [Publication guard](../setup/project/check_repository.py): staged-content checks, not an exhaustive security scan.

Dataset, run, footage, installed engine and environment directories are not committed. Opening a saved JSON record does not execute its experiment.
