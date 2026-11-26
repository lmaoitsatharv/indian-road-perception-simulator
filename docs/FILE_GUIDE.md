# File-by-file guide

Use this index when asked what a particular repository file does. Begin with the [beginner guide](MILESTONE_2_VIVA_GUIDE.md) for terminology and the [progress report](MILESTONE_2_PROGRESS.md) for current status.

Descriptions summarize source/configuration purpose; they are not a claim that every script was rerun during this handoff. Key Python entry points are extracted from the actual source. Historical experiment/report utilities are retained for reproducibility rather than deleted merely because they are not the primary launcher.

For any file, explain **input -> operation -> output**, then say whether it trains, predicts, controls actors, checks saved evidence or only documents a result. A `verify_`/`report_` helper is not another neural-network architecture. Read historical settings as historical, not as live status.

This edition indexes **468 published files**. Local datasets, checkpoints, engine installs, run folders, virtual environments, generated study PDFs and the creative archive are deliberately outside this index.

## Core and root files

### [.gitattributes](../.gitattributes)

Defines text line-ending behavior and binary image/document handling for Git.

### [.gitignore](../.gitignore)

Excludes large local data, weights, environments, credentials, runtime output and unrelated archived projects from publication.

### [CARLA_GUIDE.md](../CARLA_GUIDE.md)

Documentation: CARLA setup and first live scene. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [IDD_Preprocessing.ipynb](../IDD_Preprocessing.ipynb)

Clean educational notebook for IDD Preprocessing. Code/output contracts are explained in the main guide; execution outputs are not committed.

### [IDD_UNet_Model.ipynb](../IDD_UNet_Model.ipynb)

Clean educational notebook for IDD UNet Model. Code/output contracts are explained in the main guide; execution outputs are not committed.

### [INDIAN_SIMULATION.md](../INDIAN_SIMULATION.md)

Documentation: Indian-style traffic in CARLA. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md)

Documentation: Using the locked road model. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [MODEL_AND_TRAINING.md](../MODEL_AND_TRAINING.md)

Documentation: U-Net model: what is implemented and what comes next. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [Open_training_dashboard.cmd](../Open_training_dashboard.cmd)

Owner-local convenience shortcut to the read-only dashboard for the latest saved detector run; a fresh clone does not contain that run.

### [PREPROCESSING.md](../PREPROCESSING.md)

Documentation: IDD preprocessing. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [README.md](../README.md)

Documentation: Indian road perception and traffic simulation. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [REAL_VIDEO_GUIDE.md](../REAL_VIDEO_GUIDE.md)

Documentation: Real driving video: play, inspect and repeat. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [SECURITY.md](../SECURITY.md)

Documentation: Security and sharing. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [STREAMING_GUIDE.md](../STREAMING_GUIDE.md)

Documentation: Lean video streaming and human-reviewed scoring. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [Start_CARLA.cmd](../Start_CARLA.cmd)

Launch wrapper for CARLA. Read its parameters and target script; startup does not imply camera-based control.

### [Start_CARLA.ps1](../Start_CARLA.ps1)

Launch wrapper for CARLA. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Combined_Indian_Traffic.cmd](../Start_Combined_Indian_Traffic.cmd)

Launch wrapper for Combined Indian Traffic. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Combined_Indian_Traffic.ps1](../Start_Combined_Indian_Traffic.ps1)

Launch wrapper for Combined Indian Traffic. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Indian_Traffic.cmd](../Start_Indian_Traffic.cmd)

Launch wrapper for Indian Traffic. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Indian_Traffic.ps1](../Start_Indian_Traffic.ps1)

Launch wrapper for Indian Traffic. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Indian_Worlds.cmd](../Start_Indian_Worlds.cmd)

Launch wrapper for Indian Worlds. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Indian_Worlds.ps1](../Start_Indian_Worlds.ps1)

Launch wrapper for Indian Worlds. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Camera_Driving.cmd](../Start_Camera_Driving.cmd)

Launch wrapper for camera-only driving: the road model and detector drive your car in the connected town. See [CAMERA_DRIVING.md](CAMERA_DRIVING.md).

### [Start_Camera_Driving.ps1](../Start_Camera_Driving.ps1)

Launch wrapper for camera-only driving (`-SimulationSeconds`, `-Weather`, `-NoReset`, `-Check`). Runs `setup/run_camera_driving.py`.

### [Start_Live_Road_Preview.cmd](../Start_Live_Road_Preview.cmd)

Launch wrapper for Live Road Preview. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Live_Road_Preview.ps1](../Start_Live_Road_Preview.ps1)

Launch wrapper for Live Road Preview. Read its parameters and target script; startup does not imply camera-based control.

### [Start_Separate_Indian_Worlds.cmd](../Start_Separate_Indian_Worlds.cmd)

Launch wrapper for Separate Indian Worlds. Read its parameters and target script; startup does not imply camera-based control.

### [WSL_SETUP.md](../WSL_SETUP.md)

Documentation: RTX 4060 setup: resume after restart. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [benchmark_idd_training.py](../benchmark_idd_training.py)

Disposable GPU capacity benchmark; uses train images, never saves model updates.

Key entry points: `digest`, `main`.

### [benchmark_road_stream.py](../benchmark_road_stream.py)

Bounded file-stream benchmark, not a physical-camera or driving controller.

Key entry points: `StreamFrame`, `run_stream`, `compare_saved_masks`, `save_rows`, `main`.

### [carla_road_reference.py](../carla_road_reference.py)

Constructs a visible-road reference from synchronized semantics/instances/depth and OpenDRIVE geometry, retaining uncertain pixels as ignored.

Key entry points: `checked_appearance`, `build_reference`.

### [check_preprocessing.py](../check_preprocessing.py)

Check train/validation preprocessing in a new folder; test files stay unread.

Key entry points: `save_preview`, `main`.

### [check_wsl_gpu.py](../check_wsl_gpu.py)

Verify WSL GPU computation and one disposable project training batch.

Key entry points: `digest`, `main`.

### [detection_policy.json](../detection_policy.json)

Authoritative 12-class order, source-label mapping, coordinate/unknown-label policy and observation-only detector scope.

### [detection_refinement.py](../detection_refinement.py)

Adds training-only rare-image sampling, duplicate-pixel probability correction and full-precision detector recovery checkpoints to the pinned framework.

Key entry points: `parallel_dataset_hash`, `save_full_precision_resume`, `sampling_weights`, `make_refinement_trainer`.

### [detection_refinement_clean_recipe.json](../detection_refinement_clean_recipe.json)

Recorded clean-v3 960-input, 30-epoch parent settings including rare-class sampling and readiness/provenance references.

### [detection_refinement_recipe.json](../detection_refinement_recipe.json)

Earlier v2-based refinement settings retained as experiment history; use the clean-view recipe to understand the later parent.

### [detection_training_recipe.json](../detection_training_recipe.json)

Historical 640-input, 20-epoch first detector recipe; not the latest 1280 pilot.

### [evaluate_carla_road.py](../evaluate_carla_road.py)

Score a road model on verified paired CARLA frames; never controls actors.

Key entry points: `load`, `reference`, `confusion`, `metrics`, `safe_source`, `inspect_capture`, `panel` (plus supporting helpers).

### [evaluate_simulation.py](../evaluate_simulation.py)

Evaluates one previously fixed simulator checkpoint on its reserved test split and exports predictions/counts for independent recounting.

Key entry points: `scores`, `weight_hash`, `evaluate`.

### [idd_boundary.py](../idd_boundary.py)

Training-only boundary emphasis and an explicitly defined outline metric.

Key entry points: `inner_boundary`, `BoundaryWeightedBCEDice`, `BoundaryIoU`.

### [idd_data.py](../idd_data.py)

Reads the cleaned road split manifest, pairs images/masks, checks 512x512 binary targets, applies paired flips and one MobileNetV2 normalization, and yields finite batches. It does not train.

Key entry points: `read_pairs`, `decode_pair`, `augment_pair`, `normalize_pair`, `make_dataset`, `build_datasets`.

### [idd_detection_data.py](../idd_detection_data.py)

Parses reviewed JSON/XML annotations and class/coordinate contracts, verifies images/boxes, converts normalized YOLO labels and records dataset audits/exclusions.

Key entry points: `AnnotationError`, `file_sha`, `save_json`, `load_policy`, `normal_name`, `annotation_objects`, `convert_annotation` (plus supporting helpers).

### [idd_experiments.py](../idd_experiments.py)

Optional overnight experiment helpers; originals and the established loader stay intact.

Key entry points: `light_augment`, `_light_augment_cpu`, `experiment_dataset`, `mixed_model_from_checkpoint`.

### [idd_inference.py](../idd_inference.py)

Loads only the locked real-road model for inference, validates its hash/protocol, and keeps raw RGB and normalized inputs distinct. It does not update weights.

Key entry points: `file_sha256`, `write_json_atomic`, `load_final_lock`, `require_raw_rgb`, `normalize_raw_rgb`, `require_normalized_rgb`, `read_rgb` (plus supporting helpers).

### [idd_live_simulation_inference.py](../idd_live_simulation_inference.py)

Validates the distinct native-live-observer scope for the simulator road model. Observation permission is separate from vehicle control.

Key entry points: `load_live_spec`, `LiveSimulationRoadSegmenter`.

### [idd_driving_inference.py](../idd_driving_inference.py)

Loads the simulation road model under its own camera-driving scope (`simulator_camera_driving_v1`, `control_allowed: true`), after checking the checkpoint, its completed confirmation and the locked real-model anchor. The observation-only loader above is unchanged. Never trains; not approved for real images.

Key entry points: `load_driving_spec`, `DrivingRoadSegmenter`.

### [idd_model.py](../idd_model.py)

Defines the MobileNetV2 U-Net, BCE-plus-soft-Dice loss, globally counted road metrics and compile/fine-tuning helpers. Importing it does not start training.

Key entry points: `get_encoder_weights`, `build_unet`, `BCEDiceLoss`, `_RoadOverlap`, `RoadIoU`, `RoadDice`, `compile_model` (plus supporting helpers).

### [idd_simulation_inference.py](../idd_simulation_inference.py)

Validates an explicit recorded-simulator-only manifest before loading a separate road checkpoint. It cannot silently replace real inference.

Key entry points: `load_simulation_model_spec`, `SimulationRoadSegmenter`.

### [keep_overnight_awake.ps1](../keep_overnight_awake.ps1)

Temporary keep-awake helper for an owned local job; it manages sleep requests, not model weights or accuracy.

### [keep_training_awake.ps1](../keep_training_awake.ps1)

Temporary keep-awake helper for an owned local job; it manages sleep requests, not model weights or accuracy.

### [overnight_idd_continuation.py](../overnight_idd_continuation.py)

Continue only new controlled trials after the completed first overnight queue.

Key entry points: `main`.

### [overnight_idd_controller.py](../overnight_idd_controller.py)

Bounded overnight queue, independent watchdog, and continuously saved report.

Key entry points: `read`, `write`, `stop_reason`, `owned`, `terminate_owned`, `report`, `guard` (plus supporting helpers).

### [overnight_idd_worker.py](../overnight_idd_worker.py)

One bounded overnight trial; retains complete validated checkpoints atomically.

Key entry points: `sha`, `write_json`, `atomic_model`, `BoundedCheckpoint`, `smoke`, `train`.

### [plot_idd_training.py](../plot_idd_training.py)

Create training curves and validation contact sheets from saved run outputs.

Key entry points: `plot_run`.

### [predict_road.py](../predict_road.py)

Predict road regions with the locked model; never train or overwrite a run.

Key entry points: `main`.

### [predict_road_video.py](../predict_road_video.py)

Offline real-video diagnostics for the unchanged, locked road segmenter.

Key entry points: `utc_now`, `inspect_video`, `resize_input`, `mask_change`, `display_pair`, `make_display`, `DisplayWriter` (plus supporting helpers).

### [prepare_idd_splits.py](../prepare_idd_splits.py)

Historical initial road split creator: checks numeric pairs and hashes, groups exact-image matches and copies a deterministic 80/10/10 split. Do not rerun over the cleaned split.

Key entry points: `inventory`, `sha256`, `duplicate_stats`, `main`.

### [requirements-camera-observation.txt](../requirements-camera-observation.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [requirements-detection-wsl.txt](../requirements-detection-wsl.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [requirements-inference.txt](../requirements-inference.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [requirements-preprocessing.txt](../requirements-preprocessing.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [requirements-training-review.txt](../requirements-training-review.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [requirements-wsl.txt](../requirements-wsl.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [review_idd_validation.py](../review_idd_validation.py)

Reload the best checkpoint and review validation only; never open test images.

Key entry points: `sha256`, `resolve_recorded_checkpoint`, `verify_initial_comparison`, `review`.

### [road_video_labels.py](../road_video_labels.py)

Prepare blind real-video labels and score only reviewed human annotations.

Key entry points: `render_label_editor`, `decode_rle`, `counts`, `metrics`, `validate_annotation`, `create_pack`, `score_pack` (plus supporting helpers).

### [run_carla_road_evaluation.ps1](../run_carla_road_evaluation.ps1)

Environment/session helper for run carla road evaluation. Review machine/runtime prerequisites before use.

### [run_frozen_training.sh](../run_frozen_training.sh)

Environment/session helper for run frozen training. Review machine/runtime prerequisites before use.

### [run_road_prediction.ps1](../run_road_prediction.ps1)

Environment/session helper for run road prediction. Review machine/runtime prerequisites before use.

### [run_road_stream.ps1](../run_road_stream.ps1)

Environment/session helper for run road stream. Review machine/runtime prerequisites before use.

### [run_road_video_prediction.ps1](../run_road_video_prediction.ps1)

Environment/session helper for run road video prediction. Review machine/runtime prerequisites before use.

### [setup_wsl_linux.sh](../setup_wsl_linux.sh)

Environment/session helper for setup wsl linux. Review machine/runtime prerequisites before use.

### [setup_wsl_windows.ps1](../setup_wsl_windows.ps1)

Environment/session helper for setup wsl windows. Review machine/runtime prerequisites before use.

### [show_gpu_activity.py](../show_gpu_activity.py)

Bounded GPU-only demonstration. Does not read data or train/save a model.

Key entry points: `multiply`.

### [simulation_data.py](../simulation_data.py)

Loads synthetic RGB, road and valid-pixel targets; preserves episode split identity and aligns augmentation. Validity is metadata, not an input feature.

Key entry points: `read_pairs`, `decode_pair`, `make_dataset`.

### [simulation_live_protocol.py](../simulation_live_protocol.py)

Defines bounded packet framing, latest-only buffering and frame/session freshness rules used by the live observer.

Key entry points: `receive_exact`, `send_packet`, `receive_packet`, `LatestSlot`, `freshness`.

### [simulation_model.py](../simulation_model.py)

Defines validity-masked BCE/Dice and road metrics so ignored synthetic pixels cannot affect loss/score. The visual network still outputs one road channel.

Key entry points: `MaskedBCEDiceLoss`, `_weights`, `MaskedRoadIoU`, `MaskedRoadDice`, `pack_targets`, `compile_simulation_model`.

### [start_jupyter_wsl.ps1](../start_jupyter_wsl.ps1)

Environment/session helper for start jupyter wsl. Review machine/runtime prerequisites before use.

### [train_detection.py](../train_detection.py)

Defaults to readiness inspection. An explicit start flag trains the separate YOLO detector after provenance/data/runtime checks, with progress and preserved failures.

Key entry points: `load`, `check_prepared_files`, `readiness`, `train`.

### [train_idd.py](../train_idd.py)

Explicit real-road training entry point. Builds/loads the network, chooses frozen/fine phase, saves validation-selected checkpoints and history in a fresh run.

Key entry points: `sha256`, `TrainingProgress`, `train`.

### [train_simulation.py](../train_simulation.py)

Starts separate synthetic-road training only after verification, protecting real model files and avoiding reserved test image contents during learning.

Key entry points: `source_hashes`, `utc`, `configure_gpu`, `dataset_checks`, `Progress`, `train`.

### [validate_idd_dataset.py](../validate_idd_dataset.py)

Audit the split PNG pairs and create display-only mask previews.

Key entry points: `main`.

### [verify_unet.py](../verify_unet.py)

Verify a real U-Net forward/backward pass without applying optimizer updates.

Key entry points: `sha256`, `weight_fingerprint`, `verify_metrics_and_loss`, `verify_model`.

## Runtime Python integration

### [integration/carla_camera_probe.py](../integration/carla_camera_probe.py)

Small hardware diagnosis: RGB camera without Traffic Manager or pedestrians.

### [integration/carla_india_scene.py](../integration/carla_india_scene.py)

Native CARLA LHT neighbourhood: physics-controlled traffic and auditable episodes.

Key entry points: `road_markings`, `destroy_episode`, `episode`, `main`.

### [integration/carla_starter.py](../integration/carla_starter.py)

Live CARLA installation/demo check. Stock traffic rules, not the Indian scenario or learned driving.

Key entry points: `write_json`, `main`.

### [integration/india_behavior.py](../integration/india_behavior.py)

Observable, seeded rule-based traffic on the small LHT junction.

Key entry points: `wrap`, `rectangle`, `body`, `polygon_gap`, `Route`, `Driver`, `Pedestrian` (plus supporting helpers).

### [integration/india_camera.py](../integration/india_camera.py)

Shared RGB post-processing for the viewer and recorded model camera.

Key entry points: `configure_rgb`.

### [integration/india_combined_map.py](../integration/india_combined_map.py)

One connected Indian LHT town with five districts and two extra loops.

Key entry points: `Street`, `native_junction`, `adjacent`, `road_between`, `incoming_lane`, `outgoing_lane`, `lane_offset` (plus supporting helpers).

### [integration/india_combined_scene.py](../integration/india_combined_scene.py)

One live Indian town: every district and all traffic share one native world.

Key entry points: `minimap`, `EvidenceWriter`, `main`.

If CARLA drops a pedestrian from its registry, the run continues. The walker is listed under `lost_walkers` with its last position, and the run is marked not passed. A rare physics glitch can fling a waiting jaywalker; `actor_trace.jsonl` now logs every jaywalker position each tick so this can be traced.

The viewer shows two 1280x720 display cameras (chase and driver) that render every tick, so the picture updates at 20 Hz. The 5 Hz evidence cameras are unchanged. Evidence PNGs are encoded in worker processes, because pygame holds the GIL for a whole PNG encode and froze the window once a second. `result.json` records `display_loop_ms` (measured only while running at 1× speed) and `server_tick_ms`.

Viewer keys:
- TAB: chase or driver camera.
- 1–5 or L: a street camera in each district.
- V and Shift+V: follow other vehicles.
- C: back to your car.
- + and − and 0: speed ½×, 1×, 2× or max, or back to real time. Max reached about 1.8× on this laptop; the HUD clock shows the actual rate.
- SPACE: pause, with `.` stepping one frame.
- R: reset your car. It is put back stopped on its lane centre, moved ahead along its route to a spot clear of junctions, vehicles and people. This is a live pose teleport: `result.json` then shows `vehicle_pose_teleport: true` and counts `ego_resets`.
- W: weather.
- H: hide the HUD.
- F: fullscreen.
- S: snapshot.
- F1: all keys.

Street and follow cameras are extra sensors spawned on demand. No vehicle is moved, except by R. Speed, weather and camera changes are listed in `result.json` under `viewer_events`, and each saved camera set records the current weather.

### [integration/india_town_hud.py](../integration/india_town_hud.py)

Display-only HUD for the town viewer: district title, clock and real-time factor, speed and weather, minimap with ego heading and trail (plus the followed vehicle or street-camera spot), speed and driving-state panel, areas visited, key hints, a full key help overlay (F1), toasts, pause overlay. Nothing here is recorded as model input.

Key entry points: `TownHud`.

### [integration/india_camera_geometry.py](../integration/india_camera_geometry.py)

Fixed camera model on flat ground (CARLA axes: x forward, y right): ground point to pixel and back, and a 0.1 m bird's-eye grid with a fixed lookup into the 512x512 road mask. The camera can face any way (`yaw_deg=180` is the rear camera). Pure NumPy.

Key entry points: `CameraGeometry`, `GroundGrid`.

### [integration/india_rl_speed.py](../integration/india_rl_speed.py)

The learned speed's shared parts (12 Nov): the shield's posted-limit profiles, the limit hold and the gear band, the policy's observation, and the trained policy run with numpy (`SpeedPolicy`, `SpeedGovernor`). With `DriverSettings.rl_speed = 1` the camera driver asks it how far below the shield to drive. See [RL_SPEED_DESIGN_20261112.md](RL_SPEED_DESIGN_20261112.md).

### [integration/india_camera_driver.py](../integration/india_camera_driver.py)

The camera-only driver. It receives only its own camera's road mask and detector boxes, its speedometer and gyro, and its own size. It follows the left kerb 1.5 m to the right of the road model's left edge, so it takes the leftmost exit at junctions. It slows in wide road areas and places detected objects on the ground to PROCEED, SLOW or STOP. Steering is pure pursuit and the pedals use the rule driver's speed controller.

Two cross-checks correct the models:

- **Paved shoulder:** where the road model counts a paved shoulder as road, the left edge moves in to the bright-to-dark step where the asphalt starts in the picture.
- **Objects at the picture's side:** these are placed beyond the road the road model sees at their distance.

It can pass a stopped bus or truck on its right:

- It starts only when the passing lane is clear ahead (detector) and behind (a rear camera that runs only around passes), and nothing was just seen coming up in its blind spot.
- While passing it ignores only the vehicle it is passing, matched by where that vehicle appears in the picture.
- If anything else stops it mid-pass for 3 s, it gives up and backs out the way it came until it is straight again, stopping for anything the rear camera sees.

It has no simulator access (checked by `setup/verify_camera_driving.py`).

Key entry points: `CameraDriver` (`captured`, `observe`, `control`, `reset`), `DriverSettings`, `Perception`, `Plan`.

### [integration/india_perception_client.py](../integration/india_perception_client.py)

Sends each camera frame to the road and detector workers at once over loopback, with a session token and pixel hash, and waits for both answers. Around passes the rear camera's picture goes to the detector as a second request; a missing rear answer leaves the front result usable with no rear view. A worker that is down or late makes that frame unavailable; it reconnects at most once a second.

Key entry points: `PerceptionClient` (`submit`, `result`, `summary`), `WorkerLink`.

### [integration/india_route_planner.py](../integration/india_route_planner.py)

The camera car's route planner. It builds a lane graph of the connected town from its network file and the map's lane centre lines (passed in, so it never imports CARLA). It snaps a clicked point to the nearest street lane on the clicked side and plans the shortest route, keeping to the kerb-side lane; a destination on a two-way street may be reached on either side, and a route may turn round (a U-turn) on a two-way street clear of its junctions. The 11 named places are the 7 junctions and 4 district spots. Its `Route` serves the town's junction reservations the way the other vehicles' routes do.

Key entry points: `TownRoutes` (`plan`, `snap`, `locate`, `places`, `connector_turn`), `Route`, `LanePoint`.

### [integration/india_satnav.py](../integration/india_satnav.py)

The sat-nav for the camera car, run in the viewer with the car's exact position. It holds the destination and route and, every control step, gives the driver one message:

- "lane" or "junction"
- the route line for the next 30 m in the car's own frame
- where to wait for its turn
- how far the destination is

It re-plans after a wrong turn or a reset and records every event.

Key entry points: `SatNav` (`set_destination`, `update`, `clear`, `hud`), `parse_destination`.

### [integration/india_hybrid_astar.py](../integration/india_hybrid_astar.py)

Hybrid A* for junction turns on the camera driver's top-down road grid, NumPy only:

- **Map:** cells mostly not-road by the road model, and the ground under detected objects, are blocked and widened by the car's body (seven discs along its length).
- **Search:** forward arcs from the car to a goal on the route, with a soft pull towards the route's lane.
- **Limits:** 150 ms or 4,000 expansions.

It has no simulator access (checked by `setup/verify_camera_driving.py`).

Key entry points: `occupancy`, `plan`, `Occupancy`, `PlanResult`.

### [integration/india_png_worker.py](../integration/india_png_worker.py)

PNG encode and SHA-256 for recorded town frames, run in worker processes. Same pixel path as `india_world_scene.save_native`.

Key entry points: `encode_bgra`, `encode_rgb`.

### [integration/india_combined_scenery.py](../integration/india_combined_scenery.py)

All five scenery districts alive together in the connected native town.

Key entry points: `CombinedScenery`. Its main methods:

- `greenery` adds optional background palm and tree rows, kept at least 6 m clear of any driving lane.
- `buildings` lays mixed rows of buildings.
- `shopfronts` adds shop boards and queues window air-conditioners.
- `mount_acs` and `mount_boards` run after alignment. Horizontal ray casts find a flat building wall; an AC or board is fixed flush to it, or dropped when no wall is there. Spawned meshes carry no semantic tag, so a hit counts only inside a building's oriented box. `scenery.json` reports the counts under `window_acs` and `shop_boards`.
- `power_line` places electric poles.
- `street_life` fills each district's forecourt.
- `measure` reads a mesh's native size once, so each mesh can be scaled to a real height.

Buildings form near-continuous rows of mixed CARLA meshes, chosen per district:

- **Bazaar and main road:** brick and painted apartments and terraced shop-houses, 7–10 m tall.
- **Colony and junction:** larger houses.
- **Village roads:** sparse low houses.

Each mesh keeps 85–110% of its native size, aiming at the district's target height, so doors and windows stay person-sized. Shrinking houses to about half size had made them look half buried. Meshes wider than 18 m at that scale are not used. Buildings stand behind a forecourt strip 2.6–3.4 m deep, beyond the footpath or paved shoulder. There are no free-standing ATMs, advert panels or kennels. They read as foreign objects in an Indian street. The forecourt holds district-specific items: food carts, chai tables with people, crates, barrels, garbage, plant pots, hay, roadwork and standing people (at most 18). The footpath and the shoulder stay clear for walkers and two-wheelers, and the bus stop, signs and jaywalker crossing spots are kept free.

`align_and_check` first moves a building back a metre at a time if a bend brings it near a lane, and drops it if 6 m is not enough. It then respawns each building at its aligned pose, keeping the ground-level spawn height. A mesh's box can reach below its floor or start above it, so setting the height from the box made some houses float and others sink. in CARLA 0.9.16, `set_transform` on a spawned `static.prop.mesh` changes the actor's reported pose but not the rendered mesh or its collision. Before this fix, buildings stood 4–9 m behind their intended place. The base check still fails the build if anything touches a driving lane.

Not included, because CARLA ships no suitable asset:

- **Overhead wires:** `SM_wire` is a 4 cm piece and meshes only scale uniformly.
- **Riderless parked scooters:** every CARLA two-wheeler has a rider.
- **Awnings:** they rendered as floating tents.
- **Autos and cows.**

### [integration/india_combined_traffic.py](../integration/india_combined_traffic.py)

Persistent traffic across every district of one native connected town.

Key entry points: `LoopPath`, `TownDriver`, `UrbanBusPath`, `CombinedTraffic`, `JayWalker`, `FootpathWalker`.

There are 31 vehicles, about 60% of them two-wheelers. `FootpathWalker` pedestrians walk 45–85 m routes along the footpaths, or along the yard edge on village roads, and pause now and then.

Looser, more Indian traffic:

- **Shared junctions.** Up to 3 vehicles hold one junction at once if they follow each other on the same movement, or if their paths through it never come within 3.6 m where they meet at an angle (`_junction_conflicts`). Lanes running side by side, such as a bend's two directions, conflict only within 2.4 m, a vehicle's width and room. Grants are first-in, first-out with head-of-line blocking, so side streets are not starved.
- **Two-wheelers keep left.** On village roads, bicycles ride the paved shoulder. When stuck behind slow or queued traffic, motorcycles filter past it on the shoulder, or on the lane line of the divided road (`_plan_offsets`).
- **Cars squeeze past two-wheelers.** A car may pass a two-wheeler keeping left by moving up to 0.15 m over the centre line, but only with no oncoming vehicle within 55 m. On the divided road it stays inside its lane.
- **Everyone recentres near junctions.** All vehicles return to the lane centre within 28 m of a junction. A sideways move is never made into a body alongside.
- **Lane changes on the divided road.** Vehicles stuck behind a stopped or crawling vehicle, such as the bus at its stop, change into the other same-direction lane when it is clear 20 m behind and 14 m ahead. They never do this to jump a junction queue.
- **Head-on guard.** A head-on check stops a vehicle facing oncoming traffic too close in its path, but only while one of the two is off its lane centre. Applied to centred traffic, it once locked junction B.
- **Jaywalkers.** Pedestrians cross mid-block when the gap looks acceptable, and every vehicle yields to them.

The departure check accepts the shoulder for two-wheelers. `result.json` reports these under `traffic_behaviour`.

### [integration/india_conditions.py](../integration/india_conditions.py)

Named, logged appearance conditions; not a tyre-friction or traffic calibration.

Key entry points: `weather_parameters`.

### [integration/india_dataset_maps.py](../integration/india_dataset_maps.py)

Validate genuinely different OpenDRIVE topologies, all with left-hand lanes.

Key entry points: `validate_layout`.

### [integration/india_decisions.py](../integration/india_decisions.py)

Turn observed controller decisions into frame-linked text. Never controls actors.

Key entry points: `human_name`, `describe_ego`.

### [integration/india_evidence_io.py](../integration/india_evidence_io.py)

Low-latency lossless camera evidence; exact pixels, no JPEG or reduced depth.

Key entry points: `write_rgb_png`, `write_depth_npz`.

### [integration/india_live_preview.py](../integration/india_live_preview.py)

Native camera transport and frame-aligned UI. Never reads/writes a driver.

Key entry points: `LiveRoadPreview`.

### [integration/india_pedestrian_spacing.py](../integration/india_pedestrian_spacing.py)

Conservative entry gap for a crossing person and people already on the footpath.

Key entry points: `entry_blockers`.

### [integration/india_playback.py](../integration/india_playback.py)

Wall-clock playback pacing; never alters the simulation's physics step.

Key entry points: `Playback`.

### [integration/india_road.py](../integration/india_road.py)

A small, explicit left-hand OpenDRIVE junction, with no imported dataset data.

Key entry points: `element`, `add_lane`, `make_road`, `junction_xml`, `validate_offline`.

### [integration/india_road_observer.py](../integration/india_road_observer.py)

Capture paired front RGB/semantic images; no prediction or vehicle control here.

Key entry points: `RoadObserver`.

### [integration/india_scenery.py](../integration/india_scenery.py)

Reusable installed CARLA assets for an illustrative neighbourhood street.

Key entry points: `add_scenery`.

### [integration/india_signals.py](../integration/india_signals.py)

Native vehicle lights reflecting the existing driver; never changes its controls.

Key entry points: `light_bits`, `SignalLights`.

### [integration/india_streetscape.py](../integration/india_streetscape.py)

Raised paving beside the existing driving lanes; never used by the driver.

Key entry points: `paving_pixels`, `texture`, `paving_layout`, `footprint_samples`, `apply_streetscape`.

### [integration/india_surface.py](../integration/india_surface.py)

Collision-free asphalt appearance made from an installed, paintable flat mesh.

Key entry points: `asphalt_texture`, `apply_surface`.

### [integration/india_world_asset_probe.py](../integration/india_world_asset_probe.py)

Inspect native UV/material support instead of claiming texture application worked.

Key entry points: `probe`.

### [integration/india_world_maps.py](../integration/india_world_maps.py)

Five distinct synthetic Indian LHT road settings, independent of the V4 map.

Key entry points: `WorldProfile`, `_header`, `_outer_lanes`, `_curved_road`, `_junction`, `build_world`, `road_specs` (plus supporting helpers).

### [integration/india_world_scene.py](../integration/india_world_scene.py)

Native preview and capture for five separate Indian road worlds.

Key entry points: `save_native`, `image_surface`, `wait_frame`, `cleanup`, `run_world`, `main`.

### [integration/india_world_scenery.py](../integration/india_world_scenery.py)

Auditable native scenery for the five new worlds; no old-map assumptions.

Key entry points: `image_texture`, `WorldScenery`.

### [integration/india_world_traffic.py](../integration/india_world_traffic.py)

Native physics traffic for the new road profiles, using explicit waypoint routes.

Key entry points: `WorldDriver`, `path_for`, `routes_for`, `WorldTraffic`.

## Learned speed training and scoring (rl/)

### [rl/speed_env.py](../rl/speed_env.py)

The fast stand-in town the speed policy is trained in: real town routes, traffic, people, junction turn-taking, the camera driver's perception and rules, and a pedal model fitted to the camera car in CARLA. Numpy, hundreds of drives at once, common random numbers.

### [rl/train_speed_ppo.py](../rl/train_speed_ppo.py) and [rl/launch_wsl_training.sh](../rl/launch_wsl_training.sh)

PPO training in WSL (PyTorch) with held-out evaluation against the rule driver; exports the kept policy for numpy. The launcher detaches the run so WSL does not stop it.

### [rl/export_town_routes.py](../rl/export_town_routes.py)

Builds the town map offline and exports 510 routes (limits, curvature, junctions every metre) for the stand-in.

### [rl/record_policy.py](../rl/record_policy.py)

Installs a trained policy as `models/rl_speed/policy.npz` and writes its tracked record, [models/rl_speed_policy.json](../models/rl_speed_policy.json).

### [rl/analyse_town_run.py](../rl/analyse_town_run.py), [rl/compare_town_pairs.py](../rl/compare_town_pairs.py), [rl/plot_town_pair.py](../rl/plot_town_pair.py)

Score CARLA town drives on ground truth (legs, turn waits, time over the limit, hard braking, open-road speed by limit), tabulate pairs and plot speed along the route.

### [rl/queue_town_pairs.sh](../rl/queue_town_pairs.sh) and [rl/junction_times.py](../rl/junction_times.py)

Run town drives one after another without overlap; junction wait times per movement from past drives (looked at for learned routing).

## MATLAB references

### [integration/matlab/export_reference_preview.m](../integration/matlab/export_reference_preview.m)

MATLAB reference/helper: EXPORT_REFERENCE_PREVIEW Read saved poses; do not rerun the planner.. It is separate from the native CARLA loop.

### [integration/matlab/indian_loop_pose.m](../integration/matlab/indian_loop_pose.m)

MATLAB reference/helper: INDIAN_LOOP_POSE Synthetic rounded-block road, fixed world metres/radians. Counterclockwise road reference. Positive offset is its left (inner) side; clockwise traffic uses negative offset, also left relative to its heading.. It is separate from the native CARLA loop.

### [integration/matlab/inspect_matlab_environment.m](../integration/matlab/inspect_matlab_environment.m)

MATLAB reference/helper: Read-only capability inventory for this project's next integration stage. This does not install products, change licenses or run a training job.. It is separate from the native CARLA loop.

### [integration/matlab/launch_indian_traffic_3d.m](../integration/matlab/launch_indian_traffic_3d.m)

MATLAB reference/helper: LAUNCH_INDIAN_TRAFFIC_3D Open a verified 3D replay in this MATLAB desktop.. It is separate from the native CARLA loop.

### [integration/matlab/launch_indian_traffic_world.m](../integration/matlab/launch_indian_traffic_world.m)

MATLAB reference/helper: LAUNCH_INDIAN_TRAFFIC_WORLD Generate new traffic and open the replay controls. Each new run is preserved. Replaying an existing run does not regenerate it.. It is separate from the native CARLA loop.

### [integration/matlab/load_traffic_replay.m](../integration/matlab/load_traffic_replay.m)

MATLAB reference/helper: LOAD_TRAFFIC_REPLAY Validate a saved rollout before creating graphics/timers. A replay does not use or renew a batch session deadline or monitor lease.. It is separate from the native CARLA loop.

### [integration/matlab/read_camera_prediction.m](../integration/matlab/read_camera_prediction.m)

MATLAB reference/helper: READ_CAMERA_PREDICTION Read Python's 0/1 mask without treating pixels as metres.. It is separate from the native CARLA loop.

### [integration/matlab/replay_indian_traffic_3d.m](../integration/matlab/replay_indian_traffic_3d.m)

MATLAB reference/helper: REPLAY_INDIAN_TRAFFIC_3D Native MATLAB 3D view of saved simulation states. Camera and playback changes never change the recorded experiment.. It is separate from the native CARLA loop.

### [integration/matlab/replay_indian_traffic_world.m](../integration/matlab/replay_indian_traffic_world.m)

MATLAB reference/helper: REPLAY_INDIAN_TRAFFIC_WORLD Interactive playback of an actual saved rollout. Play/Pause, seek and speed change affect playback only, not the experiment.. It is separate from the native CARLA loop.

### [integration/matlab/replay_static_reference.m](../integration/matlab/replay_static_reference.m)

MATLAB reference/helper: REPLAY_STATIC_REFERENCE Visualize the planned car poses at 1x, 4x, etc. This directly replays a geometric path. It is not controller or RL execution.. It is separate from the native CARLA loop.

### [integration/matlab/run_indian_traffic_trial.m](../integration/matlab/run_indian_traffic_trial.m)

MATLAB reference/helper: RUN_INDIAN_TRAFFIC_WORLD Ground-truth mixed-traffic loop development world. Ego: actual Stanley steering + bounded speed feedback + braking override. Other vehicles: prescribed loop positions with car-following speed rules. Pedestrians: seeded footpath walking / waiting / crossing state machine. Not a calibrated Indian traffic model, RL training, or learned-mask planner.. It is separate from the native CARLA loop.

### [integration/matlab/run_indian_traffic_world.m](../integration/matlab/run_indian_traffic_world.m)

MATLAB reference/helper: RUN_INDIAN_TRAFFIC_WORLD Compatibility entry point for reviewed mixed traffic. All actor dynamics live in run_indian_traffic_trial; do not maintain a second simulator here. Earlier run source snapshots preserve their original behavior.. It is separate from the native CARLA loop.

### [integration/matlab/run_static_reference.m](../integration/matlab/run_static_reference.m)

MATLAB reference/helper: RUN_STATIC_REFERENCE One declared ground-truth Hybrid A* development scene. Road geometry is synthetic. No camera mask, RL, moving actors or controller. Vehicle poses refer to the rear axle center; x/y are metres, heading radians.. It is separate from the native CARLA loop.

### [integration/matlab/run_static_tracking_reference.m](../integration/matlab/run_static_tracking_reference.m)

MATLAB reference/helper: RUN_STATIC_TRACKING_REFERENCE One static closed-loop Stanley/PI development run. Uses declared ground-truth geometry and a saved Hybrid A* route. No RL, learned camera map, moving actors, AEB or vehicle tyre/slip dynamics.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_body_gap.m](../integration/matlab/traffic_body_gap.m)

MATLAB reference/helper: TRAFFIC_BODY_GAP Body clearance for rear-axle vehicles and pedestrian discs. Rectangle gaps are separating-axis lower bounds, not Euclidean distances.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_follow_speed.m](../integration/matlab/traffic_follow_speed.m)

MATLAB reference/helper: Route-relative following for the synthetic loop, using current states only. This limits the nominal target; collision prediction and braking still run. Integrate lane distance through curves instead of using the current heading projection, which distorts the gap near a bend.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_maneuver_offset.m](../integration/matlab/traffic_maneuver_offset.m)

MATLAB reference/helper: Continuous position profile with zero lateral slope/curvature at endpoints.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_npc_intervention.m](../integration/matlab/traffic_npc_intervention.m)

MATLAB reference/helper: Record rejected proposals using stable actor IDs, including the ego ID 0. The guard's row index for the appended ego must not masquerade as actor 30.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_npc_speed_target.m](../integration/matlab/traffic_npc_speed_target.m)

MATLAB reference/helper: Full-body following along the curved route; parked actors/peds do not move. The former rear-point/half-length projection could brake after body contact. Only actors in the forward route corridor constrain nominal following. Snapshot geometry is conservative; the independent step guard also runs.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_npc_step_guard.m](../integration/matlab/traffic_npc_step_guard.m)

MATLAB reference/helper: Explicit synthetic safety constraint on prospective NPC motion. Holds complete proposed steps, without moving actors out of existing contact. Every hold is exposed by the caller as an intervention; it is not proof that the behavior policy itself would avoid collisions. Check middle/end poses.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_overtake_plan.m](../integration/matlab/traffic_overtake_plan.m)

MATLAB reference/helper: Ground-truth, rule-based local overtake planner. No RL or future RNG access.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_pedestrian_detour_target.m](../integration/matlab/traffic_pedestrian_detour_target.m)

MATLAB reference/helper: Choose one persistent route around the full stopped-vehicle cluster. Both the approach and eventual crossing must clear every stopped body. Rechoosing the nearest corner every 0.02 s previously oscillated between the front of a stopped car and a neighbouring parked vehicle indefinitely.. It is separate from the native CARLA loop.

### [integration/matlab/traffic_pedestrian_step_score.m](../integration/matlab/traffic_pedestrian_step_score.m)

MATLAB reference/helper: Prefer progress, then passing to the pedestrian's right. Opposite walkers therefore choose opposite side steps instead of both reversing indefinitely.. It is separate from the native CARLA loop.

### [integration/matlab/verify_camera_handoff.m](../integration/matlab/verify_camera_handoff.m)

MATLAB reference/helper: VERIFY_CAMERA_HANDOFF Actual MATLAB read of a Python prediction artifact.. It is separate from the native CARLA loop.

## Setup, audit and reporting utilities

### [setup/adapt_road_boundaries.py](../setup/adapt_road_boundaries.py)

Bounded, separate road-model adaptation. The deployed lock is read-only.

Key entry points: `sha`, `load`, `atomic`, `safe_file`, `protected`, `real_records`, `prepare_capture` (plus supporting helpers).

### [setup/analyse_detection_errors.py](../setup/analyse_detection_errors.py)

Compare fixed detector resolutions and audit misses on labelled validation only.

Key entry points: `main`.

### [setup/audit_prepared_idd_detection.py](../setup/audit_prepared_idd_detection.py)

Independently reopen saved YOLO labels and compare sample boxes to their source.

Key entry points: `box_sort`, `saved_label_path`, `original_boxes`, `main`.

### [setup/audit_real_video_20261019.py](../setup/audit_real_video_20261019.py)

Recount saved video masks/timestamps without loading TensorFlow or IDD data.

Key entry points: `main`.

### [setup/audit_traffic_rollout.py](../setup/audit_traffic_rollout.py)

Independent NumPy audit of recorded 0.1 s actor states, with incident times.

Key entry points: `validated_replay`, `intervals`, `motion_metrics`, `hits`, `audit`.

### [setup/boundary_metrics.py](../setup/boundary_metrics.py)

Independent NumPy validation metrics; no TensorFlow or data mutation.

Key entry points: `morph`, `boundary`, `counts`, `scores`, `summarize`.

### [setup/build_adaptation_gallery.py](../setup/build_adaptation_gallery.py)

CPU-only baseline/candidate gallery from completed saved masks; no inference.

Key entry points: `load`, `picture`, `main`.

### [setup/build_boundary_report.py](../setup/build_boundary_report.py)

Build an honest four-model report, curves, and all-validation review gallery.

Key entry points: `read_image`, `bootstraps`, `font`, `comparison_image`, `main`.

### [setup/build_idd_comparison_gallery.py](../setup/build_idd_comparison_gallery.py)

Build a portable, complete validation gallery from the two saved mask exports.

Key entry points: `atomic`, `load_metrics`, `font`, `render_pair`, `main`.

### [setup/build_idd_test_report.py](../setup/build_idd_test_report.py)

Build the final report and gallery from the single saved test evaluation.

Key entry points: `read`, `save`, `sha`, `font`, `comparison`, `contact_sheet`, `main`.

### [setup/build_indian_world_assets.py](../setup/build_indian_world_assets.py)

Create original Indian shop/sign textures and an import-ready three-wheel asset.

Key entry points: `font`, `text`, `textures`, `Mesh`, `autorickshaw`.

### [setup/build_simulation_dataset.py](../setup/build_simulation_dataset.py)

Build/version/audit simulation pairs from synchronized native evidence; no ML inference.

Key entry points: `load`, `png`, `source_file`, `candidates`, `diverse_selection`, `panel`, `process_episode` (plus supporting helpers).

### [setup/build_static_replay.py](../setup/build_static_replay.py)

Build a standalone geometric replay from the actual MATLAB path output.

Key entry points: `main`.

### [setup/build_validation_label_audit.py](../setup/build_validation_label_audit.py)

Audit three already-known validation cases without inference or label edits.

Key entry points: `sha`, `main`.

### [setup/cache_clean_detection_view.py](../setup/cache_clean_detection_view.py)

Reuse audited per-image framework arrays, checking every retained label again.

Key entry points: `main`.

### [setup/camera_driving_service.py](../setup/camera_driving_service.py)

Shared WSL service loop for the two camera-driving GPU workers. It accepts one authenticated viewer connection and checks protocol, session, token, role, image size and pixel hash before running a model. It stops on its marker or deadline and records whether model weights stayed unchanged.

Key entry points: `serve`.

### [setup/camera_driving_road_worker.py](../setup/camera_driving_road_worker.py)

WSL GPU worker (`idd-tf220`): resizes each 960x540 frame to 512x512 exactly as the live preview does, runs the simulation road model through `idd_driving_inference.py`, and returns the 0/1 mask plus an 8-bit probability.

### [setup/camera_driving_detector_worker.py](../setup/camera_driving_detector_worker.py)

WSL GPU worker (`idd-detect`): runs the deployed IDD detector (`f7bb7e4c`) at 1280 / NMS 0.6 / confidence 0.25, FP16, after checking its manifest, hash and class order. Its Ultralytics settings stay in the run folder with online features off. Returns boxes only.

### [setup/capture_traffic_3d.m](../setup/capture_traffic_3d.m)

MATLAB reference/helper: Export scene views from an explicitly selected saved rollout.. It is separate from the native CARLA loop.

### [setup/carla/install_package.py](../setup/carla/install_package.py)

Finish the official CARLA package installation without changing ML environments.

Key entry points: `atomic_json`, `main`.

### [setup/carla/requirements.txt](../setup/carla/requirements.txt)

Python dependency/version specification for this filename's task/runtime. It supplies packages, not datasets, checkpoints or the CARLA engine.

### [setup/carla/run_session.py](../setup/carla/run_session.py)

Bounded CARLA launcher. Its Windows Job Objects own only the processes it starts.

Key entry points: `atomic`, `owned_native_crash`, `main`.

### [setup/carla/verify_run.py](../setup/carla/verify_run.py)

Independently recount recorded camera, motion and actor evidence.

### [setup/carla/viewer_entry.py](../setup/carla/viewer_entry.py)

Capture even early viewer errors instead of losing them in a hidden console.

### [setup/carla/windows_job.py](../setup/carla/windows_job.py)

Owned Windows process tree with visible application windows and no console.

Key entry points: `CarlaJob`.

### [setup/carla_india/capture_boundary_cases.py](../setup/carla_india/capture_boundary_cases.py)

Disjoint declared training/development/withheld captures for a model candidate.

Key entry points: `main`.

### [setup/carla_india/capture_retention_holdout.py](../setup/carla_india/capture_retention_holdout.py)

Predeclare one fresh synthetic confirmation episode without scoring pixels.

Key entry points: `main`.

### [setup/carla_india/export_quality.py](../setup/carla_india/export_quality.py)

Export a completed suite's measured facts without machine paths or local assets.

Key entry points: `load`, `number`, `main`.

### [setup/carla_india/make_report.py](../setup/carla_india/make_report.py)

Assemble a report only from explicitly selected, independently checked episodes.

Key entry points: `load`, `main`.

### [setup/carla_india/recheck_quality.py](../setup/carla_india/recheck_quality.py)

Fresh owned CARLA sessions and independent audits for the V4 scene changes.

Key entry points: `load`, `main`.

### [setup/carla_india/verify_run.py](../setup/carla_india/verify_run.py)

Independent scalar audit of saved CARLA traces (does not import the controller).

Key entry points: `corners`, `overlap`, `gap`, `percentile`, `audit`, `main`.

### [setup/check_detection_progress.py](../setup/check_detection_progress.py)

Read-only checkpoint, metrics and GPU-health inspection; never resumes training.

Key entry points: `main`.

### [setup/check_idd_detection_framework.py](../setup/check_idd_detection_framework.py)

Check the actual dataset configuration in the isolated detector runtime.

Key entry points: `main`.

### [setup/check_live_repeatability.py](../setup/check_live_repeatability.py)

Repeat the same no-model baseline to quantify native-run variation.

### [setup/check_matlab_review_20261013.m](../setup/check_matlab_review_20261013.m)

MATLAB reference/helper: Save Code Analyzer findings without executing project source files.. It is separate from the native CARLA loop.

### [setup/check_simulation_maps.py](../setup/check_simulation_maps.py)

Query native CARLA maps for LHT topology and all planned footpath samples.

### [setup/check_simulation_preprocessing.py](../setup/check_simulation_preprocessing.py)

Check actual simulation train/val batches without constructing a model or reading test pixels.

Key entry points: `main`.

### [setup/confirm_model_retention.py](../setup/confirm_model_retention.py)

Run three bounded fresh checks sequentially after model selection is frozen.

Key entry points: `load`, `local`, `main`.

### [setup/confirm_simulation_model.py](../setup/confirm_simulation_model.py)

Compare the frozen new simulation model and locked baseline on fresh captures.

Key entry points: `local`, `confirm`.

### [setup/curate_idd_detection_manifest.py](../setup/curate_idd_detection_manifest.py)

Remove recording overlaps from a provider view without moving any source split.

Key entry points: `main`.

### [setup/detection_resolution_pilot.py](../setup/detection_resolution_pilot.py)

Prepare, run and compare one authorized three-epoch resolution adaptation pilot.

Key entry points: `phase`, `paired_validation`, `execute`, `launch`.

### [setup/diagnose_inference_reference.py](../setup/diagnose_inference_reference.py)

Bounded validation-only diagnosis of batch shape and graph placement.

Key entry points: `fingerprint`, `main`.

### [setup/download_idd117k.py](../setup/download_idd117k.py)

Resumable, bounded parallel downloads from the user's authorised IDD links.

Key entry points: `atomic`, `Transfer`, `main`.

### [setup/evaluate_boundary_gpu.py](../setup/evaluate_boundary_gpu.py)

Saved-checkpoint validation exports and batch-one latency; no test images.

Key entry points: `main`.

### [setup/evaluate_idd_test.py](../setup/evaluate_idd_test.py)

One fixed-model test inference pass; export masks, independently verify metrics.

Key entry points: `sha`, `atomic`, `table`, `fingerprint`, `gpu_sample`, `fixture_checks`, `bootstrap` (plus supporting helpers).

### [setup/export_idd_comparison_masks.py](../setup/export_idd_comparison_masks.py)

Inference-only export of baseline/trial-13 masks for the validation gallery.

Key entry points: `sha`, `atomic`, `main`.

### [setup/fetch_driving_clips_20261017.py](../setup/fetch_driving_clips_20261017.py)

Fetch three public, version-pinned MOTOR front-camera development clips.

Key entry points: `atomic_json`, `main`.

### [setup/finalize_detection_preparation.py](../setup/finalize_detection_preparation.py)

Record the completed preparation work while keeping real-data training gated.

Key entry points: `load`, `main`.

### [setup/finalize_idd_detection_quality.py](../setup/finalize_idd_detection_quality.py)

Exclude rejected pairs/leaked train duplicates and reuse the complete pixel audit.

Key entry points: `identity`, `reuse_pair`, `reused_pairs`, `main`.

### [setup/finalize_idd_test_evaluation.py](../setup/finalize_idd_test_evaluation.py)

Record completed test evaluation and verify saved artifacts, without inference.

Key entry points: `read`, `save`, `sha`, `main`.

### [setup/finalize_live_simulation.py](../setup/finalize_live_simulation.py)

Publish the verified live-observer implementation, timings and known limits.

Key entry points: `load`, `main`.

### [setup/finalize_model_retention.py](../setup/finalize_model_retention.py)

Publish the measured retention decision and all fresh simulator comparisons.

Key entry points: `load`, `sha`, `local`, `percent`, `main`.

### [setup/finalize_simulation_confirmation.py](../setup/finalize_simulation_confirmation.py)

Publish verified fresh-scene scores, comparisons and limits; never selects/trains.

Key entry points: `load`, `local`, `percent`, `overlap`, `overlay`, `main`.

### [setup/finalize_simulation_dataset.py](../setup/finalize_simulation_dataset.py)

Publish preparation evidence only after both independent and TensorFlow checks pass.

Key entry points: `load`, `main`.

### [setup/finalize_stream_assets_20261019.py](../setup/finalize_stream_assets_20261019.py)

Final local-file checks and an explicitly unreviewed annotation import.

Key entry points: `Links`, `main`.

### [setup/finish_idd117k_import.py](../setup/finish_idd117k_import.py)

Continue verified extraction through indexing, preprocessing and readiness only.

Key entry points: `main`.

### [setup/finish_simulation_preparation.py](../setup/finish_simulation_preparation.py)

Wait for this dataset build, run two independent checks, then publish evidence. No training.

Key entry points: `main`.

### [setup/import_idd117k_archive.py](../setup/import_idd117k_archive.py)

Validate and safely stream the five-part IDD archive without a second big copy.

Key entry points: `utc`, `write_json`, `safe_name`, `SplitReader`, `metadata_stats`, `run`.

### [setup/index_idd117k_detection.py](../setup/index_idd117k_detection.py)

Index both inspected IDD releases, keeping provider lists and recording drives.

Key entry points: `legacy_drive`, `index_legacy`, `main`.

### [setup/index_idd95k_detection.py](../setup/index_idd95k_detection.py)

Index the observed 95K layout; do not guess the older release's drive IDs.

Key entry points: `index_release`.

### [setup/inspect_driving_clips_20261017.m](../setup/inspect_driving_clips_20261017.m)

MATLAB reference/helper: Verify downloaded video metadata and decode sample frames in MATLAB.. It is separate from the native CARLA loop.

### [setup/inspect_idd_archive.py](../setup/inspect_idd_archive.py)

Inspect a verified, already downloaded prefix without extracting the archive.

Key entry points: `inspect`.

### [setup/keep_detection_pipeline_awake.ps1](../setup/keep_detection_pipeline_awake.ps1)

Temporary keep-awake helper for an owned local job; it manages sleep requests, not model weights or accuracy.

### [setup/keep_detection_training_awake.ps1](../setup/keep_detection_training_awake.ps1)

Temporary keep-awake helper for an owned local job; it manages sleep requests, not model weights or accuracy.

### [setup/keep_simulation_training_awake.ps1](../setup/keep_simulation_training_awake.ps1)

Temporary keep-awake helper for an owned local job; it manages sleep requests, not model weights or accuracy.

### [setup/launch_detection_training.py](../setup/launch_detection_training.py)

Launch one explicitly requested detector run and preserve its exit record.

Key entry points: `directories`, `worker`, `launch`.

### [setup/launch_simulation_training.py](../setup/launch_simulation_training.py)

Launch one fresh simulation run in its own persistent Linux process session.

Key entry points: `launch`.

### [setup/launch_training_dashboard.py](../setup/launch_training_dashboard.py)

Start the local, read-only training dashboard independently of the command window.

Key entry points: `main`.

### [setup/live_simulation_worker.py](../setup/live_simulation_worker.py)

Bounded WSL GPU service for a single authenticated native camera session.

Key entry points: `run`.

### [setup/make_detection_box_gallery.py](../setup/make_detection_box_gallery.py)

Show actual validation labels beside frozen-model detections; no distance claims.

Key entry points: `matches`, `prepare`, `render_pair`, `worker`, `launch`.

### [setup/plot_traffic_evidence.py](../setup/plot_traffic_evidence.py)

Reproducible diagnostic plots from saved runs; never edits rollouts.

Key entry points: `body`, `main`.

### [setup/prepare_clean_detection_view.py](../setup/prepare_clean_detection_view.py)

Preserve validation; remove ambiguous/repeated training pixels in a fresh view.

Key entry points: `select_pairs`, `main`.

### [setup/prepare_detection_lr_probe.py](../setup/prepare_detection_lr_probe.py)

Copy a small, labelled development cohort for a paired learning-rate experiment.

Key entry points: `main`.

### [setup/prepare_detection_refinement.py](../setup/prepare_detection_refinement.py)

Create a verified higher-resolution fine-tuning recipe from the retained detector.

Key entry points: `main`.

### [setup/prepare_fast_detection_probe.py](../setup/prepare_fast_detection_probe.py)

Prepare an isolated, fixed validation probe without changing active training data.

Key entry points: `main`.

### [setup/prepare_idd_detection.py](../setup/prepare_idd_detection.py)

Export an explicitly reviewed provider manifest to a separate YOLO dataset.

### [setup/prepare_idd_test_evaluation.py](../setup/prepare_idd_test_evaluation.py)

Lock the selected model and protocol before opening any test PNG contents.

Key entry points: `digest`, `main`.

### [setup/prepare_indian_world_labels.py](../setup/prepare_indian_world_labels.py)

Save asset-corrected semantic previews while preserving native sensor evidence.

Key entry points: `sha`, `prepare`.

### [setup/prepare_real_retention_guard.py](../setup/prepare_real_retention_guard.py)

Validate the existing human review and freeze an evaluation-only guard set.

Key entry points: `main`.

### [setup/prepare_simulation_confirmation.py](../setup/prepare_simulation_confirmation.py)

Predeclare, audit and capture fresh confirmation scenes; never trains.

Key entry points: `utc`, `plan`, `preflight`, `capture`.

### [setup/prepare_simulation_dataset.py](../setup/prepare_simulation_dataset.py)

Predeclare and capture a separate simulation dataset. Never trains a model.

Key entry points: `sha`, `save`, `protected`, `plan`, `capture`, `main`.

### [setup/probe_detection_recovery.py](../setup/probe_detection_recovery.py)

Discarded training-only smoke check for safe checkpoint recovery; no accuracy evaluation.

Key entry points: `main`.

### [setup/probe_idd_download.py](../setup/probe_idd_download.py)

Inspect the five authorised download endpoints without exposing their tokens.

### [setup/project/bootstrap_carla.ps1](../setup/project/bootstrap_carla.ps1)

Environment/session helper for bootstrap carla. Review machine/runtime prerequisites before use.

### [setup/project/build_model_handoff.py](../setup/project/build_model_handoff.py)

Package the three retained road checkpoints without any image or mask data.

Key entry points: `sha256`, `main`.

### [setup/project/check_repository.py](../setup/project/check_repository.py)

Check the Git index for common accidental disclosures before sharing.

Key entry points: `git`, `main`.

### [setup/queue_traffic_trial.py](../setup/queue_traffic_trial.py)

Append one distinct trial to the bounded overnight controller's queue.

Key entry points: `main`.

### [setup/recover_detection_resolution_pilot.py](../setup/recover_detection_resolution_pilot.py)

One bounded diagnostic recovery from the unchanged parent after a CUDA error.

Key entry points: `execute`, `launch`.

### [setup/rehearse_road_retention.py](../setup/rehearse_road_retention.py)

Separate decoder adaptation with real TRAIN teacher rehearsal and strict gates.

Key entry points: `path_value`, `verified_record`, `strict_ratios`, `prepare`, `decode`, `train`, `main`.

### [setup/relink_idd_prepared_images.py](../setup/relink_idd_prepared_images.py)

Replace only this import's verified duplicate copies with equivalent hard links.

Key entry points: `relink`, `main`.

### [setup/render_indian_sign_text.ps1](../setup/render_indian_sign_text.ps1)

Environment/session helper for render indian sign text. Review machine/runtime prerequisites before use.

### [setup/report_boundary_fix_20261020.py](../setup/report_boundary_fix_20261020.py)

Summarize saved scene, reference and recording evidence without rerunning a model.

Key entry points: `load`, `main`.

### [setup/report_carla_observation_20261020.py](../setup/report_carla_observation_20261020.py)

Publish the verified 20 October 2026 observation report and portable summary.

Key entry points: `main`.

### [setup/report_combined_india.py](../setup/report_combined_india.py)

Publish checked evidence for one connected Indian town without altering models.

Key entry points: `main`.

### [setup/report_detection_refinement.py](../setup/report_detection_refinement.py)

Verify a finished refinement and write its matched validation comparison.

Key entry points: `main`.

### [setup/report_detection_training.py](../setup/report_detection_training.py)

Build a detector result report from preserved local training evidence.

Key entry points: `read`, `sha`, `save`, `main`.

### [setup/report_detection_training_start.py](../setup/report_detection_training_start.py)

Verify successful real training startup without evaluating or changing a model.

Key entry points: `main`.

### [setup/report_idd_detection_import.py](../setup/report_idd_detection_import.py)

Record final detector-data evidence; never trains or changes a road model.

Key entry points: `main`.

### [setup/report_indian_worlds.py](../setup/report_indian_worlds.py)

Package checked native Indian-world previews, map plans and limitations.

Key entry points: `plan`, `main`.

### [setup/report_real_video_20261019.py](../setup/report_real_video_20261019.py)

Make the measured report/plots from a completed, independently audited run.

Key entry points: `main`.

### [setup/report_road_rehearsal.py](../setup/report_road_rehearsal.py)

CPU-only strict-retention report, every reviewed frame and honest regressions.

Key entry points: `load`, `image_path`, `delta`, `percentage`, `render`, `dataset_map`, `main`.

### [setup/report_road_stream_20261019.py](../setup/report_road_stream_20261019.py)

Independently recount stream records and make the next-step report/hub.

Key entry points: `link_source_gallery`, `main`.

### [setup/report_simulation_training.py](../setup/report_simulation_training.py)

Read-only training inspection plus a dated portable results snapshot.

Key entry points: `report`.

### [setup/review_boundary_masks.py](../setup/review_boundary_masks.py)

Review previously exported validation masks without re-running either model.

Key entry points: `sha`, `atomic`, `evaluate`.

### [setup/review_case_4160.py](../setup/review_case_4160.py)

Inspect saved validation predictions without running training or model inference.

### [setup/review_detection_lr_probe.py](../setup/review_detection_lr_probe.py)

Finish the local LR experiment with a conservative, predeclared validation gate.

Key entry points: `review`.

### [setup/review_supplied_video_labels_20261019.py](../setup/review_supplied_video_labels_20261019.py)

Audit supplied human labels and save comparisons; no model inference/training.

Key entry points: `main`.

### [setup/road_observer_template.html](../setup/road_observer_template.html)

Template/viewer source used to display saved evidence or replay; not a trained model.

### [setup/run_boundary_session.py](../setup/run_boundary_session.py)

One finite session; fresh subprocesses prevent stale GPU/optimizer state.

Key entry points: `main`.

### [setup/run_camera_driving.py](../setup/run_camera_driving.py)

Owns one camera-driving session:

1. Writes both model manifests and a session token into a fresh `runs/camera_driving_*` folder.
2. Starts the two WSL workers and waits until both are ready.
3. Runs the connected town with `--camera-drive`.
4. Stops the workers and confirms that the protected models and the rule driver are unchanged.

`--worker-check` pushes saved town frames through the workers instead of driving.

Key entry points: `run`, `check_workers`.

### [setup/run_detection_lr_probe.py](../setup/run_detection_lr_probe.py)

CPU-only paired modest-LR experiment, with matched data and augmented inputs.

Key entry points: `worker`, `launch`.

### [setup/run_fast_detection_probe.py](../setup/run_fast_detection_probe.py)

CPU-only paired validation screening; does not update weights or deploy settings.

Key entry points: `main`.

### [setup/run_full_detection_resolution_check.py](../setup/run_full_detection_resolution_check.py)

Detached CPU validation of a frozen detector while GPU training continues.

Key entry points: `evaluate`, `launch`.

### [setup/run_live_simulation.py](../setup/run_live_simulation.py)

Own one bounded live preview plus optional matched rule-control/fault checks.

Key entry points: `linux`, `native_run`, `run`.

### [setup/run_review_matlab.py](../setup/run_review_matlab.py)

Run one bounded, hidden review batch in its own kill-on-close Windows job.

Key entry points: `main`.

### [setup/score_adaptation_fixed_cases.py](../setup/score_adaptation_fixed_cases.py)

Two predeclared, known development examples; never model selection or training.

Key entry points: `load`, `sha`, `main`.

### [setup/select_simulation_model.py](../setup/select_simulation_model.py)

Freeze a DEV-only simulator choice; never grant shared real-image deployment.

Key entry points: `load`, `sha`, `save_new`, `local`, `main`.

### [setup/serve_boundary_gallery.py](../setup/serve_boundary_gallery.py)

Serve the completed local review gallery on an automatically assigned port.

### [setup/serve_idd_test_gallery.py](../setup/serve_idd_test_gallery.py)

Serve only the completed test comparison gallery on localhost.

### [setup/serve_static_reference.py](../setup/serve_static_reference.py)

Read-only localhost preview of the saved static MATLAB reference artifacts.

### [setup/summarize_overnight_results.py](../setup/summarize_overnight_results.py)

Summarize saved validation evidence; performs no inference or training.

Key entry points: `read`, `sha`, `rows`, `distribution`, `main`.

### [setup/summarize_traffic_trials.py](../setup/summarize_traffic_trials.py)

Build a compact evidence index for the bounded traffic session.

Key entry points: `atomic_text`, `main`.

### [setup/summarize_training_analysis.py](../setup/summarize_training_analysis.py)

Export saved training measurements only to an explicitly selected new folder.

Key entry points: `find_console_log`, `summarize`, `main`.

### [setup/traffic_session.py](../setup/traffic_session.py)

Bounded native Windows MATLAB job runner for the 4 October traffic session.

Key entry points: `STARTUPINFO`, `PROCESSINFO`, `BASICLIMIT`, `EXTENDEDLIMIT`, `utc`, `stamp`, `atomic` (plus supporting helpers).

### [setup/train_boundary_idd.py](../setup/train_boundary_idd.py)

Bounded matched continuation of trial13: ordinary or boundary-weighted loss.

Key entry points: `setup_gpu`, `model_for`, `atomic_model`, `SelectedCheckpoints`, `smoke`, `train`.

### [setup/training_progress_server.py](../setup/training_progress_server.py)

Read-only, localhost dashboard for an existing detection training run.

Key entry points: `read_json`, `snapshot`, `main`.

### [setup/verify_rl_speed.py](../setup/verify_rl_speed.py)

Offline checks of the learned speed: limit profiles, the policy file, the shield never exceeded for any action, the limit hold and gear band, the sat-nav's limits on a real route; `--suites` runs every camera-driving and destination check again with the learned speed on.

### [setup/verify_camera_driving.py](../setup/verify_camera_driving.py)

Offline checks for camera-only driving, with no CARLA server and no GPU:

- camera geometry round trip
- synthetic road masks for kerb following, curves, openings and obstacles
- closed-loop bicycle-model drives
- passing a stopped bus with the rear camera: a clear lane, a car coming up behind, a bus coming the other way mid-pass (stop, back out, retry), no rear view, and backing up with a car behind
- a village road with a paved shoulder the road model counts as road, roadside carts beside the car, and a stopped motorcycle it must wait behind
- worker protocol with stand-in workers, including the rear picture
- a source check that the driver has no simulator access

`--replay` also scores the driver's target line on the 120 recorded town frames against the true keep-left lane centre from the map (run with `.venv-carla`).

### [setup/verify_camera_destinations.py](../setup/verify_camera_destinations.py)

Offline checks for driving to picked destinations. It needs no CARLA server or GPU, but run it with `.venv-carla`: it builds the town map offline. It checks:

- routes between all 11 places
- clicks snapping to the right lane
- Hybrid A* turns on a synthetic crossroads
- closed-loop left, right and straight-on turns, arrival, waiting its turn and the bus cases
- the sat-nav's messages and re-plans
- the map picker and the command-line options

`--plan-test <town run folder>` chooses the town test's destination lists.

### [setup/verify_carla_road_metrics.py](../setup/verify_carla_road_metrics.py)

Check label semantics, ignored pixels and the independent depth projection kernel.

Key entry points: `RoadMetricsTests`.

### [setup/verify_carla_road_result.py](../setup/verify_carla_road_result.py)

Recount saved predictions and check depth-projected road points against OpenDRIVE.

Key entry points: `load`, `channel`, `main`.

### [setup/verify_combined_india.py](../setup/verify_combined_india.py)

Independent continuity, identity, geometry, camera and asset-label audit.

Key entry points: `sha`, `rectangle`, `overlap`, `audit`.

### [setup/verify_confirmation_capture.py](../setup/verify_confirmation_capture.py)

Audit all fresh episodes, their scene differences, cameras and native traces.

Key entry points: `main`.

### [setup/verify_confirmation_protocol.py](../setup/verify_confirmation_protocol.py)

Regression checks for the fresh-model manifest and reference edge policy.

Key entry points: `main`.

### [setup/verify_detection_data.py](../setup/verify_detection_data.py)

Independent numerical and failure-case checks for detection preparation.

Key entry points: `rejected`, `main`.

### [setup/verify_detection_file_audit.py](../setup/verify_detection_file_audit.py)

Check the optimized pre-training file guard without loading or training a model.

Key entry points: `main`.

### [setup/verify_detection_integration.py](../setup/verify_detection_integration.py)

Check exported labels with the real framework and build a 12-class head.

Key entry points: `main`.

### [setup/verify_detection_refinement.py](../setup/verify_detection_refinement.py)

Check sampling probabilities and reject mismatched refinement provenance.

Key entry points: `main`.

### [setup/verify_detection_resume.py](../setup/verify_detection_resume.py)

Verify real optimizer execution after restoring the interrupted detector run.

Key entry points: `read`, `verify`.

### [setup/verify_detection_runtime.py](../setup/verify_detection_runtime.py)

Forward-only GPU readiness check using official COCO weights, never trains.

Key entry points: `sha`, `save`, `protection`, `main`.

### [setup/verify_detection_workflow.py](../setup/verify_detection_workflow.py)

Check source indexing and that incomplete/fixture data cannot start training.

Key entry points: `rejected`, `main`.

### [setup/verify_idd117k_import.py](../setup/verify_idd117k_import.py)

Exercise split boundaries, gzip corruption and contained extraction on fixtures.

Key entry points: `main`.

### [setup/verify_idd_detection_quality.py](../setup/verify_idd_detection_quality.py)

Verify quality exclusions, pixel duplicates, reused audits and fixture isolation.

Key entry points: `main`.

### [setup/verify_india_decision_run.py](../setup/verify_india_decision_run.py)

Independently join decision labels to actor traces, camera frames and HUD captures.

Key entry points: `load`, `actor_rows`, `main`.

### [setup/verify_india_decisions.py](../setup/verify_india_decisions.py)

Verify explanations are observational and preserve the preceding driving policy.

Key entry points: `method`, `is_observation`, `RemoveObservationWrites`, `RemovePedestrianEntryGap`, `Context`, `context`, `policy_state` (plus supporting helpers).

### [setup/verify_india_geometry.py](../setup/verify_india_geometry.py)

Compare the scalar body-gap kernel with the preceding implementation and audit.

Key entry points: `baseline`, `examples`, `GeometryTests`, `benchmark`.

### [setup/verify_india_playback.py](../setup/verify_india_playback.py)

Check wall pacing and pause measurements without starting CARLA.

Key entry points: `PlaybackTests`.

### [setup/verify_india_quality.py](../setup/verify_india_quality.py)

Check pedestrian entry conflicts and native light flags without running CARLA.

Key entry points: `QualityTests`.

### [setup/verify_india_speed_run.py](../setup/verify_india_speed_run.py)

Audit actual speed key handlers, pause/replay and saved native CARLA traces.

Key entry points: `load`, `main`.

### [setup/verify_india_streetscape.py](../setup/verify_india_streetscape.py)

Check the road/parking definition against new footpath geometry offline.

Key entry points: `StreetscapeTests`.

### [setup/verify_indian_3d.m](../setup/verify_indian_3d.m)

MATLAB reference/helper: verify indian 3d. It is separate from the native CARLA loop.

### [setup/verify_indian_replay.m](../setup/verify_indian_replay.m)

MATLAB reference/helper: Exercise the actual MATLAB replay callbacks after the name-resolution fix.. It is separate from the native CARLA loop.

### [setup/verify_indian_worlds.py](../setup/verify_indian_worlds.py)

Independent map, captured-stream, actor-motion and protection checks.

Key entry points: `fingerprint`, `offline`, `write`, `native`, `main`.

### [setup/verify_inference_review_20261013.py](../setup/verify_inference_review_20261013.py)

Synthetic deployment checks; never train or read reserved test image contents.

Key entry points: `Contracts`, `Maintenance`, `preserved_artifacts`, `tensorflow_synthetic_checks`, `main`.

### [setup/verify_live_protocol.py](../setup/verify_live_protocol.py)

Check scope, message bounds, latest-only buffering and freshness arithmetic.

Key entry points: `main`.

### [setup/verify_live_simulation.py](../setup/verify_live_simulation.py)

Independent source, timing, stale-display, mask and matched-control audit.

Key entry points: `load`, `rows`, `distribution`, `episode`, `compare_controls`, `verify`.

### [setup/verify_locked_inference.py](../setup/verify_locked_inference.py)

Verify deployment inference on six predetermined validation examples only.

Key entry points: `fingerprint`, `main`.

### [setup/verify_overnight_guard.py](../setup/verify_overnight_guard.py)

Verify deadline/lease behavior and that only a session-owned test child is stopped.

### [setup/verify_review_selection_20261013.m](../setup/verify_review_selection_20261013.m)

MATLAB reference/helper: VERIFY_REVIEW_SELECTION_20261013 Check promoted recordings and baseline access. Opens saved viewers only, and closes only figures created by this verifier. The cleanup callback must not depend on a nested workspace being dismantled. A handle container keeps the current owned figures available on exception exit.. It is separate from the native CARLA loop.

### [setup/verify_road_adaptation.py](../setup/verify_road_adaptation.py)

CPU-only checks for the separate adaptation evaluation and split handling.

Key entry points: `AdaptationTests`.

### [setup/verify_road_adaptation_result.py](../setup/verify_road_adaptation_result.py)

Independent saved-PNG recount for completed adaptation validation exports.

Key entry points: `channel`, `main`.

### [setup/verify_road_evidence_writer.py](../setup/verify_road_evidence_writer.py)

Evidence writes must finish in order and propagate errors to the capture.

Key entry points: `observer`, `WriterTests`.

### [setup/verify_road_labels_20261019.py](../setup/verify_road_labels_20261019.py)

Synthetic label/scoring checks; does not create human labels for real clips.

Key entry points: `rle`, `main`.

### [setup/verify_road_rehearsal.py](../setup/verify_road_rehearsal.py)

CPU-only regression tests for exact strict real-retention decisions.

Key entry points: `StrictRetentionTests`.

### [setup/verify_road_retention_result.py](../setup/verify_road_retention_result.py)

Recount rehearsal masks independently, including the strict boundary guard.

Key entry points: `load`, `sha`, `local`, `channel`, `erode`, `recount`, `retention` (plus supporting helpers).

### [setup/verify_road_stream_20261019.py](../setup/verify_road_stream_20261019.py)

Exercise queue overload, stop/exception cleanup and exact reference masks.

Key entry points: `FakePredictor`, `main`.

### [setup/verify_road_video_20261019.py](../setup/verify_road_video_20261019.py)

Small end-to-end video contract check. Uses a fake predictor, no dataset/TF.

Key entry points: `FakeSegmenter`, `write_fixture`, `main`.

### [setup/verify_runtime_review_20261013.py](../setup/verify_runtime_review_20261013.py)

Bounded guard and independent-audit regressions; never runs the old controller.

Key entry points: `LeaseTests`, `fixture`, `AuditTests`, `JobContractTests`.

### [setup/verify_simulation_dataset.py](../setup/verify_simulation_dataset.py)

Independent saved-file/geometry audit; never predicts, trains or selects a model.

Key entry points: `load`, `pixels`, `main`.

### [setup/verify_simulation_model_scope.py](../setup/verify_simulation_model_scope.py)

Check that experimental simulator selection cannot replace the real default.

Key entry points: `main`.

### [setup/verify_simulation_test.py](../setup/verify_simulation_test.py)

Independently recount all saved test predictions; no TensorFlow/model calls.

Key entry points: `verify`.

### [setup/verify_simulation_training.py](../setup/verify_simulation_training.py)

Check masked math, global metrics, GPU learning, frozen encoder and reload.

Key entry points: `weight_hash`, `math_checks`, `verify`.

### [setup/verify_static_tracking.py](../setup/verify_static_tracking.py)

Independently audit saved MATLAB motion, bounds, and goal using NumPy.

Key entry points: `main`.

### [setup/verify_traffic_maneuvers.m](../setup/verify_traffic_maneuvers.m)

MATLAB reference/helper: Deterministic preflight gates at the real recorded overtake start state.. It is separate from the native CARLA loop.

### [setup/verify_traffic_review_20261013.m](../setup/verify_traffic_review_20261013.m)

MATLAB reference/helper: Regression evidence on historical traffic failures; no segmentation data.. It is separate from the native CARLA loop.

### [setup/verify_trained_detection_checkpoint.py](../setup/verify_trained_detection_checkpoint.py)

Verify saved detector checkpoints without training or rescoring the dataset.

Key entry points: `sha`, `save`, `main`.

### [setup/verify_viewer_review_20261013.m](../setup/verify_viewer_review_20261013.m)

MATLAB reference/helper: VERIFY_VIEWER_REVIEW_20261013 Exercise real MATLAB replay controls and inputs. Uses saved simulation data plus isolated fixtures. Generates no traffic trial.. It is separate from the native CARLA loop.

## Documentation and saved evidence

### [docs/CAMERA_DRIVING.md](../docs/CAMERA_DRIVING.md)

Documentation: camera-only driving in the connected town. Covers how to run it, measured results from the CARLA runs, and known problems.

### [docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md](../docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md)

The approved design (8 Nov) for picking a destination: what the car may know (sat-nav plus camera), the route planner, Hybrid A* junction turns, turn-taking, the map picker and the tests.

### [docs/CAMERA_DRIVING_DESTINATIONS_PLAN_20261108.md](../docs/CAMERA_DRIVING_DESTINATIONS_PLAN_20261108.md)

The implementation plan for that design: seven tasks with their checks.

### [docs/CAMERA_DRIVING_DESTINATIONS_RULINGS_20261109.md](../docs/CAMERA_DRIVING_DESTINATIONS_RULINGS_20261109.md)

Every decision taken while building the destinations plan (8-9 Nov), each with what it costs if wrong, and the minor final-review findings left for later.

### [docs/CAMERA_DRIVING_DESIGN_20261106.md](../docs/CAMERA_DRIVING_DESIGN_20261106.md)

Design for camera-only driving, approved 6 November 2026. It covers the input contract, the probe of both models on 120 saved town frames, left-kerb lane keeping, junctions, obstacles and speed, failure handling and testing.

### [docs/CAMERA_OBSERVATION.md](../docs/CAMERA_OBSERVATION.md)

Documentation: Road-model observation in the CARLA neighbourhood. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/COMBINED_INDIAN_TOWN.md](../docs/COMBINED_INDIAN_TOWN.md)

Documentation: One connected Indian town. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DECISION_DISPLAY.md](../docs/DECISION_DISPLAY.md)

Documentation: Visible driving decisions. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_BOX_EXAMPLES.md](../docs/DETECTION_BOX_EXAMPLES.md)

Documentation: Actual validation boxes and predictions. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_FAST_IMPROVEMENT.md](../docs/DETECTION_FAST_IMPROVEMENT.md)

Documentation: Fast detector improvement check — 25 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_FULL_RESOLUTION_20261025.md](../docs/DETECTION_FULL_RESOLUTION_20261025.md)

Documentation: Full detector resolution check - 25 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_IMPROVEMENT.md](../docs/DETECTION_IMPROVEMENT.md)

Documentation: Detector improvement — 24 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_LEARNING_RATE_TEST.md](../docs/DETECTION_LEARNING_RATE_TEST.md)

Documentation: Small learning-rate increase test. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_PREPARATION.md](../docs/DETECTION_PREPARATION.md)

Documentation: Object-detector preparation — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_PROGRESS_20261025.md](../docs/DETECTION_PROGRESS_20261025.md)

Documentation: Detector progress check — 25 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_RECOVERY.md](../docs/DETECTION_RECOVERY.md)

Documentation: Detector training recovery — 24 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_REFINEMENT_RESULT.md](../docs/DETECTION_REFINEMENT_RESULT.md)

Documentation: Detector refinement result — detection_train_20261025_02. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_RESOLUTION_PILOT_RESULT.md](../docs/DETECTION_RESOLUTION_PILOT_RESULT.md)

Documentation: Three-epoch resolution adaptation result - 25 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_TRAINING.md](../docs/DETECTION_TRAINING.md)

Documentation: IDD object-detector training — 24 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/DETECTION_TRAINING_REPORT.md](../docs/DETECTION_TRAINING_REPORT.md)

Documentation: IDD object-detector training report — 24 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/FILE_GUIDE.md](../docs/FILE_GUIDE.md)

This file: the complete repository-purpose index.

### [docs/IDD_DETECTION_DATASET.md](../docs/IDD_DETECTION_DATASET.md)

Documentation: Real Indian-road object detection dataset — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/IDD_DETECTION_IMPORT.md](../docs/IDD_DETECTION_IMPORT.md)

Documentation: IDD detection import and preprocessing — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/INDEX.md](../docs/INDEX.md)

Documentation: Documentation map. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/INDIAN_WORLDS.md](../docs/INDIAN_WORLDS.md)

Documentation: Five Indian road worlds — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/LIVE_SIMULATION_OBSERVATION.md](../docs/LIVE_SIMULATION_OBSERVATION.md)

Documentation: Live simulation road observation — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/MILESTONE_2_EVIDENCE.md](../docs/MILESTONE_2_EVIDENCE.md)

Documentation: Milestone 2 evidence gallery. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/MILESTONE_2_PROGRESS.md](../docs/MILESTONE_2_PROGRESS.md)

Documentation: Milestone 2 progress report. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/MILESTONE_2_VIVA_GUIDE.md](../docs/MILESTONE_2_VIVA_GUIDE.md)

Documentation: Understand the project from zero. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/NEXT_STEPS.md](../docs/NEXT_STEPS.md)

Documentation: Next project steps. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/PLAYBACK_CONTROLS.md](../docs/PLAYBACK_CONTROLS.md)

Documentation: Playback controls — checked 19 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/REAL_MODEL_RETENTION.md](../docs/REAL_MODEL_RETENTION.md)

Documentation: Simulator adaptation with protected real predictions — 21 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/ROAD_BOUNDARY_FIX.md](../docs/ROAD_BOUNDARY_FIX.md)

Documentation: Road-boundary and recording fixes — 20 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/SIMULATION_CONFIRMATION.md](../docs/SIMULATION_CONFIRMATION.md)

Documentation: Fresh simulation confirmation — 23 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/SIMULATION_DATASET.md](../docs/SIMULATION_DATASET.md)

Documentation: Separate Indian-style simulation dataset. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/SIMULATION_QUALITY.md](../docs/SIMULATION_QUALITY.md)

Documentation: Asphalt and simulation quality pass — 20 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/SIMULATION_TEST.md](../docs/SIMULATION_TEST.md)

Documentation: Simulation test score — 22 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/SIMULATION_TRAINING.md](../docs/SIMULATION_TRAINING.md)

Documentation: Separate simulation model training — 22 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/TEAM_OWNERSHIP.md](../docs/TEAM_OWNERSHIP.md)

Documentation: Atharv and Sarthak: balanced module ownership and viva preparation. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/configs/detection_pilot_1280.json](../docs/configs/detection_pilot_1280.json)

Portable snapshot of settings for a recorded experiment. Required local data/parent provenance are separate; it does not automatically start training.

### [docs/configs/road_finetuning_selected.json](../docs/configs/road_finetuning_selected.json)

Portable snapshot of settings for a recorded experiment. Required local data/parent provenance are separate; it does not automatically start training.

### [docs/results/asphalt_comparison.json](../docs/results/asphalt_comparison.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/asphalt_glare_comparison.json](../docs/results/asphalt_glare_comparison.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/camera_observation.json](../docs/results/camera_observation.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/SCENARIO_REPORT.md](../docs/results/carla/SCENARIO_REPORT.md)

Documentation: Indian traffic scenario — checked build, 19 October 2026. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [docs/results/carla/clear_overtake_seed7/actors.json](../docs/results/carla/clear_overtake_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/clear_overtake_seed7/config.json](../docs/results/carla/clear_overtake_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/clear_overtake_seed7/incidents.json](../docs/results/carla/clear_overtake_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/clear_overtake_seed7/independent_audit.json](../docs/results/carla/clear_overtake_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/clear_overtake_seed7/result.json](../docs/results/carla/clear_overtake_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/controlled_to_final_policy.diff](../docs/results/carla/controlled_to_final_policy.diff)

Saved source comparison supporting the documented controlled-to-final policy check; not executable code.

### [docs/results/carla/driver.png](../docs/results/carla/driver.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/carla/follow.png](../docs/results/carla/follow.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/carla/junction.png](../docs/results/carla/junction.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/carla/junction_seed7/actors.json](../docs/results/carla/junction_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/junction_seed7/config.json](../docs/results/carla/junction_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/junction_seed7/incidents.json](../docs/results/carla/junction_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/junction_seed7/independent_audit.json](../docs/results/carla/junction_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/junction_seed7/result.json](../docs/results/carla/junction_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed42/actors.json](../docs/results/carla/mixed_seed42/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed42/config.json](../docs/results/carla/mixed_seed42/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed42/incidents.json](../docs/results/carla/mixed_seed42/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed42/independent_audit.json](../docs/results/carla/mixed_seed42/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed42/result.json](../docs/results/carla/mixed_seed42/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed7/actors.json](../docs/results/carla/mixed_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed7/config.json](../docs/results/carla/mixed_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed7/incidents.json](../docs/results/carla/mixed_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed7/independent_audit.json](../docs/results/carla/mixed_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/mixed_seed7/result.json](../docs/results/carla/mixed_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/native_failure_check.json](../docs/results/carla/native_failure_check.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/oncoming_seed7/actors.json](../docs/results/carla/oncoming_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/oncoming_seed7/config.json](../docs/results/carla/oncoming_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/oncoming_seed7/incidents.json](../docs/results/carla/oncoming_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/oncoming_seed7/independent_audit.json](../docs/results/carla/oncoming_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/oncoming_seed7/result.json](../docs/results/carla/oncoming_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/pedestrian_seed7/actors.json](../docs/results/carla/pedestrian_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/pedestrian_seed7/config.json](../docs/results/carla/pedestrian_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/pedestrian_seed7/incidents.json](../docs/results/carla/pedestrian_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/pedestrian_seed7/independent_audit.json](../docs/results/carla/pedestrian_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/pedestrian_seed7/result.json](../docs/results/carla/pedestrian_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/replay_control_check.json](../docs/results/carla/replay_control_check.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/return_gap_seed7/actors.json](../docs/results/carla/return_gap_seed7/actors.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/return_gap_seed7/config.json](../docs/results/carla/return_gap_seed7/config.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/return_gap_seed7/incidents.json](../docs/results/carla/return_gap_seed7/incidents.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/return_gap_seed7/independent_audit.json](../docs/results/carla/return_gap_seed7/independent_audit.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/return_gap_seed7/result.json](../docs/results/carla/return_gap_seed7/result.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/carla/verified_scene.json](../docs/results/carla/verified_scene.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/combined_indian_town.json](../docs/results/combined_indian_town.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/decision_display.json](../docs/results/decision_display.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_improvement.json](../docs/results/detection_improvement.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_preparation.json](../docs/results/detection_preparation.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_progress_20261025.json](../docs/results/detection_progress_20261025.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_refinement.json](../docs/results/detection_refinement.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_refinement_start.json](../docs/results/detection_refinement_start.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_resolution_pilot.json](../docs/results/detection_resolution_pilot.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_resume_20261024.json](../docs/results/detection_resume_20261024.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_resume_20261025.json](../docs/results/detection_resume_20261025.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_training.json](../docs/results/detection_training.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/detection_training_start.json](../docs/results/detection_training_start.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/geometry_kernel_benchmark.json](../docs/results/geometry_kernel_benchmark.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/idd_detection_import.json](../docs/results/idd_detection_import.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/indian_worlds.json](../docs/results/indian_worlds.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/live_simulation_observation.json](../docs/results/live_simulation_observation.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/milestone2/detector_class_comparison.png](../docs/results/milestone2/detector_class_comparison.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/milestone2/evidence_manifest.json](../docs/results/milestone2/evidence_manifest.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/milestone2/live_road_preview.png](../docs/results/milestone2/live_road_preview.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/milestone2/market_street.png](../docs/results/milestone2/market_street.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/milestone2/publication_checks.json](../docs/results/milestone2/publication_checks.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/milestone2/rural_road.png](../docs/results/milestone2/rural_road.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/milestone2/town_plan.png](../docs/results/milestone2/town_plan.png)

Curated first-party synthetic evidence or aggregate chart. Its caption/report states scope; a picture is not a general performance guarantee.

### [docs/results/model_retention.json](../docs/results/model_retention.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/playback_controls.json](../docs/results/playback_controls.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/road_boundary_fix.json](../docs/results/road_boundary_fix.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/segmentation.json](../docs/results/segmentation.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/simulation_confirmation.json](../docs/results/simulation_confirmation.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/simulation_dataset.json](../docs/results/simulation_dataset.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/simulation_quality.json](../docs/results/simulation_quality.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/simulation_test.json](../docs/results/simulation_test.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

### [docs/results/simulation_training.json](../docs/results/simulation_training.json)

Portable saved result/configuration/audit snapshot. Read its schema, date, passed/status fields and parent report; it does not execute an experiment.

## Scene assets and road descriptions

### [integration/INDIAN_TRAFFIC_WORLD.md](../integration/INDIAN_TRAFFIC_WORLD.md)

Documentation: Indian traffic prototype. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [integration/README.md](../integration/README.md)

Documentation: Planning and simulation continuation. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [integration/TRAFFIC_3D_GUIDE.md](../integration/TRAFFIC_3D_GUIDE.md)

Documentation: 3D traffic in MATLAB. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [integration/carla_india/map_verification.json](../integration/carla_india/map_verification.json)

Structured settings or saved verification metadata for the corresponding module. Read its schema and associated source; this file does not execute code.

### [integration/carla_india/neighbourhood_lht.xodr](../integration/carla_india/neighbourhood_lht.xodr)

Generated OpenDRIVE road/lane/junction geometry for the documented map, not a model or image dataset.

### [integration/indian_world_assets/autorickshaw_source/asset_manifest.json](../integration/indian_world_assets/autorickshaw_source/asset_manifest.json)

Declares dimensions/artwork metadata and explicitly unverified physics/semantics; runtime/training use is disabled.

### [integration/indian_world_assets/autorickshaw_source/autorickshaw.mtl](../integration/indian_world_assets/autorickshaw_source/autorickshaw.mtl)

Color/material declarations used by the OBJ source mesh.

### [integration/indian_world_assets/autorickshaw_source/autorickshaw.obj](../integration/indian_world_assets/autorickshaw_source/autorickshaw.obj)

Original three-wheel mesh source artwork, not a cooked CARLA vehicle blueprint.

### [integration/indian_world_assets/manifest.json](../integration/indian_world_assets/manifest.json)

Hashes and metadata for original sign/shop textures and the source-only autorickshaw artwork; distinguishes artwork from a registered runtime vehicle.

### [integration/indian_world_assets/textures/banner_bus_stop.png](../integration/indian_world_assets/textures/banner_bus_stop.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_delivery.png](../integration/indian_world_assets/textures/banner_delivery.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_keep_left.png](../integration/indian_world_assets/textures/banner_keep_left.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_school.png](../integration/indian_world_assets/textures/banner_school.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_0.png](../integration/indian_world_assets/textures/banner_shop_0.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_1.png](../integration/indian_world_assets/textures/banner_shop_1.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_2.png](../integration/indian_world_assets/textures/banner_shop_2.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_3.png](../integration/indian_world_assets/textures/banner_shop_3.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_4.png](../integration/indian_world_assets/textures/banner_shop_4.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_shop_5.png](../integration/indian_world_assets/textures/banner_shop_5.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_speed_20.png](../integration/indian_world_assets/textures/banner_speed_20.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_speed_30.png](../integration/indian_world_assets/textures/banner_speed_30.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_speed_40.png](../integration/indian_world_assets/textures/banner_speed_40.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/banner_village.png](../integration/indian_world_assets/textures/banner_village.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/bus_stop.png](../integration/indian_world_assets/textures/bus_stop.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/delivery.png](../integration/indian_world_assets/textures/delivery.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/keep_left.png](../integration/indian_world_assets/textures/keep_left.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/school.png](../integration/indian_world_assets/textures/school.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_0.png](../integration/indian_world_assets/textures/shop_0.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_1.png](../integration/indian_world_assets/textures/shop_1.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_2.png](../integration/indian_world_assets/textures/shop_2.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_3.png](../integration/indian_world_assets/textures/shop_3.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_4.png](../integration/indian_world_assets/textures/shop_4.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/shop_5.png](../integration/indian_world_assets/textures/shop_5.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/speed_20.png](../integration/indian_world_assets/textures/speed_20.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/speed_30.png](../integration/indian_world_assets/textures/speed_30.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/speed_40.png](../integration/indian_world_assets/textures/speed_40.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

### [integration/indian_world_assets/textures/text_layout.json](../integration/indian_world_assets/textures/text_layout.json)

Text-layout metadata used to reproduce the project-authored bilingual signs.

### [integration/indian_world_assets/textures/uv_grid.png](../integration/indian_world_assets/textures/uv_grid.png)

Diagnostic texture for inspecting native mesh UV/material mapping, not a dataset photo.

### [integration/indian_world_assets/textures/village.png](../integration/indian_world_assets/textures/village.png)

Original small scene sign/shop texture used by the native scenery. Banner variants account for the inspected mesh UV orientation.

## Model metadata

### [models/README.md](../models/README.md)

Documentation: Retained models. Read its date/scope; older documents are historical snapshots, as indexed in docs/INDEX.md.

### [models/model_config.json](../models/model_config.json)

Real-road architecture and initial training choices plus current selection/test pointers; not every later experiment recipe.

### [models/perception_models.json](../models/perception_models.json)

Current cross-module handoff catalog: real road, synthetic road and latest detector hashes, tasks and result scopes; checkpoint binaries stay local.

### [models/retained_models.json](../models/retained_models.json)

Protected original road-model catalog with original, morning-best and boundary variants; it is not the detector catalog.

### [models/segmentation_final_lock.json](../models/segmentation_final_lock.json)

Fixed real-road model hash, threshold, preprocessing and split/test protocol chosen before final test access.
