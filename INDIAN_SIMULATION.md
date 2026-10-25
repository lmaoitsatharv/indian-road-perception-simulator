# Indian-style traffic in CARLA

## One connected Indian town

Double-click **Start_Combined_Indian_Traffic.cmd** or **Start_Indian_Worlds.cmd**
for a single connected world containing every district: market, colony, divided
main road, village roads and irregular junction. Traffic exists across the whole
town at once; areas are connected by driving lanes. TAB switches front/follow
views; SPACE pauses; S saves a picture; ESC closes. Read
[the combined-town guide](docs/COMBINED_INDIAN_TOWN.md).

## Five additional road settings

Double-click **Start_Separate_Indian_Worlds.cmd** for separate market street, residential colony,
divided urban road, rural road and irregular-junction scenes. This uses a separate
native preview/traffic setup. See [the world guide and asset limitations](docs/INDIAN_WORLDS.md).

## Live road-model preview

Double-click **Start_Live_Road_Preview.cmd** to run the trained simulation model
on the live native front camera. The left panel shows the current camera; the
right panel draws the road prediction on its own matching source frame. The
existing driver view and action explanations remain below. This mode uses a
separate simulation-only model and leaves vehicle control with the rule driver.

The default is a 110-second Indian-style left-hand mixed-traffic scene, seed
4101, clear weather, 1x speed and Low graphics, with a 5 Hz model camera. Model
and simulator startup take some time. SPACE pauses, S saves, and ESC closes this
owned simulator and GPU service. Use one dedicated session at a time.

```powershell
.\Start_Live_Road_Preview.ps1 -Seconds 60 -Scenario mixed -Weather clear
.\Start_Live_Road_Preview.ps1 -Seconds 40 -Check
```

`-Check` deliberately delays/disconnects the service, pauses the scene, and also
records a matched no-model baseline. Normal launch injects no faults. Every run
preserves its settings and evidence under `runs/simulation_live_*`.

Stale/offline masks are hidden on redraw, with 500 ms wall/simulation age limits.
The final check processed 100/100 frames with zero reconnects and a 60.36 ms
median source-step-to-result age upper bound. Two redraw stalls held a previous
mask up to 14.87 ms beyond the target; this is an observation preview with a
measured display scheduling limit. See
[the live preview report](docs/LIVE_SIMULATION_OBSERVATION.md) for timing,
screenshots, fault recovery and exact controller comparison.

## Play

The default scene now uses asphalt-grey roads and worn white markings. Read the
[20 October quality pass](docs/SIMULATION_QUALITY.md) for the pedestrian-spacing,
vehicle-light, weather and evaluation changes.

The V4.2 follow-up adds raised paving and a faster lossless recorder. Read
[the boundary/throughput results](docs/ROAD_BOUNDARY_FIX.md). The normal viewer
does not record the extra analysis cameras. To record them explicitly:

```powershell
.\Start_Indian_Traffic.ps1 -Speed 3 -ObserveRoad
.\Start_Indian_Traffic.ps1 -Speed 3 -ObserveRoad -RoadCameraHz 1
```

The first records paired evidence at 5 Hz; the second uses 1 Hz to collect fewer
nearby frames. Both retain the same physics step and viewer camera. The HUD shows
the recording rate, and `performance.json` records measured timing by stage.

Double-click **Start_Indian_Traffic.cmd** in this project folder. It starts seed 7 and opens the 3D camera window. Allow the simulator to load before interacting. The session is bounded to ten minutes; closing the camera window also closes its dedicated simulator.

Use the window titled **Indian traffic neighbourhood** for controls. The extra CARLA engine window shows a junction overview. Close through **Esc** in the camera window so both shut down together.

In the camera window:

| Key | Action |
| --- | --- |
| **Tab** | Follow car → driver camera → fixed junction camera |
| **Space** | Pause/resume the entire simulated world |
| **R** | Start a fresh episode of the same scenario and seed; retain the previous episode's evidence |
| **S** | Save the camera frame, full window with decision text, and matching JSON record |
| **+ / -** | Increase/decrease whole-world playback: 1×, 1.5×, 2×, 2.5×, 3× |
| **1 / 2 / 3** | Select 1× / 2× / 3× directly |
| **Esc** | Close and clean up this session |

The ego car has a finite route across the neighbourhood. At its goal, the viewer pauses. Press **R** to replay. This is a native CARLA scene, separate from the older MATLAB replay.

While paused, **S** saves the frozen image. Camera changes take effect when simulation resumes.

To select a seed or scenario, open PowerShell in this folder:

```powershell
.\Start_Indian_Traffic.ps1 -Seed 7 -Scenario mixed
.\Start_Indian_Traffic.ps1 -Seed 42 -Scenario mixed
.\Start_Indian_Traffic.ps1 -Seed 7 -Camera driver
.\Start_Indian_Traffic.ps1 -Seed 7 -Speed 3
.\Start_Indian_Traffic.ps1 -Scenario clear_overtake
.\Start_Indian_Traffic.ps1 -Scenario oncoming
.\Start_Indian_Traffic.ps1 -Scenario pedestrian
.\Start_Indian_Traffic.ps1 -Scenario return_gap
.\Start_Indian_Traffic.ps1 -Scenario junction
.\Start_Indian_Traffic.ps1 -Seed 7 -ObserveRoad
.\Start_Indian_Traffic.ps1 -Weather wet -Quality Epic
.\Start_Indian_Traffic.ps1 -Weather dusk -Camera driver
```

Run one session at a time. The launcher refuses a second dedicated session. **Start_CARLA.cmd** remains the separate stock-map installation demo.

`-ObserveRoad` also records a fixed front RGB/depth/semantic camera independently
of the selected viewer camera. Read [the camera-observation guide](docs/CAMERA_OBSERVATION.md)
to run the locked model on the saved frames and replay its reference comparisons.

The speed row shows the **selected speed** and the **measured speed** over recent
active playback. For example, selected 3× and actual 2.4× means the PC is running
below the requested rate. Pauses are excluded from this recent measurement.
The fixed 0.05-second physics step, vehicle target speeds, controllers and safety
checks stay unchanged. Startup, screenshot writes and GPU/CPU capacity can reduce
the achieved rate. Playback speed carries through **R**; the default launch is 1×.

The next two rows explain the car's **current action** and **passing decision**.
For example, a junction wait shows the available distance alongside the distance
the existing policy requires. These explanations use the displayed camera frame's
decision. They do not change the driving rules. Read the
[decision display report](docs/DECISION_DISPLAY.md) for examples, logging and checks.

## What this scene represents

- A four-way, unsignalised neighbourhood junction with **actual left-hand OpenDRIVE roads** and all twelve turning/straight connections.
- Mixed cars, a bus, delivery van, scooter, motorcycle and cycles. Different vehicle sizes and target speeds affect gaps and queues.
- People walking on footpaths and requesting crossings at seeded times. In the mixed scene they first accept a gap; once crossing they continue walking and the vehicles must yield. The controlled pedestrian test deliberately requires a driver response.
- Slow-vehicle passing with checks for a clear approach, oncoming traffic, pedestrians, body clearance and the space needed to return.
- Bus stopping, a marked outer road edge, roadside kiosks/trees and a parking bay separated from the through lane.
- Compatible traffic may share junction reservations. Conflicting paths wait, and following vehicles retain their own headway checks.

## Evidence and limitations

Each new `runs/carla_india_*` folder retains the exact source snapshot, generated map, settings, actor list, per-frame actor trace, rendered screenshots, incidents, camera-frame alignment and cleanup results. `independent_audit.json` independently checks saved body rectangles and motion. Failed development runs are retained.

Read [the portable scenario report](docs/results/carla/SCENARIO_REPORT.md) for the current measured results and unresolved issues. The full local evidence remains under `setup/carla_india` and `runs/`.

This is an **illustrative Indian-style scenario**, not a calibrated Indian city or a claim of road safety. Building meshes are generic installed CARLA assets. There is no authentic autorickshaw asset yet, and weather, crowd density, traffic negotiation and city-scale traffic still need expansion and comparison with real footage.

Vehicles use CARLA physics, throttle, brake and steering. The controller is hand-coded and sees simulator positions; the camera is for observation. **The locked road model is not controlling the car and no RL policy is trained here.** The three saved road models, data splits and completed test evaluation are preserved.

The future RL scope remains STOP / SLOW / PROCEED above the route controller and independent safety checks. The current work establishes a measurable environment for that later stage.
