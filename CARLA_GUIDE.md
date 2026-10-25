# CARLA setup and first live scene

This guide describes the stock-map engine starter, separate from MATLAB references and road-model environments. Custom left-hand scenes and the connected town are described in [simulation controls](INDIAN_SIMULATION.md) and [the connected-town guide](docs/COMBINED_INDIAN_TOWN.md). See [getting started](docs/GETTING_STARTED.md) for a fresh setup.

## Current verification status

**Verified on 19 October 2026:** Vulkan, Epic graphics and stock Town10HD_Opt.
Seeds 7 and 42 each passed a 400-image / 20-simulation-second check with one ego
car, 12 surrounding vehicles and 10 pedestrians. All surrounding vehicles and
pedestrians moved; no vehicle collision sensor events occurred in these short
checks. Camera timestamps, driver/follow switching, actual world pause, screenshot,
Escape and owned-worker cleanup passed. Independent CSV/frame recounts agree.

See `setup/carla/review_20261019/RECHECK_REPORT.md`. These checks establish that
the starter works; they do not establish general driving safety or Indian-traffic
realism. Earlier failed tests and process-cleanup records remain local. These results describe the stock starter rather than the later custom town.

## Open the simulation

Double-click **Start_CARLA.cmd** in the project folder. It starts a dedicated CARLA server and opens a live camera window, for up to 10 minutes.

Use the window titled **Tesla project - CARLA live starter**. The separate Unreal
engine window is also part of this owned session.

- **Tab:** switch between follow-car and right-hand driver cameras.
- **Space:** pause/resume the world.
- **S:** save the current view into the session's output folder.
- **Esc** or close the camera window: finish and stop this session's server.
- No steering is required: CARLA's built-in Traffic Manager drives the car in this starter.

For a different seed or duration, run in PowerShell from this project:

```powershell
.\Start_CARLA.ps1 -Seed 7 -Seconds 600
# Lower graphics quality if other GPU work is running:
.\Start_CARLA.ps1 -Seed 7 -Seconds 600 -Quality Low
```

The launcher uses separate Windows Job Objects and dedicated ports 2050–2052 / 8050. It refuses occupied ports and duplicate sessions. It never attaches an existing MATLAB desktop or simulator. Its owned server tree stops when the launcher ends; other programs are not targeted. The live window has a 30-minute maximum wall-time limit.

## What is installed

- Official Windows CARLA **0.9.16**, using Unreal Engine 4.26.
- `.venv-carla`: separate Python 3.12 environment, with CARLA 0.9.16, pygame 2.6.1 and NumPy 1.26.4.
- Package under `tools/carla/0.9.16`; original ZIP under `setup/carla/downloads`.
- Vulkan renderer, Epic graphics, 960×540 RGB camera and a 0.05-second fixed simulation step. This is a 20 Hz simulation setting, not a throughput benchmark. The optional Low setting reduces shadows and draw distance; comparisons must record graphics settings.
- Stock Town10HD_Opt map, seeded traffic, nearby walking pedestrians, follow/driver cameras, collision sensors on every spawned vehicle, camera-frame records and traces for all actors.
- GPU images may arrive one simulation frame late. Each image is paired with its actual capture-frame state and timestamp; it is never labelled with a newer pose. Driver/follow labels also use the capture frame.
- Actor speed is measured from successive positions and timestamps. CARLA's native pedestrian velocity field was observed to return zero despite walking; the stored position trace is used instead.

Every run uses a unique `runs/carla_*` directory containing exact configuration, camera captures, ego trace, collision events and result/cleanup status. Seeds help repeat setup; deterministic equivalence has not yet been established.

## Important scope

**This starter is a working-engine check, not the completed Indian junction.** The stock map follows its own road rules; inspect `config.json` → `map_road_rules`. Do not call a right-hand stock town an Indian left-hand scene. A right-hand driver camera position does not change road rules.

CARLA autopilot controls this demonstration using simulator state. The trained road model, separate actor perception, MATLAB planner/controller and RL are not controlling the vehicle. Vehicle collision sensors are not a continuous geometric audit of every road user. Pedestrian navigation is CARLA's built-in model, not calibrated local crossing behaviour. This starter disables ego lane changes; sensible overtaking is a future controlled scenario, not a completed feature.

The later custom left-hand neighbourhood, selected passing/conflict scenarios and connected town have separate documented checks. Camera perception remains observation-only; destination and perception-driven control are subsequent integration work. See [results](docs/RESULTS.md) for the current scope.

## Installation recovery

If the official package download was interrupted, resume it; do not download another copy:

```powershell
curl.exe -L --fail --retry 4 --continue-at - --output setup/carla/downloads/CARLA_0.9.16.zip https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip
.\.venv-carla\Scripts\python.exe setup/carla/install_package.py
```

The installer checks the expected archive size, validates extraction paths, checks ZIP CRCs while extracting and records its SHA256. The SHA256 is locally recorded, not a publisher-authenticated checksum. It refuses to overwrite an unverified existing installation.

If a session fails, inspect `setup/carla/server_owner.json`, its `viewer_log` under
`setup/carla/logs`, and the newest `runs/carla_*` directory. Native engine crash
metadata is normally under `%LOCALAPPDATA%/CarlaUE4/Saved/Crashes`. Never stop all
Python/Unreal processes to recover. Lock files should only be removed after
confirming that their recorded session has ended.

The packaged Windows build ignored startup map URLs and command-line config
overrides. The launcher therefore backs up `DefaultEngine.ini` to
`setup/carla/DefaultEngine.original.ini` and sets four startup-map keys in this
installation only. DirectX 12 failed shader compilation and DirectX 11 had camera/
map instability on this laptop; Vulkan passed. Town03 was not revalidated with
Vulkan. The supported default is Town10HD_Opt; other maps remain experimental.

## Official references

- [CARLA 0.9.16 release and left-hand traffic support](https://carla.org/2025/09/16/release-0.9.16/)
- [Packaged installation](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
- [Synchronous simulation and fixed time steps](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/)
- [Rendering modes and graphics quality](https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/)
