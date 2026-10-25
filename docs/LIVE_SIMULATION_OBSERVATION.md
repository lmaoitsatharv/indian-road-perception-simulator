# Live simulation road observation — 23 October 2026

The simulation model now runs on the native CARLA front-camera stream while the simulator is active. A separate GPU service supplies frame-aligned masks to the preview. The rule driver, real-image/video inference and all checkpoint weights remain unchanged. No training or RL occurred.

## Start the preview

Double-click **Start_Live_Road_Preview.cmd** in the project folder. It loads the existing WSL GPU runtime and starts an owned Indian-style left-hand scene. Allow model/simulator startup to finish. The default scene is mixed traffic, seed 4101, clear weather, 1x playback and Low graphics, bounded to 110 simulation seconds. The scene uses the already verified new low-rise assets and 10 m junction half-spacing.

The top-left panel shows the current front camera. The top-right panel shows the latest admissible road mask on **its own matching source image**, with source-frame number and age. Green means predicted road. Waiting/stale/offline states replace that overlay with a blank labelled panel. The small lower-left image retains the existing follow/driver/junction view and the lower HUD retains the existing driving explanation. SPACE pauses/resumes; S saves a screenshot; ESC closes the owned simulator and GPU service.

PowerShell options, when wanted:

```powershell
.\Start_Live_Road_Preview.ps1 -Seconds 60 -Scenario mixed -Weather clear
.\Start_Live_Road_Preview.ps1 -Seconds 40 -Check
```

`-Check` enables explicit delay/disconnection/pause tests and a matched no-model control comparison. Normal launch does not inject faults. Existing `Start_Indian_Traffic` behavior remains available separately.

## Measured normal operation

The final normal run received **100/100 results**, with **zero reconnects**, in 20.0 simulated seconds. Camera sampling was 5 Hz. No over-age mask was admitted at a completed redraw. All native and GPU worker processes and the transport thread closed cleanly.

**Display timing limitation:** 2 intervals held the previous overlay beyond the 500 ms age target, by up to 14.87 ms. The next redraw replaced it with a fresh result or a blank stale panel. A synchronous simulator/UI operation can postpone that redraw. The per-redraw freshness gate passed; continuous on-screen deadline compliance did not. This preview is suitable for observation, with that measured limitation.

| Timing | Median | 95th percentile | Maximum |
|---|---:|---:|---:|
| Source-step start → result received (age upper bound) | 60.36 ms | 145.49 ms | 278.18 ms |
| Camera callback → result received | 42.01 ms | 87.13 ms | 125.14 ms |
| GPU inference stage | 17.31 ms | 22.79 ms | 29.73 ms |
| Source-step start → first preview presentation (upper bound) | 120.11 ms | 242.73 ms | 311.63 ms |
| Source age across all fresh preview redraws (upper bound) | 215.90 ms | 334.25 ms | 401.03 ms |

Later redraws may reuse a fresh result until a newer one arrives, so their source age differs from the latency to the first display of each unique result. All absolute timestamps used in these age differences originate from the same Windows monotonic clock. Linux timestamps are never subtracted from Windows timestamps; the worker uses its own clock only for local stage durations.

CARLA does not provide an exact camera-exposure wall timestamp in this pipeline. The timestamp immediately before the source physics step is a conservative earlier bound, including sensor delivery, transport, resize and inference. These are measured age bounds, not exact exposure latency or a hard real-time guarantee. Prolonged OS/GPU/UI stalls can delay the next redraw.

## Freshness, bounded buffering and recovery

There is at most one in-flight request and one pending frame. A newer pending frame replaces the older one instead of growing a queue. Received frame/session identities and pixel hashes must match. Raw RGB is resized bilinearly to 512x512 and normalized once; probability threshold remains 0.5. Every mask is rendered on its matching resized source RGB, never a newer camera frame.

Freshness gates apply a 500 ms Windows wall-age bound and 500 ms of simulated age. The final display uses a 100 ms presentation reserve, so it stops admitting an older mask at 400 ms rather than waiting until its last permissible instant. The independent audit checks both completed presentations and intervals for which the previous overlay remained on screen. This is a redraw-based policy, not a hard-scheduling guarantee. Pausing the simulated world does not freeze the wall-age clock.

The 40 s controlled run received 196/200 results. Its 0.9 s delay produced an over-age result that was hidden; its deliberate TCP disconnection produced an offline panel and recovered automatically; the 1.2 s pause also aged the mask out while the world frame stayed fixed. 3 pending frames were replaced and the only connection error was the deliberate disconnection. The fault run used the earlier 50 ms presentation reserve; the final normal run verified the stricter 100 ms reserve and retained its measured display overruns. The transport's 3 s timeout never extends the 500 ms overlay admission gate.

The model runtime reserves 1 GiB of GPU memory to leave room for CARLA. Earlier 2 GiB/short-idle-timeout prototypes showed unnecessary reconnects and GPU stalls; they remain preserved under runs01–04. Run 06 had no reconnects but a held-screen audit found a 15.67 ms tail beyond the nominal deadline on three intervals. Run 07 used the larger reserve; its separate held-screen result is reported above. Increasing a timing reserve alone cannot guarantee redraw scheduling.

## Controller and model isolation

The final fault test matched **15,219 actor/control records** from the same-seed no-model baseline exactly: positions, heading, speed, route position, throttle, brake, steering, targets, decision reasons and pass states all matched. A repeated no-model baseline also matched exactly. This is evidence for this 40 s scene, not universal simulator determinism.

The controller source is byte-identical to the protection snapshot. Its call still receives only `snapshot` and `sim_time`. The preview has no actuator calls, and no image/mask is supplied to the driver. Model weights, real deployment/catalog/split metadata and the old recorded-only simulator loader remain unchanged. The new live-observer manifest has its own native-CARLA observation-only scope; neither recorded nor live scope grants real deployment or vehicle control.

An independent script reopened each actual captured camera image and verified the source bytes, exact resize, prediction hashes, saved probability thresholds, native reference masks, all timing arithmetic and source-frame identities. It confirmed stale/offline suppression and recovery. The source scene/seed was already inspected; any saved road-IoU sanity score is not a new independent held-out test result. The completed 500-image test and 480-frame confirmation remain their own reports.

## Evidence

- Final normal run: `runs/simulation_live_20261023_07/`.
- Controlled fault/control comparison: `runs/simulation_live_20261023_05/`.
- Each run has `independent_verification.json`, `worker_frames.jsonl`, worker/capture status and model manifest.
- Native episode `live_road/` contains receipts, source ticks, every display event and transport-error logs.
- Model masks and selected probability arrays remain beside the worker logs.
- Portable result: [results/live_simulation_observation.json](results/live_simulation_observation.json).

Normal fresh view:

![Matching-frame live road preview](results/overview/live_road_preview.png)

This portable capture is copied from the normal run. Stale/offline diagnostic captures listed below remain local.

Controlled stale-result suppression:

Stale road mask hidden: `runs/simulation_live_20261023_07/screenshots/stale.png` (local artifact, supplied separately).

Controlled service-disconnection state:

Disconnected observer: `runs/simulation_live_20261023_07/screenshots/offline.png` (local artifact, supplied separately).

The first new-seed attempt (run01, seed4601) failed during pedestrian spawning before inference. That failed capture is retained; the latency/controller checks use the previously verified seed4101. Earlier prototype timing/control differences are also retained, rather than overwritten.

## Following work

If strict continuous display expiry is required, decouple preview refresh from blocking simulator calls and evaluate OS/GPU stalls explicitly. The current per-redraw expiry policy is measured above.

Add independently evaluated vehicle/pedestrian detection and tracking in observation-only mode, including missed actors, timing and stale-track behavior. Road masks alone do not provide object identity or motion. Any later planning/controller/RL use remains a separate step.
