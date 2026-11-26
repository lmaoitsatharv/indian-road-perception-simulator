# Camera-only driving in the connected town: design for review (6 November 2026)

Status: **design for review. Nothing is implemented yet.** After approval, an implementation plan follows, then the build.

## 1. What was asked and what was chosen

Goal: the trained models drive your car around the connected Indian town in CARLA.

Choices made on 6 November:

- **Pure camera driving.** Steering comes from the road model's mask and speed from the detector. There is no map route and no
  ground-truth safety backstop. At junctions the car goes "whichever way the mask leads".
- **Spec first.** This document is reviewed before any code is written.

Today the car is a rule-based `TownDriver`. It reads the exact positions of every other actor and follows a fixed map route,
and the HUD says "no trained model in control". The road model has only ever run live as a display, and the detector only offline.

## 2. What your car may and may not use

| May use (sensors a real car has) | May not use (simulator knowledge) |
| --- | --- |
| One front camera image | Positions, speeds or boxes of other actors |
| The road model's mask of that image | The OpenDRIVE map, lanes, waypoints, routes, junction locations |
| The detector's boxes on that image | Junction tokens or other vehicles' intentions |
| Its own speed (speedometer) and yaw rate (gyroscope) | Depth, semantic or instance cameras |
| Its own steering and pedal commands | Any ground-truth emergency brake |

The driver lives in its own module, which receives only the items in the left column. It has no handle to the world, the map
or the traffic. A check script verifies this (section 10).

The simulator itself keeps using ground truth for things that are not your car's decisions: moving the other 24 vehicles and
the people, scoring the run, the minimap and recovering a stuck car (section 7).

## 3. Measured starting point

Both models were run offline on **120 frames saved from the town**: run `carla_indian_worlds_live_20261106T101542_611447Z_seed8101`,
960x540, field of view 95 degrees, covering all five districts and 15 junction frames. Evidence and scripts are in
`runs/camera_driving_probe_20261106_01/`.

The truth was built separately, because the semantic camera tags the generated road as "Building" in this town. Each pixel's 3D
point (depth camera + camera pose) is looked up in the run's own OpenDRIVE map: a point on a driving lane surface counts as road,
shoulders are ignored, and lane paint counts as road.

**Road model** (simulation U-Net `8d8c9fd7`, threshold 0.5, 18.8 ms median on the GPU):

| | Road IoU | Precision | Recall |
| --- | ---: | ---: | ---: |
| All 120 frames | 0.734 | 0.862 | 0.832 |
| Inside junctions (15) | 0.690 | 0.918 | 0.735 |
| Village roads | 0.786 | 0.894 | 0.867 |
| Divided main road | 0.699 | 0.860 | 0.788 |

This is lower than its 99.7% on the neighbourhood scenes it was trained on, because the town is a different place.
What matters for steering is where the road edges land on the ground ahead:

| Edge on the ground, 6–14 m ahead | Market, colony, irregular-junction streets | Village roads | Divided road |
| --- | --- | --- | --- |
| **Left edge (your kerb)** within 0.5 m | 86–100% of frames, typically 0.2 m too far left | 35–75% (shoulder and sand confusion) | 57% at 6 m, about 10% at 10–14 m (misses the outer lane) |
| **Right edge (far kerb)** within 0.5 m | 0–57%; the oncoming lane is often missed by about 2.4 m | about 50% | 93–97% |

Conclusion: **the left kerb is usable for lane keeping on most streets. The right edge is not.** The design steers from the left
edge. On the divided road the car will mostly keep to the inner lane, and on village roads it may ride about 0.3–0.4 m left of the
lane centre.

**Detector** (deployed `f7bb7e4c`, NMS 0.6, confidence 0.25, FP16). Recall against the instance-camera truth, objects of at least 150 px:

| Input size | GPU time | 0–10 m | 10–20 m | 20–40 m | People found | Cars found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 960 | 13.1 ms | 93% | 92% | 80% | 51/65 | 23/27 |
| **1280** | 16.7 ms | **97%** | **97%** | **90%** | 58/65 | 25/27 |
| 1536 (recorded IDD setting) | 20.3 ms | 95% | 97% | 80% | 57/65 | 21/27 |

Every bicycle, rider and motorcycle within view was found at 1280 (bicycles 17/17, riders 60/60, motorcycles 33/34).

Distance from the bottom of a box, assuming flat ground: median +7% to +17% too far, 90th percentile about 20–29% off.
It reads too far because a box bottom sits at the wheels, behind the bumper.

Unmatched boxes ran at about 1.6 per frame. Many are real things the instance camera does not label, such as the standing-people
props in shop forecourts and distant riders. Their effect on driving is limited because only objects near the car's path matter.

These timings were measured with CARLA closed. Live timings will be measured in the first run.

## 4. How it works

```
CARLA (Windows, synchronous, 20 Hz physics)
  front camera 960x540, 95 deg, 10 Hz  (same mount and size as the road model's training camera)
        |  raw RGB over loopback TCP, session token
        v
  WSL GPU:  road worker (idd-tf220, U-Net 512x512)   detector worker (idd-detect, YOLO11s 1280)
        |  road probability 512x512                   boxes, classes, confidences
        v
  camera driver (new, plain Python + NumPy, no simulator access)
    1. ground grid ("bird's-eye view") from the mask using the fixed camera geometry
    2. left-kerb lane reference -> planned path
    3. detected objects placed on the ground -> obstacles near the path
    4. speed decision: PROCEED / SLOW / STOP with a reason
    5. steering (pure pursuit on the path) + throttle/brake (speed controller)
        |
        v
  VehicleControl for your car;  the other 24 vehicles keep their rules
```

### 4.1 Camera and timing

- One RGB camera at x 1.35 m, y 0.25 m, z 1.45 m, pitch 0, 960x540, field of view 95 degrees, 10 frames per second. This is the
  town's existing evidence-camera mount and lies inside the road model's training range (heights 1.25–1.50 m, fields of view 90–100 degrees).
- **Fixed 100 ms perception latency.** A frame captured at tick *n* is sent to both workers at once. Its result steers the car
  from tick *n*+2 onwards. The simulation waits for the result if it is late. A slow GPU then slows the simulation clock, never the
  car's reaction time, and runs are repeatable.
- Between frames, the last planned path is carried forward using the car's own speed and yaw rate (dead reckoning).

### 4.2 GPU workers

- **Road worker**, in the `idd-tf220` environment, runs the simulation road model `runs/simulation_frozen_20261022_01/best.keras`
  (`8d8c9fd7`). The input is resized to 512x512, exactly as the existing live worker does. It returns the probability map as 8-bit.
- **Detector worker**, in the `idd-detect` environment, runs `runs/detection_train_20261025_02/weights/best.pt` (`f7bb7e4c`) at
  1280 input, NMS 0.6, confidence 0.25, FP16 (the setting measured above). It returns boxes only. Ultralytics settings live in the
  run folder, with online features off, as the other detector scripts do.
- Each worker checks its checkpoint hash, loads once and never trains. Each listens on its own loopback port with a per-run session
  token. The frame protocol reuses `simulation_live_protocol.py`.
- New scope manifest for the road model: `simulator_camera_driving_v1`, with `control_allowed: true`. It is loaded by a new
  `idd_driving_inference.py`. The existing observation-only loader and its `control_allowed: false` scope stay untouched. The
  detector gets a matching manifest.

### 4.3 Ground grid from the mask

The camera is fixed to the car and the town is flat (about z = 0). So each ground cell maps to one pixel, worked out once at startup.

- The grid uses 0.1 m cells, from 2.5 to 25 m ahead and 10 m to each side.
- Road probability per cell is looked up from the 512x512 map. Cells outside the image are "unknown".
- Cost: a lookup of about 50,000 cells per frame, well under a millisecond.

### 4.4 Lane keeping: follow the left kerb

India drives on the left. The probe shows the left kerb is the reliable edge.

1. For each 0.5 m row from 4 to 14 m ahead, scan left through connected road to the first non-road cell. The scan starts from
   the previous path, carried forward by dead reckoning, or from straight ahead at the start. Gaps of up to 0.3 m are bridged,
   for lane paint and mask specks. This gives the left edge `yL(x)`.
2. Target line: `y(x) = yL(x) + 1.5 m`, placing the car's centre half a town lane (lanes are 2.75–3.25 m) right of the kerb.
3. The rows are fitted with a smooth curve (quadratic in x):
   - Weights are 1.0 for 4–9 m and 0.4 for 9–14 m, because far rows are where the divided road fails.
   - Rows more than 0.8 m off the fit are dropped.
   - The result is blended with the previous frame's path (moved by dead reckoning), so one bad mask cannot jerk the wheel.
4. Clearance check: the car's body (half width plus 0.3 m) must stay on road cells along the path. If not, the path shifts
   right where road allows, or the car slows (section 4.6).
5. If the left edge cannot be found for more than 1.5 s, the car follows the free road in front at 2.5 m/s. If that also fails,
   it stops with the reason "Lost the road".

Things in the road are not road in the mask. A two-wheeler at the kerb, a stopped bus or a parked car therefore moves the
left edge to the right, and the car passes around them if the road on the right is free. An oncoming vehicle narrows the road
on the right. Passing behaviour comes from the same rule and is not scripted.

### 4.5 Junctions: "whichever way the mask leads"

There is no map, so the car does not know a junction exists until the road opens up in the mask. It counts as a **wide road
area** when any row 5–12 m ahead has more than 9 m of connected road, or a left edge more than 4.5 m from the path. There the
car slows to 3.5 m/s. The road counts as **uncertain** when the kerb fit used fewer than 6 rows.

Following the left kerb through an opening takes the **leftmost exit**, the easy turn in left-hand traffic. In this town that
should give a repeating loop: Market junction E, Bazaar gate A, Village gate B, Irregular junction C, Residential square G, back to E.
- It is about 1.5 km and passes through four of the five districts.
- It skips the C–D street, the irregular-junction district.
- This loop is a prediction from the road layout and will be confirmed in the first run.
- The car never crosses the divided road's median, because that road joins only at the bends A and B.

Only the front camera exists, so traffic coming in from the sides is not seen until it is in front. Section 7 covers how the
other vehicles avoid driving into your car.

### 4.6 Obstacles and speed (PROCEED / SLOW / STOP)

Each box becomes a ground footprint:

- **Position:** the bottom centre of the box on flat ground, made 0.8 m closer for cars, vans and buses (the bumper overhang),
  0.3 m closer for two-wheelers and unchanged for people.
- **Width:** from the box width at that distance.
- **Length:** by class: car 4 m, two-wheeler 1.8 m, person 0.6 m, bus 10 m, truck 7 m.
- Boxes are kept for 0.3 s after their last sighting, so a single missed frame does not release the brake.

The speed target is the lowest of these:

| Rule | Value |
| --- | --- |
| Cruise | 6.0 m/s (about 22 km/h) |
| Wide road area (junction) or uncertain road | 3.5 m/s |
| Left edge lost | 2.5 m/s |
| Curvature | `sqrt(2.0 / curvature)`, at most 2 m/s² sideways |
| Obstacle on the path (body + 0.3 m vehicles, + 0.8 m people) at distance d | `sqrt(2 * 2.5 * (d - 4))` (5 m stop gap for people); full brake under 2.5 m |
| Person within 1 m of the path and closer than 15 m | 2.5 m/s |
| Road ends on the path at distance d (mask not road for the car's width) | treated as an obstacle at d |
| Perception result missing or worker offline | brake to a stop, reason "Perception unavailable" |

The decision shown is PROCEED (target at least 90% of cruise), SLOW or STOP, with the rule that set it. Examples:
"Pedestrian 9 m ahead", "Vehicle 5.2 m ahead", "Road ahead not visible".

### 4.7 Control

- **Steering:** pure pursuit to the path point 5–11 m ahead (4 m + 0.5 s of travel), using the car's wheelbase and maximum steer
  angle. It is rate-limited like the existing driver (0.07 per tick).
- **Throttle and brake:** the same PI-style speed controller as `Driver.control`, copied into the new module so the rule driver
  stays unchanged.

## 5. What you see

The normal town viewer opens with every existing key.

- **New "AI view" panel:** the camera picture with the green road overlay, boxes, the planned path and the kerb points.
  Boxes are red when braking for them, amber near the path and grey when ignored.
- **Decision line:** for example "SLOW · pedestrian 9 m ahead · target 9 km/h".
- **Timing line:** for example "frame 1234 · latency 100 ms · road 19 ms · detector 17 ms".
- **A** toggles a full-screen AI view. The subtitle reads "Camera-only AI driver · road model + detector in control".
- The minimap and areas panel stay; they are display only.

## 6. What gets recorded

The run folder is `runs/carla_indian_worlds_*` as today. A new `camera_driving/` subfolder holds:

- `decisions.jsonl`: one line per perception frame with timings, detection and road counts, edge fit, path, decision,
  target speed and steering.
- `views/`: the AI-view image once per second for review.
- `session.json` and the manifests for both workers, with their logs.

`result.json` gains a `camera_driving` block, scored by the simulator. Your car's driver never sees these:

- distance driven, districts visited, the loop actually taken
- collisions involving your car (with what, and how hard)
- time on a driving lane, time off the road, and time on the wrong side outside junctions, where your heading opposes the
  lane under the car
- median and 90th-percentile distance from the centre of the nearest correct-direction lane
- stops, the longest stall, resets
- perception frames, latency, worker errors, and the real-time factor achieved

`passed` keeps meaning "the software ran correctly and the other traffic stayed healthy". Your car's crashes and off-road time
are driving results reported in `camera_driving`. They do not decide `passed`.

## 7. Changes outside your car

These are needed so the other road users react to a car they cannot coordinate with.

- **Out of the junction-token system.** Your car cannot request tokens without a map. It is excluded from token queues and grants.
- **Other vehicles yield to your car** (proposed; see decision 2). Each other vehicle checks whether your car's true body lies
  within its own half width plus 0.4 m of its route ahead. It looks clip(6 + 2.5 × its speed, 6, 25) m past its front, including
  through junctions and towards oncoming traffic. If so, it applies its normal following-gap braking.
  - Today they yield to each other through tokens and lane-based checks. Neither of those covers your car any more.
  - This gives your car no information.
  - It is logged as `yield_to_camera_car`.
- **Distance, district and direction** for your car are worked out from its true position on the map for scoring only. The
  route-progress counter assumes the fixed tour, which the camera car will leave at C.
- **Recovery** (proposed; see decision 3). After 30 s with no progress, for example after a crash, a lost road or a deadlock, the
  simulator places the car stopped on the nearest lane ahead that is clear. It counts as a reset, the same as pressing R. It never
  brakes or steers for the car.

## 8. Failure handling

| Situation | Behaviour |
| --- | --- |
| A worker fails to start | The run stops before the town is built; the logs are kept |
| A worker disconnects, or a result is not back within 1 s (wall clock) | The simulation stops waiting and carries on with perception marked unavailable, so the car brakes to a stop ("Perception unavailable"). The client reconnects once a second, and every gap is counted in the result |
| Perception unavailable for 10 s of wall time | The run ends with a recorded error |
| Empty mask or no left edge | Section 4.4, step 5 |
| Collision | Recorded; the car keeps driving if it can |
| Stuck for 30 s | Recovery reset, if enabled (section 7) |
| ESC, window closed or wall limit | Workers stopped, sensors destroyed, protected files checked |

## 9. Launch

```powershell
.\Start_Camera_Driving.ps1                      # 600 simulated seconds, seed 8101, clear weather
.\Start_Camera_Driving.ps1 -SimulationSeconds 300 -Weather overcast -NoReset
```

`setup/run_camera_driving.py` owns one bounded session:

1. Writes the manifests and session token.
2. Starts both WSL workers and waits until they are ready.
3. Runs the existing `setup/carla/run_session.py --india-world combined --camera-drive <session.json>`.
4. Stops the workers.
5. Confirms the protected model files and the rule driver are unchanged.

The stale `setup/carla/session.lock` from 16:07 today, whose processes are no longer running, is removed before the first run.

## 10. Testing

1. **Offline checks**, `setup/verify_camera_driving.py` (no CARLA or GPU):
   - Camera geometry round trip: a ground point to a pixel and back, within 2 cm.
   - Synthetic road masks drawn with the same camera model, for a straight road, a curve, a left opening, a T-junction and a
     road narrowed by an obstacle. The left edge must be found within 0.1 m, and the path must turn or stop correctly.
   - A closed-loop bicycle-model drive along synthetic roads stays within 0.3 m of the lane centre. This catches steering-sign errors.
   - Speed rules: stop for an obstacle at 4 m, slow for a person near the path, brake when perception goes missing.
   - Protocol round trip with stand-in workers.
   - A source check that the driver module never reaches the world, the map or the traffic.
2. **Replay on the 120 recorded frames:** saved masks and boxes are fed through the driver. Its target line is compared with the
   true lane centre from the map, with target medians of 0.4 m or less on market, colony and junction streets, and village and
   divided roads reported separately.
3. **Live runs:** a 120 s smoke run covering startup, both workers, the HUD and clean shutdown, then 600 s runs. Screenshots are
   reviewed as a viewer would see an Indian street, and failures are fixed within this design.

**Done means:** in a 600 s run the car drives only on model output and completes at least one full loop (about 1.5 km) with no
reset. All the measures in section 6 are reported, including the bad ones. If tuning within this design cannot get there, I stop
and report what failed and the options. One option is fine-tuning the road model on town frames, which would need your approval.

## 11. Decisions for you

1. **Detector setting for driving: 1280 input, NMS 0.6, confidence 0.25.** The recorded IDD setting is 1536 / 0.6. On town frames
   1280 found more actors (cars 25/27 against 21/27) and is faster. The recorded setting in `models/perception_models.json` is not
   changed; the driving manifest records its own setting.
2. **Other vehicles yield to your car** using their own view (section 7). Without it, they drive into it at junctions.
3. **Stuck recovery after 30 s**, on by default and counted as a reset; `-NoReset` turns it off. It is not a safety brake.
4. **Turning: follow the left kerb**, which takes the leftmost exit. The alternative, choosing among exits that the mask shows,
   needs branch detection and is not in this design.
5. **Speeds:** 6 m/s cruise, 3.5 m/s in wide road areas.
6. **The evidence cameras stay** (RGB, semantic, instance and depth at 5 Hz, saved at 1 Hz). They let the run be scored
   afterwards, at some cost in speed. They can be switched off if the live rate is poor.

## 12. Not in this design

- Training or fine-tuning any model
- Side or rear cameras
- Object tracking beyond the 0.3 s hold
- Destinations or route choice
- Traffic lights (the town has none)
- RL
- Changes to the other vehicles' rules beyond yielding to your car
- The older neighbourhood scene and the live road preview

## 13. Files

| New | Purpose |
| --- | --- |
| `integration/india_camera_geometry.py` | Camera model, ground-grid lookup, pixel and ground conversions |
| `integration/india_camera_driver.py` | Ground grid, left-kerb path, obstacles, speed decision, controller (no simulator access) |
| `integration/india_perception_client.py` | Sends frames to both workers and collects results; handles timeouts and offline workers |
| `idd_driving_inference.py` | Road-model loader for the new driving scope |
| `setup/camera_driving_road_worker.py`, `setup/camera_driving_detector_worker.py` | The two WSL GPU workers |
| `setup/run_camera_driving.py`, `Start_Camera_Driving.ps1` and `.cmd` | Session owner and launchers |
| `setup/verify_camera_driving.py` | Offline checks and recorded-frame replay |
| `docs/CAMERA_DRIVING.md` | Results report after the runs |

| Changed | Change |
| --- | --- |
| `integration/india_combined_scene.py` | `--camera-drive` option, perception camera, fixed-latency loop, `camera_driving` result block |
| `integration/india_combined_traffic.py` | Override hook for your car, no tokens for it, other vehicles yield to it, true-position scoring, recovery reset |
| `integration/india_town_hud.py` | AI-view panel, decision and timing lines, **A** key |
| `setup/carla/run_session.py` | Accepts `--camera-drive` with `--india-world combined` |
| `docs/FILE_GUIDE.md`, `docs/INDEX.md` | Entries for the new files |

The connected-town files already carry uncommitted changes from 5–6 November (scenery, HUD, traffic). This work builds on top
of them and does not commit anything unless asked.
