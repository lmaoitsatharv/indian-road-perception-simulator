# Camera-only driving in the connected town

**What this is.** Your car in the connected Indian town is now driven only by the two trained models looking through one
front camera:

- The **road model** (simulation U-Net `8d8c9fd7`) finds the road, and the car keeps 1.5 m right of the left kerb.
- The **detector** (deployed IDD YOLO11s `f7bb7e4c`) finds vehicles, riders and people, and the car slows, stops or
  passes them.
- Around passes only, a **rear camera** goes through the same detector, so the car can look behind before pulling out
  and while backing up.
- **Side cameras** at the door pillars look straight out to the left and right. With the front and rear ones they see
  all round the car. The right one runs around passes (the passing lane beside the car), both while turning round
  (the street then runs past the car's sides). Each frame sends one of the rear and side pictures to the detector, in
  turn, so its load stays that of the rear camera alone: during a turn each extra view refreshes about 3 times a
  second, during a pass the rear and right ones about 5 times.

There is no map, no route and no simulator knowledge of other actors, and nothing brakes for the car. The other 24 vehicles
and all the people still follow their rules. Design: [CAMERA_DRIVING_DESIGN_20261106.md](CAMERA_DRIVING_DESIGN_20261106.md).
Everything here is simulation evidence from one seed in clear weather.

## Run it

Double-click **Start_Camera_Driving.cmd**, or in PowerShell:

```powershell
.\Start_Camera_Driving.ps1                         # 600 simulated seconds, seed 8101, clear weather
.\Start_Camera_Driving.ps1 -SimulationSeconds 300 -Weather overcast
.\Start_Camera_Driving.ps1 -NoReset                # never lift a stuck car back onto its lane
.\Start_Camera_Driving.ps1 -Check                  # run as fast as the PC allows
.\Start_Camera_Driving.ps1 -Destination "Colony junction"   # drive there (a place name, or x,y)
```

It takes about 3 minutes to load both models in WSL and build the town. Then the usual town viewer opens:

- The panel on the left, **WHAT THE AI SEES**, shows the camera picture with:
  - green: the road model's road
  - white dots: the left kerb it follows
  - cyan: the path it plans
  - boxes: red when braking for it, amber when near its path, grey when ignored, and thin grey "ignored ... (road visible
    under it)" for detector boxes that the road model contradicts
- While the rear and side cameras run, their pictures sit along the top of the AI view, as mirrors would (left, rear
  in the middle, right), with what the detector found there (amber; metres behind, or metres away for the sides). The vehicle being passed is drawn blue, and a line at the bottom says why a pass is waiting or how
  it ended.
- Under the picture: the decision (PROCEED, SLOW or STOP, with the reason), then the time each model took.
- **A** switches to a full-screen AI view. Every other key works as before; **R** lifts the car back onto a lane.
- **M** opens the town map to pick a destination (see "Picking a destination" below); **X** clears it.

Each run leaves:

- `runs/camera_driving_*/`, which holds both model manifests, the session settings, the worker logs and `status.json`.
- The town's usual `runs/carla_indian_worlds_*/` folder, with a `camera_driving/` subfolder:
  - `decisions.jsonl`: every camera frame
  - `views/`: the AI view once a second
  - `collisions.jsonl`
- `result.json` → `camera_driving`, which scores the drive using the simulator's ground truth. The driver never sees it.

## What the car may use

Only its own camera images through the two models, its speedometer and gyro (for dead reckoning between frames), its own
gear, and its own size and steering limit. `setup/verify_camera_driving.py` checks the driver's source has no route to the simulator, the map
or other actors.

With a destination it also gets a sat-nav, which knows its exact position and the road map (see "Picking a destination").
The sat-nav plans the route and tells the driver which way to go; it never steers on a street.

Two things in the simulator use ground truth on the car's behalf:

- **Other vehicles give way to your car** when its true body is in their path. Without a destination the car has no map,
  so it cannot join the junction turn-taking the other vehicles use. With one it does, and this rule stays as a safety net.
- **Stuck recovery** lifts the car back onto its lane after 30 s without progress, or 90 s when it is queuing behind a
  vehicle. Each one is counted as a reset.

## Picking a destination (8 November)

Design: [CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md](CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md). Plan:
[CAMERA_DRIVING_DESTINATIONS_PLAN_20261108.md](CAMERA_DRIVING_DESTINATIONS_PLAN_20261108.md).

**How to use it:**

- **M** in the viewer opens a town map. Click any street, or one of 11 place names: the 7 junctions and the bus stop,
  village road, colony street and market street.
  - The click snaps to the lane on the side you clicked, and a cyan route and a pink pin appear.
  - A click more than 15 m from a street is refused ("Not on a street").
  - **M** or **Esc** closes the map, and **X** clears the destination.
- **Start with one:** `.\Start_Camera_Driving.ps1 -Destination "Colony junction"`, or `-Destination "120,-45"`.
- **Test list:** `-Destinations "A;B;C"` sets the next place 5 s after it arrives at each one.
- **On screen:**
  - **Minimap:** the route and the pin.
  - **AI panel:** "To Colony junction · 640 m · next: right at Irregular junction in 80 m", then "Arrived".
  - **AI view:** the planned junction path, drawn in magenta.
- **On arrival** it slows to 3 m/s within 30 m, moves closer to the kerb and stops at the pin. It then waits for the next
  destination.

**How it works:**

- **Sat-nav** (`integration/india_satnav.py`):
  - It takes the car's exact position and plans on the town's lane graph (`integration/india_route_planner.py`).
  - Routes keep to the kerb lane. A destination on a two-way street can be reached on either side of it, and a route
    may turn round (a U-turn) on a two-way street, 25 m or more from its junctions, when that saves going round the block.
  - It re-plans after a wrong turn or a reset.
- **Between junctions:** the camera driver follows the left kerb exactly as before.
- **Turning round:** it stops where the route turns, waits for a gap in traffic, then turns on full lock forward and
  back between the road's edges (from the sat-nav's map) until it faces the other way. Stalls, parked vehicles and
  people that stay off the road do not hold it up. After 12 moves or 80 s it gives up and the sat-nav re-plans without
  U-turns. A rear box nearer than 1.5 m counts only if something was seen coming up behind first: phantom boxes over
  the rear picture held the backing moves. Across the street, traffic along it is beside the car, where only the side
  cameras see it: a move stops where it is while anything on the road within 20 m comes towards the car at 1.2 m/s
  or more ("car coming on the left, 12 m"), until it has stopped or gone by. The first CARLA run (18 Nov) stuck on a
  stall behind. After the fixes in the destinations design doc, the second turned round in 6 moves. The phantom rule
  and the side cameras have not been run in CARLA yet.
- **Wedged:** with the throttle on and the car not moving for 2 s, something the cameras do not show is holding it
  (a kerb). It backs off 1.5 m the way it came, with the rear camera's checks, and drives on; at most 3 times a
  minute, then it stops and waits. During a turn such a move ends there instead, as at the road's edge. 26 Nov: it
  cut the inside of a sharp left at Market junction and pushed on a kerb for 29 s, until the town's stall reset.
- **At junctions:** from 20 m before a junction until the car is back on a street, the driver steers by a Hybrid A* path
  (`integration/india_hybrid_astar.py`):
  - **What it plans on:** the camera's top-down road grid.
  - **Where it aims:** the farthest point of the route the camera can see where the whole car fits. A vehicle or person
    on the route ends the search, so the car follows it rather than planning round it.
  - **How often:** it re-plans every 0.5 s.
  - **When the camera cannot see the turn:** when no path is found, or the only goal in view is under 6 m away, it
    follows the route's own line instead, at 2.5 m/s ("junction_fallback").
- **Turn-taking:** with a route the car joins the junction reservations and waits about 2 m short until it is given its
  turn. `navigation.jsonl` records why it is waiting.
- **Records:**
  - `camera_driving/navigation.jsonl`: every destination, route, junction, re-plan and arrival
  - `screenshots/destination_map_*.png`: the map each time a route is planned
  - `result.json` → `camera_driving.navigation`: destinations reached, re-plans, junction turns (planned or fallback
    frames), plan times and waiting time

**Offline checks:** `setup/verify_camera_destinations.py` (run with `.venv-carla`) passes 66 of 66. These cover:

- **Routes:**
  - every place reaches every other; it turns round only by a U-turn on a two-way street, 25 m or more from a junction
  - clicks snap to the clicked side, and clicks on a bend's arcs or a junction connector to that connector's own lanes
  - a pin is reached on either side of a two-way street (the 18 Nov case: 1836 m before, 677 m now)
  - a destination 40 m behind the car is a U-turn (68 m), not round the block (1499 m)
- **Turning round** (a kinematic car that pivots about its rear axle):
  - a three-point turn on a 6 m street, the body inside the road; more moves with a 35 degree steering lock
  - an oncoming car is waited for; a car parked 25 m ahead does not hold the turn up
  - a stall on the pavement beside the turn does not stop the backing move; a person on the pavement is not waited
    for, one in the road is; a bus standing straight behind is not taken for one overtaking
  - phantom boxes over the rear picture ("bus 0 m", "autorickshaw 0.7 m behind") do not hold the turn; a car seen
    coming up behind still counts at 0.5 m, and so does one already there when the rear camera starts
  - with the side cameras: a car coming along the far lane mid-turn holds the move until it stops; a car parked 14 m
    along the far lane does not
  - a kerb the map does not show, 0.5 m inside its road edge: the move ends at it and the turn still finishes
- **Hybrid A\*:**
  - left, right and straight-on turns at a synthetic crossroads in about 30-50 ms each, round a stopped car, and a
    walled-off exit
  - the same turns with the town's 2.8 m lanes and 3% stray not-road pixels in the road
  - with a vehicle ahead on the route, the goal stays behind it with room for the whole car
- **Closed loop:**
  - turns end within 0.2 m of the exit lane centre
  - a turn the camera cannot see into still follows the route
  - arrival stops within 0.4 m of the pin
  - it waits its turn about 1.8 m short
  - a destination just past or just behind the parked bus
- **Arrival and the command line:**
  - a car queuing 11 m before the pin is waited behind, not taken for the destination
  - a destination "x,y" with a negative x is passed on as `--destination=x,y` at every step
- **The sat-nav:**
  - junction mode starts 20 m before the junction
  - a stale "arrived" from the driver does not mark a new, far destination reached
  - a new destination picked 0.5-1 m off a junction connector still starts on that connector
  - a wrong turn re-plans once
  - driving in the divided road's other lane while passing is not "leaving the route"
  - the driver is told where a U-turn is until it has turned round; turning round is not "leaving the route"; a turn
    the driver gives up is re-planned without U-turns

### Town test (8-9 November)

Each test is a pair of 900-simulated-second drives (seed 8101, clear weather, `-Check`). Between them the two lists cover
left, right and straight-on turns and every district:

- **List 1:** Residential square, Outskirts bend, Irregular junction
- **List 2:** Village road, Outskirts bend, Colony street, Market junction

The next destination is set 5 s after each arrival. Owner folders are `runs/camera_driving_2026110[89]_destinations*`.

| Pair | Code | Destinations reached | Collisions | Resets | Wrong side | Junction steps on a planned path |
| --- | --- | --- | --- | --- | --- | --- |
| 01 / 02 | first build | 2 of 3, 2 of 4 | 0, 0 | 3, 1 | 0 s, 0 s | 16%, 10% |
| 08 / 09 | before the crossing-pedestrian rule | **3 of 3, 4 of 4** | 1, 1 (the same jaywalker) | **0, 0** | 1.4 s, 0 s | 45%, 53% |
| 12 / 13 | with the crossing-pedestrian rule | 2 of 3, 2 of 4 | **0, 0** | **0, 0** | 0 s, 2.6 s | 49%, 39% |
| 14 / 15 | after the final review's fixes | 2 of 3, 2 of 4 | 0, 0 | 0, 1 (phantom) | 0.3 s, 0 s | 48%, 35% |
| 16 / 17 | plain-sky band near the horizon | 2 of 3, 1 of 4 | 0, 0 | 1 (turn wait), 1 (phantom) | 0 s, 0 s | 53%, 34% |
| 18 / 19 | sky compared across the box's edge | 2 of 3, 2 of 4 | 0, 1 (rear-ended a waiting car) | 2 (roadside carts), 0 | 0.1 s, 1.4 s | 47%, 36% |

Since pair 08 / 09, every run reached its first two destinations. The last one ran out of time, usually after long waits
for its turn: 156-209 s of each 900 s run went to waiting at junctions. Every leg of both lists was driven to its pin in
pair 08 / 09. Cars came to rest 0.2-0.5 m from the pin; the sat-nav records arrival at 2 m, which is `short_m` in
result.json. Hybrid A* took 19-31 ms per plan (median) and 43-68 ms for the slowest 5%; in the first pair it was 150 ms,
the limit.

What the later pairs found:

- **Pair 14 / 15 and 16 / 17: phantom 'bus' boxes inside Colony junction, facing the desert.** The plain-sky test now
  looks only at the band up to 100 rows above the horizon, because higher up a real sky darkens and has clouds. It also
  compares the sky just inside the box's edge with the sky just outside it, because the sky brightens towards the sun,
  ±15 luma across a wide box. On saved views of 7 town runs, real buses and trucks were never rejected (0 of 116
  frames), and boxes with no vehicle within 15 m were rejected in 231 of 242 frames.
- **Run 16: a 90 s wait for its turn at Irregular junction.** The car stood centred in its lane while the junction was
  given to crossing traffic; the stuck-car recovery then reset it. The town's turn-taking predates this work.
- **Run 18: two stops for roadside food carts** on the village road's left shoulder, read as a low-confidence "truck";
  see "Village roads" under Known problems.
- **Run 19: rear-ended a waiting car** at 1.3 m/s. The car had stopped 0.7 m behind it; the car filled the picture and
  the detector lost it for 1.1 s, while a sighting was kept only 0.3 s. A vehicle seen within 4 m ahead is now kept for
  2.5 s, except during a pass. Offline check: lost for 1.2 s, the car stays stopped. **This last fix has not been
  through a town run.**

**Against the design's "done"** (for the last two pairs):

- **Met:**
  - under 5 s on the wrong side in every run
  - 0 collisions in 16 / 17
- **Not met:**
  - **Every destination reached:** 2-3 of 3-4 per pair in 900 s; 7 of 7 in pair 08 / 09.
  - **0 resets:** 4 resets in the last 4 runs:
    - one turn-taking wait
    - one phantom, since fixed
    - two cart stops
  - **0 collisions:** 1 in 18 / 19, since fixed (not yet town-tested).
  - **Turns planned by Hybrid A\* in most junctions:** 34-53% of junction-mode frames. Those frames include queuing
    behind other vehicles (at D in run 13, 553 of 792 frames had no goal in view), so the share for turning alone is
    higher, but it was not measured.

**Without a destination, after the final review's fixes** (`runs/camera_driving_20261109_nodest01`, 600 s): 2.29 km, and the
town's own check passed. The new obstacle rules caused no false stops: 0.7 s of "person ahead" in 600 s. At Irregular
junction the kerb follower took the tight left along the edge of a patchy road-model area on the sandy junction
surface. It ended on the footpath, touched something static, and was reset at 262 s, leaving 35 s off the road.
Steering without a destination is unchanged by this work; this is the junction weakness under "Known problems" (7 Nov
run06 and run11 at the same junction).

What the earlier pairs found, each fixed with an offline check:

- **Destinations reached at once.** After each arrival the driver still said "arrived" on the next step, so the next
  destination, about 1 km away, was marked reached. The sat-nav now trusts the driver's "arrived" only within 25 m of the
  pin.
- **Hybrid A\* rarely found a path** (62 of 446 tries). It needed 1.32 m of clearance each side, too much for the
  town's 2.8 m lanes, and any stray not-road pixel blocked a cell. It now uses seven body discs (1.16 m) and a cell is
  blocked only when most of it is not road.
- **Junction goal behind a vehicle.** The goal was the farthest free route point, right behind a bicycle, where the
  car's centre fitted but its body did not. The planner either ran to its time limit or found a path that swung the car
  right off a left bend into the sand. The goal now needs room for the whole car, and a vehicle or person on the route
  ends the search, so the car follows them.
- **Stops beside vehicles waiting in the opposite lane** (three resets). A vehicle seen at an angle has its far corner
  as its box's inner edge, which read 1-2 m nearer the path. Its sides are now taken from where the road model sees no
  road along the box's base, and its width-based distance from that base width.
- **Phantom 'bus' boxes facing the open desert** (two resets). A wide box thrown out as a phantom in the last 2 s, in
  about the same place, stays out while a quarter of its base shows road. A bus or truck box whose body would rise well
  above the horizon at its own distance, yet shows plain sky there, is thrown out. On the saved views the sky band
  measured a luma spread of about 7 for the phantoms and 46-63 for real buses.
- **A jaywalker walked into the car's side.** The car saw them 2.3 m ahead of its front, 1.2 m clear of its side, only
  slowed, then lost them beside its bumper. People are now tracked from frame to frame. One walking towards the path at
  0.5 m/s or more, who will be in the car's way as it passes, makes the car stop.
- **The viewer crashed on TAB** (the spot camera): a local variable hid the function it called.

The town's own pass check failed in all four of the last runs, for reasons outside your car:

- collisions and overlaps between other vehicles and walkers
- walkers lost from CARLA's registry (`walker_4` at 516 s every time)

## Results (simulation, seed 8101)

All 7 November drives, 600 simulated seconds each. Owner folders are `runs/camera_driving_20261107_run*`, and each status.json
names its town folder. Each drive ran on slightly different code, because every one found something that was then fixed
(see the sections above). None was run on the final code.

| Measure | Run04, clear (passing fixes) | Run06, clear (+ village roads, carts) | Run11, clear (+ run06 fixes) | Run12, dusk (same code as run11) |
| --- | --- | --- | --- | --- |
| Distance | 2.40 km | 2.09 km | 1.80 km | 2.03 km |
| Districts | 4 of 5 | **all 5** (first time) | **all 5** | 4 of 5 |
| Collisions involving your car | 0 | 0 | 1 (a static object, impulse 35, after leaving the road) | 0 |
| Resets | 0 | 2 | 2 | 1 |
| Off the road | 3.8 s | 4.0 s | 36 s | **0 s** |
| Wrong side | 3.1 s | 31 s | **0 s** | **0 s** |
| Lane centre, outside junctions | median 0.15 m, p90 1.06 m | median 0.10 m, p90 **0.20 m** | median 0.07 m, p90 0.57 m | median 0.15 m, p90 0.75 m |
| Village road 3, signed offset (median) | −0.78 m | −0.06 m | −0.25 m | −0.01 m |
| Bus at its stop | passed | passed | passed | passed once; at the second visit (532 s) two passes given up |
| Workers, protected files | clean, unchanged | clean, unchanged | clean, unchanged | clean, unchanged |

Run05 (clear) and run07 (wet) are under "Village roads" and "Other weather" above. Run08 (dusk) lost the GPU at 110 s.

**What went wrong, and what was done:**

- **Village roads** are fixed in clear weather and at dusk. The lane-centre spread went from p90 1.06 m to 0.20–0.75 m.
- **Roadside carts:** run11 (18 s) and run12 (a 30 s stop, then a reset) both stopped for a cart again. Its wheels were in view,
  so the box was cut off at the side only. The box joins the cart's far corner to its near corner, which placed it 0.9 m
  inside the car's path, and the first cart fix only handled boxes cut at the bottom too.
  - **Fixed afterwards:** side-cut boxes are now checked against the road at the object's own distance.
- **A phantom came back.** Run11 had a phantom "bus" over three quarters of the picture, cut at its right edge. Run06's
  side-cut exemption let it through.
  - **Fixed afterwards:** the exemption now covers only boxes under 45% of the picture's width.
- **Passing at dusk (run12, second visit):** beside the bus the car lost sight of the left kerb, and "lost the road: stop"
  halted it mid-pass. It gave the pass up twice; the second time its way back was blocked by a car queuing behind it.
  - **Fixed afterwards:** the lost-kerb stop does not apply while following a planned pass, which never used the kerb. The
    obstacle and road-end checks still apply.
- **Junctions (not fixed):**
  - **Run06:** the car came out of junction C on the wrong side and stopped facing the road edge.
  - **Run11, 287 s:** at a junction, another car gave way to yours while yours braked for it, and both waited until the 30 s
    reset.
  - **Run11, 514 s:** the car lost the kerb, drove onto the footpath and touched a static object there.

These three fixes pass the offline checks but have not been driven in the town yet.

"Done" means on this seed only. It is not a safety result.

**Run03, 6 November**, `runs/carla_indian_worlds_smoke_20261106T150207_125951Z_seed8101`, owner folder
`runs/camera_driving_20261106_run03`, first met the design's "done" bar: a full loop driven only by the models, with no reset.
It improved on run02:

- 2.39 km instead of 2.05 km.
- No collisions; run02 scraped the parked bus while passing.
- No resets; run02 had two.
- 2.2 s on the wrong side instead of 37 s; run02 "passed" a queued motorcycle.
- Longest stall 4.9 s instead of 30 s.

Its AI views at key moments are in `camera_driving/run_sheet.png` in its town folder.

**Earlier runs:**

- **120 s smoke run:** 486 m, nothing hit, lane-centre median 4 cm.
- **Run01:** CARLA lost the GPU (`VK_ERROR_DEVICE_LOST`) at 325 s. This happened once before without the models, on 27 Oct.
  The semantic camera has since been dropped and the evidence cameras slowed to the 1 Hz they are saved at.

**Offline checks:** `setup/verify_camera_driving.py` passes 62 of 62 (`--replay` adds the replay checks). These cover:

- camera geometry
- lane keeping on straight roads and curves
- obstacles and phantom boxes:
  - a car cut off at the picture's side is kept
  - a phantom over three quarters of the picture is still ignored
  - the same phantom stays ignored after the car turns and half its base shows no road
  - a 'bus' 4 m away over plain sky is ignored, while a real bus and a car under plain sky are kept
  - vehicles waiting in the opposite lane, seen at an angle, are not on the path; one over the lane line is
  - a plain-sided truck is kept: the plain-sky test also needs the sky beside the box to match
  - a phantom from the top of the picture over a graded, cloudy sky that brightens towards the sun is still ignored
  - a car stopped just ahead and lost by the detector for 1.2 s keeps the car stopped
- a jaywalker crossing in front makes the car stop clear of them; people walking or standing at the road edge do not
- a motorcycle stopped ahead for 30 s: the car waits, never pulls out
- closed-loop driving within 0.07 m of the lane centre
- passing a stopped 3.9 m CARLA bus with a 0.79 m body gap
- passing with the rear camera:
  - a clear lane
  - a car coming up behind in the passing lane (it waits until that car is 47 m ahead)
  - a bus coming the other way mid-pass (it stops, backs out 12 m, lets the bus by, then passes; closest 1.2 m)
  - no rear view (it does not pull out)
  - backing up with a car behind (it stops)
- wedged (throttle on, not moving, nothing seen): it backs off the way it came after 2 s and drives on; standing behind
  a stopped car is not taken for it
- with the side cameras too (rear and right in turn): a clear lane, and a car coming up behind, pass as before. Before
  pulling out, the right camera waits for a car moving alongside (one never seen behind), not one standing there, and
  the "seen behind, may be alongside" wait holds only while something shows beside the car
- a village road whose shoulder always counts as road:
  - with the asphalt edge, the car settles within 0.10 m of the lane centre, starting 1.2 m left
  - without it, the car rides 1.25 m left, as in the town
  - a kerb street with a bright sidewalk is unchanged
  - roadside carts beside the car, with their wheels in view or not, are not on its path, while a car cutting in at the
    front-left still is
- the worker protocol and the no-simulator-access check
- the viewer: no function hides one of the scene's own functions with a local variable
- a recorded-frame replay: 0.11 m median target error on market, colony and junction streets

## Passing safely (7 November)

The first pass design had three holes. Each is now fixed and has an offline check.

| Hole | Fix |
| --- | --- |
| While passing a bus, every box labelled "bus" was ignored, so a bus coming the other way in the passing lane would not have stopped it, not even the emergency stop. | Only the vehicle being passed is ignored. It is recognised by where it appears in the picture: its box continues from the last frames, or it covers where its planned footprint should be. Everything else on the path counts. |
| It never changed its mind. If something blocked the pass, it waited on the pass path, possibly in the oncoming lane, and gave up after 25 s by steering back towards the bus. | Before pulling out, the passing lane must be clear ahead, to 10 m past the stopped vehicle. If it is stopped mid-pass for 3 s, it gives up and backs out slowly the way it came, then straight back until it is straight again. It then waits behind (no creeping closer) and tries again after 6 s. |
| It could not see behind or beside itself. | A rear camera, fitted 0.1 m behind the rear bumper and 1.3 m up and looking 20 degrees down (it sees the ground from 1 m behind; level until 18 Nov), runs only while the car waits behind something it may pass, passes, backs up or turns round (it is removed 5 s after). It must show the passing lane clear for 30 m behind. A vehicle seen close behind in that lane is assumed to be alongside, in neither camera's view, until it shows up ahead or 8 s have passed. While backing up, the car stops 3 m short of anything behind. Since 23 Nov, side cameras at the door pillars see beside the car: the right one around passes (a vehicle moving alongside is waited for, and the guess above holds only while something shows there), both while turning round. |

Things found while testing the fixes:

- The first blind-spot test had the car start its pass with the other car right beside it. That car had left the rear
  view but was not yet in the front view, which is why the blind-spot memory exists.
- Following the old path while reversing brought the car back to the right spot but turned up to 24 degrees, too far to
  try again beside the 3.9 m CARLA bus. It now carries on straight back until within 2.5 degrees of its starting
  heading.
- After giving up, the car was still steering along the abandoned pass path and crept closer to the bus, which made
  each retry steeper. It now steers straight and holds its position.

## Village roads: the paved shoulder (7 November)

Village roads have 2.8 m lanes with a 1.3 m paved shoulder on each side, lighter than the asphalt. In run04 the car rode
0.5–1.1 m left of the lane centre there (median by road) and every other street was within 0.2 m.

A probe ran the road model on run04's own village-road pictures (`runs/camera_driving_village_probe_20261107`) and
compared its left edge with the map. The road model has two states:

| Where the car is | Shoulder scored as road | Left edge the car follows |
| --- | --- | --- |
| Within 0.4 m of the lane centre | 3% | the true lane edge (+0.05 m) |
| 0.6 m or more left | 99% | the shoulder's outer edge (−1.27 m) |

It counts the shoulder as road once the car is near it. Following the left kerb out of a junction puts the car there, so it
then stays locked about 1.25 m left. A stricter cut-off on the road model's probability does not help, because it is just as
confident either way. The right edge is not a usable cue either: the measured road width swings between 2.5 and 6.9 m.

**Fix:** inside the road model's left edge, the driver now also looks at the picture for the bright-to-dark step where
the asphalt starts.

- **When it applies:** at least 4 rows, 6–12 m ahead, must agree on a step 0.6–1.8 m inside the edge, with the shoulder band
  at least 1.25 times brighter than the asphalt.
- **What it does:** that step becomes the lane's edge, and it is remembered for 1.5 s.
- **Otherwise:** the road model's edge is used, as before. On kerb streets the step is at the edge itself, so nothing changes.

On the probe pictures with the car already left, it moved the edge from −1.22 m to +0.10 m (median). It never triggered on
market or colony frames and never made an edge worse by more than 0.3 m. Kerb dots drawn amber in the AI view mark rows it
moved.

This is a hand-made picture cue next to the trained road model, not a model. It relies on the shoulder being brighter
than the asphalt, as in this town, and a shadow edge running along the road could in principle fool it.

**Roadside carts.** With the car now in its lane, the first test drive (run05) stopped beside a food cart on the shoulder
for 30 s and had to be reset. The cart filled the left edge of the picture with its base out of view. Its box was cut off
at the bottom and the side, so its bottom edge placed it about 0.5 m inside the car's path. In run04 the car rode on the
shoulder, so the carts really were ahead of it. Now a box cut off at the bottom and a side is cross-checked with the road
model. Where the nearest visible rows show unbroken road from straight ahead out to an edge, the object stands beyond that
edge. A car cutting in at the front-left still blocks that road, so it still stops the car.

## Other weather (7 November)

Two more drives tested the village-road and cart fixes. Neither counts as a result for the final code, which changed again
afterwards, but both found problems:

- **Wet (run07) failed badly.** The road model marked shiny wet sand beside the roads as road. The car drove off the road
  for 180 s, was reset 5 times, hit static props twice and brushed a pedestrian (impulse 12). This is the road model failing
  outside what it was trained on; the driver cannot fix it. It needs the road model fine-tuned on wet scenes, which means
  changing the locked models, so it waits for your decision.
- **Dusk (run08) did not finish.** CARLA lost the GPU (`VK_ERROR_DEVICE_LOST`) at 110 s, on the third 600 s drive in a
  row. That is the third time this has happened (27 Oct, 6 Nov, 7 Nov).

The clear drive in between (run06) found three more things, fixed afterwards:

| Found | Fix |
| --- | --- |
| It started to pass a motorcycle that had stood still for 30 s, which was really queuing for junction D. This is the same mistake as run02. | Passing is now for stopped buses and trucks only. Behind anything else it waits; `pass_small` turns the old rule back on. |
| The stuck-car timer reset the car the moment that pass began, because the 90 s allowance for queuing no longer applied once the reason changed to "passing". | Passing, backing out and changing gear get the same 90 s allowance. |
| The phantom-box check threw out a real car close at the front-left. Its box was cut off at the picture's edge and took in the road in front of it. | Boxes cut off at a side are never judged as phantoms. The frame-wide phantoms it was written for sit in the middle of the picture. |

Run06 also left junction C on the wrong side of the C–D street and stopped facing the road edge for 30 s until it was
reset. That is the known junction weakness.

## Changes from the approved design (all found in testing)

| Change | Why |
| --- | --- |
| Person "on the path" margin 0.5 m (was 0.8) | 0.8 made people standing at the kerb stop the car |
| The driver's own brake response when faster than its target | The rule driver's controller eased off too slowly to stop for a late obstacle |
| Vehicle distance = nearer of the box-bottom and box-width estimates; bus overhang 2.0 m | A raised bus bumper read up to 2× too far |
| Wide boxes with road visible under their base are ignored | The detector drew frame-wide phantom "bus" and "car" boxes over empty road |
| Thin non-road bands do not count as "road ends" | A junction seam stopped the car |
| Re-finding the road up to 2.5 m away after a lost frame | The car lost the road inside junction B |
| Planned pass: after waiting 3 s behind a bus or truck (15 s behind others, never near a junction), an S-curve out past its right side, 1.3 m clear, at 2.5 m/s | The bus stops in the left lane of the divided road on every loop |
| Stop 6 m behind buses and trucks | To leave room to see and steer round them |
| Recovery waits 90 s when queuing behind a vehicle | Junction queues can exceed 30 s |
| Perception camera 10 Hz; semantic camera dropped; evidence cameras 1 Hz | GPU time and the device-lost crash |
| Rear camera around passes only; only the passed vehicle ignored; clear-lane checks ahead and behind; backing out of a stalled pass (7 Nov) | See "Passing safely" above |
| The left edge also uses the asphalt step beside a paved shoulder (7 Nov) | See "Village roads" above |
| Boxes cut off at the bottom and a side are placed beyond the road the road model sees (7 Nov) | A roadside cart beside the car read as on its path |
| The pass record survives a reset (7 Nov) | run05's record was wiped when the car was reset |
| Passing only buses and trucks; a 90 s stuck-car allowance while passing; side-cut boxes under 45% of the picture never treated as phantoms (7 Nov) | See "Other weather" above (found in run06) |
| Side-cut boxes checked against the road at their own distance; no lost-kerb stop during a planned pass (7 Nov) | See "Results" (found in run11, run12) |

## Known problems

- **Over-cautious "road not visible".** The car often slows where the road curves or narrows out of sight. That was 117 s
  of slowing in run03, and its average speed was 4.0 m/s.
- **Village roads.** The car rides left, partly on the shoulder (signed lane offset 10th percentile −1.07 m in run03).
  In run04 that brought it within about 1 m of roadside food carts, so it braked hard twice.
- **Route.** It never chose the C–D street, so the irregular-junction district was not reached in run03 or run04. At E it does not
  always take the leftmost exit either.
- **Junctions.** The road model is weakest here (IoU 0.69). The car slows and occasionally loses the left kerb:
  - it came out of junction C on the wrong side (run06)
  - it mounted a footpath (run11)
  - another car giving way to it while it braked for that car stalled both until a reset (run11)
- **No side view at junctions.** The side cameras run only around passes and turns, so the other vehicles giving way
  to your car is still what keeps junctions safe.
- **Side cameras see only what is in their picture.** A vehicle close alongside fills it, so its ends are not seen; one
  crawling past at walking pace then looks still, and before a pass is taken for one standing there. Their distances
  are from the ground under the box's bottom only, and they have not been compared with true distances in CARLA.
- **Passing still relies on others giving way.** A vehicle that arrives after the pass has started stops for your car
  (the town's courtesy rule). Backing out only uses the detector on the rear camera, not the road model, so it trusts
  that the way it came is still road.
- **Rear-camera distances** use the same flat-ground and typical-width estimates as the front. The offline checks use
  synthetic pictures. In run04 the town's rear pictures gave sensible labels (an empty lane, then the passed bus beside the
  car), but they have not been compared with true distances.
- **Wet weather.** The road model takes wet sand for road, so the car leaves the road (run07). This needs model work.
- **GPU.** Sharing it with CARLA can still crash it: three times so far, the last on the third drive in a row.
- **Destinations (9 November):**
  - **Vehicles beside the car at a junction exit.** A vehicle waiting in the opposite lane can still stop the car when
    its box is cut off at the bottom of the picture, because then its base is not seen. destinations05 waited 90 s
    beside one, about 1.25 m clear.
  - **Long turn-taking waits.** At busy junctions a single wait can take 60-75 s, so a 900 s test may not reach the last
    destination.
  - **Runs that never start.** About 1 launch in 3 stalls before CARLA starts: the session's Python process never runs.
    Stopping it and starting again has worked every time.
