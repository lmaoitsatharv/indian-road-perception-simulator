# Destinations for the camera-driven car: design (8 November 2026)

Status: approved 8 Nov and built 8-9 Nov on branch `camera-driving-destinations`. Results, and where the build differs from this design, are in [CAMERA_DRIVING.md](CAMERA_DRIVING.md) ("Picking a destination").

## Goal

You pick where your car goes, either from a list of named places or by clicking a point on the town map. The car plans
a route, chooses its own turns, drives there and pulls in to the left kerb. It waits there until you pick the next
destination. With no destination it behaves exactly as today, following the left kerb wherever it leads.

Choices made on 8 Nov:

| Question | Answer |
| --- | --- |
| What may the car know? | **Sat-nav plus camera.** The car knows its position and the road map, to plan the route and aim junction turns. The camera models still steer between junctions and plan the turn itself. |
| Position accuracy | **Exact** simulator position, with no GPS noise. |
| On arrival | **Pull in and wait.** A new destination can be picked at any time, even mid-route, and the car re-plans. |
| Junction right of way | **Join the town's junction turn-taking**, as the other vehicles do. The rule that makes them give way to your car stays as a safety net. |

## What the car uses, and where

| Information | Used for | Not used for |
| --- | --- | --- |
| Front camera through the road model and detector (as now) | Lane keeping between junctions; the drivable area for junction turns; braking for obstacles | — |
| Rear camera (as now) | Passing and backing out | — |
| Exact position and heading, and the road map (new: the "sat-nav") | Planning the route; knowing which junction comes next and which exit to take; aiming the junction turn; the stopping point on arrival; the junction queue | Lane keeping between junctions; obstacle detection |
| Junction turn-taking (new) | Waiting for its turn before entering a junction | — |

`integration/india_camera_driver.py` still never touches the simulator or the map; `setup/verify_camera_driving.py` keeps
checking that. The sat-nav runs in the viewer process, reads the map, and hands the driver one small message each control
step. The new planner modules do no simulator calls either; they work on the network file and on arrays.

## Components

### 1. Route planner (`integration/india_route_planner.py`, new)

- **Lane graph** from the town's network file (`network.json`) and the map's lane centre lines (sampled every 0.35 m, as
  the other vehicles' routes are):
  - **Nodes:** every street lane in its direction of travel; the divided road's two lanes per direction are one node.
  - **Edges:** the 36 junction connectors, which list exactly the turns allowed from each incoming lane. There were no
    U-turns, so a destination behind the car was reached by going round the block (changed 18 Nov: see the addendum).
- **`snap(x, y)`** places a clicked point on the lane centre line of the nearest street, on the side of the street that
  was clicked, giving that lane and the distance along it.
  - A point inside a junction moves to the nearest street point at least 5 m outside it.
  - A point more than 15 m from any street is refused, with a message.
- **`plan(pose, destination)`** runs Dijkstra on the lane graph, with costs in metres. It starts from the lane the car is on
  (from its exact position and heading) and its distance along it.
  - If the car is inside a junction, it starts from the exit it is already taking.
  - It returns a `Route`:
    - the lane-centre polyline from the car to the destination (x, y, heading, distance)
    - the legs (street, then connector, then street, …)
    - for each junction: its name, where it starts and ends along the route, and the turn (left, right or straight
      on, from the heading change)
- **`Route.progress(pose)`** gives the distance along the route, the distance to the next junction and to the
  destination, and whether the car has left the route: more than 4 m from the polyline, or on a different street.
- **Named places,** 11 of them:
  - **The 7 junctions by their names:** Bazaar gate, Village gate, Irregular junction, Colony junction, Market junction,
    Residential square and Outskirts bend. Each place is the point 15 m into the leaving lane of the first street listed
    for that junction in the network file.
  - **4 district spots:** the bus stop on the divided road; the middles of the village road (street 3), the colony street
    (street 5) and the market street (street 1), in the lane of each street's own direction.

### 2. Sat-nav (in the viewer, `integration/india_satnav.py`, new)

It holds the current destination and route and, each control step, takes the car's exact pose from the simulator. It:

- **Re-plans:** when you pick a destination, when the car leaves the route (for example a missed turn), and after a reset.
- **Builds the driver's message** (a plain dict; no simulator objects):
  - **`mode`:** `"lane"` on a street; `"junction"` from 20 m before a junction's start until the car is on the exit street.
  - **`guide`:** the route polyline from 0 to 30 m ahead, in the car's own coordinates (x forward, y right).
  - **`junction`:** name and turn, for display.
  - **`hold_at_m`:** how far ahead to stop when it has not been given its turn at the next junction. `None` when it may go.
  - **`arrive_in_m`:** the distance along the route to the destination, when under 60 m.
- **Tells the HUD** the route, the destination pin, the remaining distance and the next instruction ("right at Irregular
  junction in 80 m").

### 3. Junction turn planner (`integration/india_hybrid_astar.py`, new; pure NumPy)

- **Grid:** the camera driver's top-down road grid (2.5–25 m ahead, ±10 m, 0.1 m cells), pooled to 0.25 m cells.
  - **Blocked:**
    - cells the camera sees that the road model marks not-road
    - the ground under detected vehicles and people (their footprints, as placed today)
    - every blocked cell widened by the car's half-width plus 0.2 m, using a distance transform
  - **Unknown:** cells the camera cannot see are allowed but cost extra.
- **Start:** the car's pose (0, 0, 0). **Goal:** the farthest point on `guide` that lies in the camera's visible area, at
  least 2 m inside the grid's edges, with the guide's heading there.
  - The goal moves round the corner as the car turns, because a tight left turn's exit, about 23 m to the side, is outside
    the grid at first.
  - **Reached:** within 1.0 m and 15°.
- **Search:**
  - **States:** x, y, heading in 5° steps.
  - **Moves:** forward only. Five steering angles (full and half lock either way, and straight) over 1.0 m arcs at the
    car's own wheelbase.
  - **Cost:** length, plus 0.5 per steering change, plus 0.3 per metre of distance from `guide` (this keeps it in its own
    lane rather than cutting the corner across the other lane), plus 1.0 per unknown cell.
  - **Heuristic:** straight-line distance to the goal.
  - **Limits:** 150 ms wall time or 4,000 expansions.
- **Re-planning:** every 0.5 s while in junction mode, and immediately when an obstacle appears on the current plan.
- **Driving the plan:** the existing pure pursuit, with a 4 m look-ahead along the planned path in the odometry frame.
  - **Speed:** at most 3.5 m/s in a junction, plus the existing curve limit.
  - **Obstacles:** the existing braking checks run along the planned path.
- **Fallback:** if no path is found, the car follows `guide` itself at 2.5 m/s, with obstacle braking on. `junction_fallback`
  is logged. If it is still blocked after 10 s, it stops and the usual stuck-car recovery applies.

### 4. Camera driver changes (`integration/india_camera_driver.py`)

- **New `navigate(message)` input**, used by `control()`:
  - **`"lane"`:** unchanged. It follows the left kerb, and the route is not used for steering.
  - **`"junction"`:** it steers along the Hybrid A\* plan (or the fallback) instead of the left kerb. It returns to kerb
    following once the message says `"lane"` again (on the exit street) and the kerb fit has 6 or more rows.
  - **`hold_at_m`:** a new limit, "waiting for its turn at the junction", that stops it 2 m short.
  - **`arrive_in_m`:**
    - under 30 m: a speed cap of 3 m/s, and the keep-left offset eases from 1.5 m to 1.1 m, closer to the kerb
    - under 1 m: it stops and holds, with the new reason `"arrived"` (decision STOP)
- **Pass rules stay as they are.** No pass starts within 40 m of the destination or in junction mode.
- **New reasons:** `junction_turn`, `junction_fallback`, `waiting_turn`, `arriving`, `arrived`, `no_route`.

### 5. Turn-taking for your car (`integration/india_combined_traffic.py`)

- With a route, your car takes part in the junction reservations like the other vehicles. The sat-nav's `Route`
  provides what the reservation code reads from a vehicle route: `at(s)`, `next_junction(s)`, the distance to the junction
  and the connector it will use.
- If it has not been given its turn, the sat-nav sets `hold_at_m`.
- Without a route (no destination) it stays outside the reservations, as today.
- The other vehicles' give-way rule for your car stays as a safety net.

### 6. Viewer (`integration/india_combined_scene.py`, `integration/india_town_hud.py`)

- **M:** a full-screen town map with the streets, junction names, your car, the place list (clickable) and, if set, the
  route and pin.
  - **Click a road point or a place name:** sets the destination.
  - **Esc or M:** closes the map.
  - **A refused click** (too far from any street): "Not on a street" for 2 s.
- **X:** clears the destination; the car goes back to cruising.
- **HUD:**
  - **Minimap:** the route line and pin.
  - **AI panel:** "To Colony junction · 640 m · next: right at Irregular junction in 80 m", or "Arrived".
  - **AI view:** in junction mode, the planned path drawn in magenta and the goal point.
  - **Help:** lists M and X.
- **Command line:**
  - **`--destination "Colony junction"`** or **`--destination x,y`:** starts with that destination.
  - **`--destinations "A;B;C"`:** for test runs; after arriving it waits 5 s, then sets the next.

  Both pass from `Start_Camera_Driving.ps1` and `setup/run_camera_driving.py` through `run_session.py`.

## Records

- **`camera_driving/navigation.jsonl`:** one line per change. Destination set, route planned (legs, length), junction entered
  and left (turn, fallback used or not), re-planned (why), arrived (time, distance from the pin), cleared.
- **`decisions.jsonl`:** each frame gets `mode`, `plan_found`, `plan_ms` and the goal.
- **`result.json` → `camera_driving.navigation`:**
  - destinations reached and their times
  - re-plans
  - junction turns: planned, fallback and per junction
  - Hybrid A\* plan time (median and p95)
  - time spent waiting for its turn

## Edge cases

| Case | Behaviour |
| --- | --- |
| Destination behind the car or on the other side of the street | Either side of a two-way street will do; a U-turn when it saves going round the block (addendum, 18 Nov) |
| Car takes a wrong exit | The sat-nav sees it has left the route and re-plans from where it is |
| Reset (stuck recovery or R) | Re-plan from the new position |
| Destination picked while inside a junction | The route starts from the exit the car is already taking |
| Perception unavailable | Stops, as today; the route is kept |
| Hybrid A\* too slow or no path | Fallback along the route line (above) |
| Click inside a junction or off-street | Moved out of the junction, or refused if more than 15 m from a street |
| Destination reached while a pass is in progress | Cannot happen: no pass starts within 40 m of it |

## Testing

**Offline** (`setup/verify_camera_driving.py`, no CARLA server or GPU; the route checks use the offline map with
`.venv-carla`):

- **Route planner:**
  - every place reachable from every other
  - no route uses a U-turn
  - every route continuous (lane joins under 0.65 m)
  - clicks on each side of a two-way street snap to the matching lane
  - a click inside a junction moves out
  - a far-off click is refused
- **Hybrid A\*** on a synthetic four-way junction drawn with the same camera model:
  - plans left, right and straight on within the time limit, staying in the correct lanes
  - plans round a stopped car in the junction
  - reports no path when the exit is walled off
- **Closed loop:** the synthetic car with the full driver turns right at the synthetic junction and is in the correct exit
  lane within 0.3 m, 10 m after it. A left turn and straight on are run the same way.
- **Arrival:** on a straight synthetic road with a destination 80 m ahead it stops within 2 m of the pin, at most 1.3 m from
  the kerb.
- **Turn-taking:** with `hold_at_m` = 12 it stops 10–12 m ahead and waits.
- **Isolation:** the driver and the Hybrid A\* module have no simulator access.

**Town:** one 900 s run in clear weather with `--destinations` chosen so that the car makes:

- each available turn at least once at C (four arms), D and E (three arms each)
- all five districts
- a destination behind the car, which needs a loop

"Done" means:

- every destination is reached
- 0 collisions involving your car and 0 resets
- under 5 s on the wrong side
- the AI view shows the turns planned by Hybrid A\* rather than by the fallback in most junctions (the share is reported)

The same run at dusk follows if the GPU allows. Results go in `docs/CAMERA_DRIVING.md`.

## Out of scope

Noisy GPS. Side cameras. Wet weather. Hybrid A\* for passing. Lane changes on the divided road other than the
existing pass. Multiple stops in one request beyond the test-only `--destinations`. (U-turns were out of scope here;
they were added on 18 Nov, below.)

## Addendum, 18 November: either side of the street, and U-turns

A pin clicked on the far side of a two-way street ahead of the car was routed 1.8 km round the town, because a
destination meant one lane and routes never turned round.

- **Either side.** A destination on a two-way street can be reached on either side of it (`TownRoutes.across`); the
  route arrives on whichever is nearer and the map's pin moves to that side. The divided main road's far carriageway is
  not "the same spot".
- **U-turns.** A route may turn round on a two-way street, at least 25 m from its junctions (and at least 12 m ahead of
  the car on the street it is driving), counted as 80 m more than its length. Never on the divided road.
- **Driving one** (`CameraDriver._turn_around`): the sat-nav tells the driver where the turn is and the road's edges
  from its map. The car stops there, waits until no vehicle is coming within 35 m ahead (one standing still more than
  8 m ahead does not count), nobody is within 12 m ahead and nothing is overtaking on its right, then turns on full
  lock: forward right, back left, and so on, ending each move 0.35 m short of the road's edge (any corner of the car)
  and stopping for anything the cameras see in the way. Once it faces the other way it lines up with the far lane and
  follows the route line until it has the kerb again. It tracks its position during the turn as a car turning about
  its rear axle.
- **Giving up:** after 12 moves or 80 s it gives the turn up and the sat-nav re-plans to the same destination
  without U-turns. A stall reset in the middle of a turn re-plans without U-turns too.
- **Offline checks** (`setup/verify_camera_destinations.py`): a three-point turn on a 6 m street inside the road's edges;
  more moves with a 35 degree steering lock; waiting for an oncoming car; a car parked ahead does not hold it up; a
  stall on the pavement beside the turn; a person on the pavement (not waited for) or in the road (waited for); a bus
  standing straight behind; phantom boxes over the rear picture, and a real car coming up close behind; a car coming
  along the far lane mid-turn (side cameras), and one parked there.
- **The route export for the learned speed** (`rl/export_town_routes.py`) still plans without either change, as the
  policy was trained.

### First CARLA run, 18 November (Market street to point 117, -76)

The car stopped, waited 27 s for a gap, turned to 74 degrees and then stood still for 80 s until the stall reset lifted
it back, which planned the same U-turn again; the second try waited a minute more. From the run's logs:

- **A stall on the pavement behind** was read as "truck 1.0 m behind" and held the backing move. No vehicle was there.
  The level rear camera saw no ground nearer than 2.1 m, so the stall's box was cut off at the picture's bottom, and a
  truck's 1.2 m bumper overhang was then taken off that.
- **A phantom "bus 0 m behind on its right"** (cut off the same way, standing still) held the first gap for 15 s.
- **People standing on the pavement** ("person 7 m ahead") held the second gap, and then the forward move.

Changes:

- The rear camera looks 20 degrees down (`setup/run_camera_driving.py`), so it sees the ground from 1 m behind the car.
- During the turn, anything that does not reach more than 0.35 m onto the road (the strip the car's body never enters)
  is ignored (`_on_turning_road`). Across the street, vehicles and stalls behind are seen side-on, so their overhang
  counts only as far as the car faces along the street; a box cut off at the picture's bottom is taken to reach the
  bumper.
- Something behind that stands still is not "overtaking on its right".
- A move held for 2 s by something in its way, after covering 0.3 m, ends there and the car goes the other way, as at
  the road's edge.
- The turn gives up after 80 s, before the town's 90 s stall reset; a reset during a turn re-plans without U-turns.
- `decisions.jsonl` records the driver's detail line (`previous_detail`), which says what it is waiting for.

### Second CARLA run, 18 November (the same pin, with the changes above)

The car waited 27 s for a real gap, then turned round in 6 moves (32.5 to 78.8 s) without touching anything and drove
on through Market junction. The turn took about 46 s where about 25 s would do: boxes over most of the rear picture,
with nothing there, held the backing moves for about 20 s in all:

- "bus 0.0 m behind" and "truck 0.0 m behind": a box from the picture's top to its bottom, three quarters of its width,
  cut off at the bottom, so taken to reach the bumper.
- "autorickshaw 0.7 m behind": nearly as large, ending just above the bottom.

The ground truth had nothing within 6 m behind. Each move was freed only by the 2 s "blocked" rule.

Change (`_rear_believed`): nothing gets within 1.5 m of the car's tail unseen. The car stops 2.5 m short of what it
sees behind, and anything coming up is read farther out first, where the ground under it shows. So during a turn, a
rear box nearer than 1.5 m counts only if something was read 1.5-6 m behind and on the road, or the rear camera has
just started. It stops counting once the car has driven 0.5 m forward since then, which would show a real one farther
out. A car that comes up behind and stops 0.5 m from the tail still counts, and so does one already there when the
camera starts. With the two boxes above, from CARLA's exact readings, the offline turn takes 1.5 s longer than
without them; without the rule it never finishes.

`decisions.jsonl` now records each rear box's pixels and confidence (`rear[].box`, `rear[].conf`); this run's log had
only the distances.

### Side cameras, 23 November

Across the street the car's front camera looks at one pavement and its rear camera at the other, so traffic coming
along the street in either lane was in neither view. Two side cameras now sit at the door pillars, 1.3 m up, looking
straight out and 15 degrees down (`setup/run_camera_driving.py`); both run while turning round. Each frame sends one
of the rear, left and right pictures to the detector in turn, so each refreshes about 3 times a second and the
detector's load is what it was with the rear camera alone.

During a move (`_coming`), anything a side camera sees on the road within 20 m that comes towards the car at 1.2 m/s
or more stops the move where it is, until it has stopped or gone by. Each side sighting is matched to the nearest
sighting by any camera 0.6-1.6 s before. The front camera is left out: its distances shift as the car turns, so a
parked car looked like one coming. Offline, a car coming along the far lane mid-turn (from 150 m at 6 m/s, too far
off to wait for at the start) holds the backing move from 18 m until it stops 7 m short; without the side cameras the
car backs on while it comes within 12 m. A car parked 14 m along the far lane does not hold the turn.

### CARLA run, 26 November (the phantom rule and the side cameras)

The first turn (to point 117, -76) took 33 s from the first move to done, against 46 s on 18 Nov, without touching
anything:
- **Rear phantoms:** the "bus 0 m behind" boxes showed up 35 times and held nothing.
- **Side cameras:** they held the moves for about 4 s for a real motorcycle coming along the street (ground truth:
  closing at 4.4 m/s from 17.5 m). About 1 s more was for boxes with nothing within 25 m.
- **Labels:** that motorcycle, 5 m forward and 17 m right, was labelled "coming ahead"; labels now go by its
  direction.

Later in the run, on a sharp left at Market junction towards another pin, the car cut the inside corner (1.3 m left
of its lane) and pushed on a kerb at full throttle for 29 s, until the town's stall reset lifted it. It now backs off
when wedged (`_wedged`, `_back_off`; see CAMERA_DRIVING.md). Cutting the corner is the junction path's doing and is
not changed.

The phantom rule and the side cameras have now run in CARLA; the wedge recovery has not. The 20 degree rear camera also serves the passing
checks; their offline checks pass, but passing has not been re-run in CARLA.

## Files

| File | Change |
| --- | --- |
| `integration/india_route_planner.py` | new: lane graph, snapping, Dijkstra, `Route`, named places |
| `integration/india_satnav.py` | new: destination state, re-planning, driver message, HUD data |
| `integration/india_hybrid_astar.py` | new: Hybrid A\* on the camera grid |
| `integration/india_camera_driver.py` | `navigate()`, junction mode, arrival, hold at junction |
| `integration/india_combined_traffic.py` | your car in the junction reservations when it has a route |
| `integration/india_combined_scene.py` | sat-nav wiring, map overlay and clicks, command-line options, records |
| `integration/india_town_hud.py` | route on the minimap, AI-panel line, planned path in the AI view, help |
| `setup/run_camera_driving.py`, `setup/carla/run_session.py`, `Start_Camera_Driving.ps1` | destination options |
| `setup/verify_camera_driving.py` | the offline checks above |
| `docs/CAMERA_DRIVING.md`, `docs/FILE_GUIDE.md` | how to use it, results, file entries |
