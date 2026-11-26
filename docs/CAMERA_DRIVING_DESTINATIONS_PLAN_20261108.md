# Destinations for the camera-driven car: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** You pick a destination (a named place, or a click on the town map) and the camera-driven car plans a route, takes
its own turns, and pulls in there.

**Architecture:**

- **Route planner:** builds a lane graph from the town's network file and the map's lane centre lines, and plans routes on it.
- **Sat-nav:** in the viewer process, it turns the car's exact pose into a small message for the driver each control step.
- **Driver between junctions:** the camera driver keeps following the left kerb.
- **Driver at junctions:** it steers along a Hybrid A\* path planned on the camera's top-down road grid, aimed at the route's
  exit. If no path is found, it follows the route line.
- **Turn-taking:** your car joins the town's junction reservations.

**Tech Stack:** Python 3.12, NumPy, pygame; CARLA 0.9.16 (only `carla.Map` offline in checks, and the live world in the
viewer).

**Spec:** `docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md`

## Global Constraints

- The sat-nav uses the **exact** simulator position and heading, with no noise.
- `integration/india_camera_driver.py` and `integration/india_hybrid_astar.py` get no simulator, map, route or actor
  access. The isolation check in `setup/verify_camera_driving.py` enforces this.
- `india_route_planner.py` and `india_satnav.py` never import `carla`. Map lane centres come in through an injected
  `sample(road_id, lane, s) -> (x, y, yaw)` function.
- Junction mode runs from 20 m before a junction's start to the exit street. The guide covers 0–30 m ahead.
- Hybrid A\* parameters:
  - **Grid:** 0.25 m cells. Blocked cells are inflated by the car's half-width + 0.2 m (via the body-disc rule in Task 2).
  - **Search:** 5° heading bins. Five steering angles over 1.0 m arcs, forward only.
  - **Cost:** 0.5 per steering change, plus 0.3 per metre from the guide, plus 1.0 per unknown cell.
  - **Goal:** reached within 1.0 m and 15°.
  - **Limits:** 150 ms or 4,000 expansions.
  - **Re-planning:** every 0.5 s.
- **Driving:**
  - **Pure pursuit:** 4 m look-ahead in a junction.
  - **Junction speeds:** 3.5 m/s; 2.5 m/s on the fallback, which gives up after 10 s blocked.
- **Arrival:**
  - **Under 30 m:** at most 3 m/s, and the keep-left offset is 1.1 m instead of 1.5 m.
  - **Under 1 m:** stop (reason `arrived`).
  - **Under 60 m:** the message carries `arrive_in_m`.
- **Snapping:**
  - **Junctions:** a point inside a junction moves to a street point at least 5 m outside it.
  - **Too far:** more than 15 m from any street is refused.
- **Re-planning triggers:**
  - **Off route:** more than 4 m from the route line.
  - **Reset:** the pose jumps more than 10 m between updates.
- **Turn-taking:** the car holds 2 m short of the junction start when it has not been given its turn.
- **`--destinations`:** the car waits 5 s at each stop, then the next is set.
- **Checks:** these are `setup/verify_*.py` scripts printing PASS/FAIL, not pytest. Evidence goes in `runs/` (gitignored).
- **Commits:** as Atharv, to `main`, with plain messages.
  Commit only when the user has said to commit.
- **Before Task 1:** the 7 Nov camera-driving fixes are still uncommitted in the same files. Ask the user to commit them (or
  say to fold them in) so each task's commit holds only that task.

## Review Focus

1. **Destination just past a stopped bus** (the "Bus stop" place): the car must pass the bus and arrive, not wait behind it
   forever. If the pin is behind the bus's rear, it stops behind the bus and reports arrived, short. Tested in Task 3.
2. **New destination picked mid-junction:** the new route starts on the connector being driven, so the guide doesn't jump.
   Tested in Task 4.
3. **Reset mid-route** (R key or stuck recovery): the sat-nav re-plans and the driver drops its junction path. Tested in Tasks 3 and 4.
4. **Click on a bend junction (A, B, G, H) or right at a junction mouth:** it snaps out of the junction, on the clicked
   side. Tested in Task 1.
5. **Hybrid A\* over its time budget every time** (a slow PC, an unthrottled loop): the fallback takes over, it's logged, and
   the turn still happens. Tested in Tasks 2 and 3.

---

### Task 1: Route planner

**Files:**
- Create: `integration/india_route_planner.py`
- Create: `setup/verify_camera_destinations.py` (its own `check()`/`results`, like `verify_camera_driving.py`; run with
  `.venv-carla` because the route checks build `carla.Map("town", xml)` from `india_combined_map.build_combined()`)

**Interfaces:**
- Produces:
  - `LanePoint` (frozen dataclass: `road_id: int, lane: int, s: float, x: float, y: float, yaw: float`)
  - `TownRoutes(network: dict, sample: Callable[[int, int, float], tuple[float, float, float]], bus_stop_s: float = 180.)`
    with:
    - `.places: dict[str, LanePoint]` (11 entries, as in the spec)
    - `.connector_turn: dict[int, str]` (the turn label of every connector id)
    - `.snap(x, y) -> LanePoint | None`
    - `.locate(x, y, yaw) -> LanePoint` (a street lane or a connector, `lane=1`)
    - `.plan(x, y, yaw, destination: LanePoint) -> Route`
  - `Route`, LoopPath-compatible for the traffic code:
    - fields: `.points (N,3) x/y/yaw`, `.distances (N,)`, `.length`, `.parts`, `.junctions`, `.destination: LanePoint`
    - `.at(s, offset=0.) -> np.ndarray(3)`, clamped to `[0, length]`
    - `.project(x, y, last=0) -> (index, s)`
    - `.part_at(s) -> dict`
    - `.next_junction(s) -> (distance_m, junction_dict)`, with `(1e5, {"node": None, "start_m": 1e5, "end_m": 1e5 + 1., "road_id": None})`
      after the last junction
    - `.progress(x, y, yaw, last=0) -> {"index", "s", "offset_m", "off_route", "remaining_m"}`
  - Each part and junction dict has the keys `road_id, lane, district, node, cruise, start_m, end_m`. Junctions also have
    `name` (from `network["node_names"]`) and `turn` (`"left"|"right"|"straight"|"bend"`).
- Lane choice: the kerb-side lane in each direction (`street.lanes[-1]`, signed by direction: positive lanes run from `a`
  to `b`; the divided road uses lane ±3). Connectors are chosen by matching `from_lane`/`to_lane` to those lanes, or any
  connector between the two streets at that node if no exact one exists.
- Turn label: from the CARLA-yaw change across the connector, Δ = wrap(end − start):
  - Δ > +45° is `right`
  - Δ < −45° is `left`
  - otherwise `straight`
  - every connector at A, B, G and H (two-street nodes) is `bend`

- [ ] **Step 1: Write the failing checks** (`route_checks()` in the new verify script)

```python
routes = TownRoutes(network, sample)            # sample wraps town.get_waypoint_xodr(rid, lane, s).transform
names = list(routes.places)
check("routes: 11 named places", len(names) == 11)
for a in names:
    for b in names:
        if a == b: continue
        p = routes.places[a]; r = routes.plan(p.x, p.y, p.yaw, routes.places[b])
        gaps = np.hypot(*np.diff(r.points[:, :2], axis=0).T)
        assert r.length > 0 and gaps.max() < .65                      # continuous
        assert not any(u["road_id"] == v["road_id"] and u["lane"] == -v["lane"] for u, v in zip(r.parts, r.parts[1:]))  # no U-turn
check("routes: every place reaches every other, continuous, no U-turns", True)
c = routes.connector_turn
check("routes: connector 109 (village road to colony street at C) is a left turn; C, D, E have left, right and straight",
      c[109] == "left" and {"left", "right", "straight"} <= {c[i] for i in range(108, 132)})
check("routes: A, B, G, H connectors are bends", all(c[i] == "bend" for i in (100, 101, 102, 103, 104, 105, 106, 107, 132, 133, 134, 135)))
left_side, right_side = point_beside(street=1, lane=1, s=100., offset=-2.), point_beside(street=1, lane=-1, s=100., offset=-2.)
check("routes: a click on each side of a two-way street snaps to that side's lane",
      routes.snap(*left_side).lane == 1 and routes.snap(*right_side).lane == -1)
p = routes.snap(*NODES_CARLA["C"])                                     # click in the middle of junction C
check("routes: a click inside a junction moves out to a street", p is not None and p.road_id in range(1, 10) and 5. <= p.s)
check("routes: a click far from any street is refused", routes.snap(650., 400.) is None)   # outside the town's extent
on_c = connector_pose(110, s=10.)                                      # pose inside junction C on connector 110
r = routes.plan(*on_c, routes.places["Outskirts bend"])
check("routes: planning from inside a junction starts on that connector", r.parts[0]["road_id"] == 110)
ahead = routes.places["Bazaar gate"]; behind = lane_point(street=1, lane=ahead.lane, s=ahead.s - 40.)
check("routes: a destination behind the car goes round the block", routes.plan(ahead.x, ahead.y, ahead.yaw, behind).length > 400.)
```

(`point_beside`, `connector_pose`, `lane_point` and `NODES_CARLA`, the node coordinates with y negated, are small helpers in
the verify script. Each is built from `sample()`.)

- [ ] **Step 2: Run them and see them fail**

Run: `.venv-carla/Scripts/python.exe setup/verify_camera_destinations.py`
Expected: an ImportError for `india_route_planner`.

- [ ] **Step 3: Implement `india_route_planner.py`**

Sample street lanes and connectors every 0.35 m at s from 0.02 to length − 0.02, in the direction of travel, as
`LoopPath._part` does. Run Dijkstra over (street, lane) nodes, with the edge cost = connector length + the next street's
length, and the first and last streets counted partially. `locate` picks the nearest sampled point whose yaw is within 60° of
the car's heading.

- [ ] **Step 4: Run the checks and see them pass**

Run: `.venv-carla/Scripts/python.exe setup/verify_camera_destinations.py`
Expected: every `routes:` line is PASS.

- [ ] **Step 5: Commit**

```bash
git add integration/india_route_planner.py setup/verify_camera_destinations.py
git commit -m "Plan routes on the connected town's lane graph for picked destinations"
```

### Task 2: Hybrid A\* on the camera grid

**Files:**
- Create: `integration/india_hybrid_astar.py` (NumPy only)
- Modify: `setup/verify_camera_destinations.py` (adds `astar_checks()`)
- Modify: `setup/verify_camera_driving.py` (`isolation_check`: add `india_hybrid_astar.py`, and allow it as an import of
  the driver)

**Interfaces:**
- Produces:
  - `Occupancy` (dataclass: `blocked: bool[nx, ny], unknown: bool[nx, ny], xs: float[nx], ys: float[ny], cell: float`)
  - `occupancy(road, valid, xs, ys, obstacles, half_width, half_length, cell=.25) -> Occupancy`
    - `road` and `valid` are the driver's 0.1 m `GroundGrid` arrays.
    - `obstacles` are driver obstacle dicts (`x, length, y_left, y_right`).
    - A pooled cell is blocked if any visible cell in it is not road, or it lies under an obstacle footprint.
    - Blocked cells are then dilated by the disc radius r = √(hw² + (hl/3)²) + 0.2 m.
    - Unknown means no visible cell in the block.
  - `PlanResult` (dataclass: `path: float[M, 3] | None` (x, y, yaw at 0.25 m spacing, vehicle frame), `reason: "found"|"no_path"|"budget"`,
    `expansions: int, ms: float, cost: float`)
  - `plan(occ, goal: tuple[float, float, float], guide: float[K, 2], wheelbase, max_steer, budget_s=.15, max_expansions=4000) -> PlanResult`
    - starts at (0, 0, 0)
    - steering δ ∈ {−δm, −δm/2, 0, δm/2, δm} with δm = 0.7 · max_steer, over 1.0 m arcs (curvature tan δ / wheelbase),
      checked at 0.25 m sub-steps
    - a state is free when the three disc centres at −2hl/3, 0 and +2hl/3 along the heading fall on unblocked cells
      inside the grid
    - closed-set key: (round(x/0.5), round(y/0.5), round(yaw/5°))
    - goal and costs as in Global Constraints; heuristic: straight-line distance to the goal

- [ ] **Step 1: Write the failing checks** (`astar_checks()`). Use a synthetic crossroads in the vehicle frame: the main road at
  y ∈ [−1.5, 4.5] for x < 14 and x > 20, and the cross road at x ∈ [14, 20] for all y. Build `road`/`valid` directly
  (all visible); the car is at (0, 0, 0) in the left lane.

```python
left = plan(occ, (15.5, -8., -pi/2), guide_left, 2.45, radians(70))
check("astar: left turn found in time, stays in the near half of the cross road",
      left.reason == "found" and left.ms < 150 and dist(left.path[-1][:2], (15.5, -8.)) <= 1. and (left.path[left.path[:, 0] > 14, 0] < 17.3).all())
right = plan(occ, (18.5, 8., pi/2), guide_right, 2.45, radians(70))
check("astar: right turn found, ends in the far half (its own lane)",
      right.reason == "found" and (right.path[right.path[:, 2] > pi/4, 0] > 16.7).all())
straight = plan(occ_with_car_at(8, 12, -.9, .9), (24., 0., 0.), guide_straight, 2.45, radians(70))
check("astar: straight on round a stopped car", straight.reason == "found" and straight.path[:, 1].max() > 1.)
check("astar: walled-off exit gives no_path", plan(occ_with_wall(x=10, x2=11), (24., 0., 0.), guide_straight, 2.45, radians(70)).reason == "no_path")
check("astar: an expansion limit too small gives budget", plan(occ, (18.5, 8., pi/2), guide_right, 2.45, radians(70), max_expansions=50).reason == "budget")
```

- [ ] **Step 2: Run them and see them fail.** Expected: an ImportError for `india_hybrid_astar`.
- [ ] **Step 3: Implement `occupancy` and `plan`.** Dilation is a union of shifted copies over the disc offsets. The open
  list is `heapq` of (f, counter, node), with parents kept for path reconstruction. Re-sample the path to 0.25 m.
- [ ] **Step 4: Run the checks and see them pass,** along with `.venv-carla/Scripts/python.exe setup/verify_camera_driving.py`
  (the isolation line stays PASS).
- [ ] **Step 5: Commit:** `git commit -m "Plan junction turns with Hybrid A* on the camera's road grid"`

### Task 3: The driver follows a route (`navigate`)

**Files:**
- Modify: `integration/india_camera_driver.py`
- Modify: `setup/verify_camera_destinations.py` (adds `driver_checks()`, and `simulate_route()`: the kinematic loop of
  `verify_camera_driving.simulate`, which calls `driver.navigate(message)` before every `control()`, with the message
  built from a synthetic world route polyline)

**Interfaces:**
- Consumes: Task 2's `occupancy` and `plan`.
- Produces:
  - `CameraDriver.navigate(message: dict | None) -> None`. The message has these keys:
    - `mode` (`"lane"|"junction"`)
    - `guide` (float[K, 2], current vehicle frame)
    - `junction` (`{"name", "turn"}` or None)
    - `hold_at_m` (float or None: distance from the car's front to the junction start)
    - `arrive_in_m` (float or None)
  - The driver keeps the message as `self.nav`, which is None without one, storing `guide` in odometry using `self.pose` at
    that moment.
  - It keeps its junction state as `self.junction`, which is None outside junctions:
    `{"path": odometry float[M, 2] | None, "fallback": bool, "planned_at": float, "blocked_since": float | None}`.
  - `navigate(None)` leaves behaviour exactly as today; the 48 existing checks prove it.
  - New `DriverSettings` fields:
    - `junction_speed=3.5`, `junction_fallback_speed=2.5`, `junction_replan_s=.5`, `junction_lookahead=4.`,
      `junction_fallback_give_up_s=10.`
    - `arrive_slow_m=30.`, `arrive_speed=3.`, `arrive_keep_left_m=1.1`, `arrive_stop_m=1.`
    - `hold_gap_m=2.`
    - `no_pass_near_destination_m=40.`
  - `Plan` gains `nav_mode: str = "free"`, `junction_path: np.ndarray | None = None` (this frame's vehicle frame),
    `junction_goal: tuple | None = None`, `plan_found: bool | None = None` and `plan_ms: float | None = None`.
  - New `REASONS` keys, with their texts:
    - `junction_turn`: "Turning at the junction (planned path)"
    - `junction_fallback`: "Turning at the junction (route line)"
    - `waiting_turn`: "Waiting for its turn at the junction"
    - `arriving`: "Arriving at the destination"
    - `arrived`: "Arrived"
    - `no_route`: "No route to the destination"
- **Junction mode, in `_plan`:**
  - **Goal:** the farthest guide point (converted to this frame) with 3 ≤ x ≤ 22, |y| ≤ 8 and a visible grid cell, with the
    guide's heading there.
  - **When to re-plan:** every `junction_replan_s`, or when the stored path has an in-path obstacle.
  - **On success:** the path is stored in odometry.
  - **On `no_path` or `budget`:** the guide is used instead (the fallback, logged).
  - **Following it:** the hits loop and `path_y` use that path while in junction mode, as they do for `pass_hold`.
- **Leaving junction mode:** back to kerb following when the message says `"lane"` and `plan.fit_rows >= 6`. Until then it
  follows the guide.
- **Passing near the destination:** a pass may start within `no_pass_near_destination_m` of the destination only if the
  destination lies more than 8 m beyond the blocker's far end. If the destination is at or behind the blocker's rear and the
  car has waited 5 s within 25 m of it, it stops with reason `arrived` and the detail "arrived N m short (a stopped vehicle is
  at the destination)".
- **Reset:** `reset()` clears the route message and the junction path.

- [ ] **Step 1: Write the failing checks** (`driver_checks()`). Use the Task 2 crossroads as a world road drawn with
  `verify_camera_driving.render`, with the car starting at x = −30.

```python
for turn, (gx, heading) in {"right": (18.5, pi/2), "left": (15.5, -pi/2)}.items():
    log = simulate_route(crossroads, route_through(turn), seconds=40.)
    end = pose_10m_after_exit(log)
    check(f"route: {turn} turn at a crossroads ends in its exit lane", abs(end.x - gx) <= .3 and abs(wrap(end.yaw - heading)) <= radians(10))
log = simulate_route(crossroads, route_through("straight"), seconds=30.)
check("route: straight on through a crossroads stays in lane", max(abs(y) for t, x, y, *_ in log if 20 < x < 35) <= .3)
log = simulate_route(straight_road, route_to(x=80.), seconds=40.)
check("route: arrives within 2 m of the pin, at most 1.3 m from the kerb, and stops",
      abs(log[-1].x + CAR["front_m"] - 80.) <= 2. and log[-1].y + 1.5 <= 1.3 and log[-1].v < .05 and log[-1].why == "arrived")
log = simulate_route(straight_road, route_to(x=200.), seconds=20., hold_at_world_x=12.)
check("route: told to wait its turn, stops 1-2.5 m short of the junction", 9.5 <= log[-1].x + CAR["front_m"] <= 11. and log[-1].why == "waiting_turn")
log = simulate_route(divided_road, route_to(x=bus[1] + 15.), seconds=60., world_objects=[bus], rear=True)
check("route: destination just past a stopped bus: passes it and arrives", log[-1].why == "arrived" and log[-1].x > bus[1])
log = simulate_route(divided_road, route_to(x=bus[0] - 2.), seconds=40., world_objects=[bus], rear=True)
check("route: destination behind a stopped bus: stops behind it and reports arrived short", log[-1].why == "arrived" and log[-1].x < bus[0])
log = simulate_route(crossroads, route_through("right"), seconds=40., settings=DriverSettings(), astar_budget=1e-6)
check("route: Hybrid A* always over budget: the route-line fallback still makes the right turn",
      abs(pose_10m_after_exit(log).x - 18.5) <= .5 and any(r.why == "junction_fallback" for r in log))
d = fresh_driver(); d.navigate(junction_message()); d.reset()
check("route: reset drops the route message and junction path", d.nav is None and d.junction is None)
```

(`astar_budget` goes through a module-level override in `india_hybrid_astar` that the check sets and restores. `log` rows are
named tuples `(t, x, y, v, why, yaw)`. The script's helpers:

- **Roads:** `crossroads` (the Task 2 crossroads in world coordinates), `straight_road` (`straight(-1.5, 4.5)`) and
  `divided_road` (`straight(-1.6, 4.9)`).
- **The bus:** `bus` is the 3.94 m CARLA bus tuple from `verify_camera_driving`.
- **Routes:** `route_through(turn)` and `route_to(x)` return world polylines along the lane centre. `simulate_route` turns
  them into messages.
- **`pose_10m_after_exit(log)`:** the log row 10 m past the cross road's far edge, along the exit.
- **`junction_message()`:** a fixed junction-mode message.)

- [ ] **Step 2: Run them and see them fail.** Expected: AttributeError `navigate`.
- [ ] **Step 3: Implement as described in the Interfaces block.**
- [ ] **Step 4: Run both verify scripts and see them pass.** The 48 existing camera-driving checks must stay PASS.
- [ ] **Step 5: Commit:** `git commit -m "Let the camera driver take route turns with planned junction paths and arrive"`

### Task 4: Sat-nav

**Files:**
- Create: `integration/india_satnav.py` (no `carla` import)
- Modify: `setup/verify_camera_destinations.py` (adds `satnav_checks()`)

**Interfaces:**
- Consumes: Task 1's `TownRoutes`, `Route` and `LanePoint`.
- Produces:
  - `parse_destination(text: str) -> str | tuple[float, float]`: a place name (case-insensitive match against the 11 places,
    otherwise ValueError) or `"x,y"`.
  - `SatNav(routes, junction_lead_m=20., guide_m=30., off_route_m=4., arrive_report_m=60.)` with:
    - `.destination: LanePoint | None`, `.destination_name: str`, `.route: Route | None`, `.arrived: bool`, `.events: list[dict]`
    - `.set_destination(target: str | tuple[float, float], x, y, yaw, sim_time) -> str | None` (an error text, or None)
    - `.clear(sim_time)`
    - `.update(x, y, yaw, sim_time, turn_granted: bool | None = None, driver_arrived: bool = False) -> dict | None`
      (the driver message, or None without a destination)
    - `.hud() -> {"destination", "remaining_m", "next", "route_xy": float[N, 2] world, "pin": (x, y) | None}`
  - Event names: `destination_set`, `route_planned`, `junction_entered`, `junction_left`, `replanned` (with `why`:
    `left_route|reset|new_destination`), `arrived`, `cleared`. Each event is `{"sim_s", "event", ...details}`.
  - `next` reads: "right at Irregular junction in 80 m" / "straight on at Market junction in 40 m" / "follow the road" (for a
    bend) / "Arrived".

- [ ] **Step 1: Write the failing checks**

```python
nav = SatNav(routes); start = routes.places["Market junction"]
nav.set_destination("colony junction", start.x, start.y, start.yaw, 0.)
modes = [(s, nav.update(*nav.route.at(s), s / 5.)["mode"]) for s in np.arange(0., nav.route.length, .5)]
j = nav.route.junctions[0]
first_junction = next(s for s, m in modes if m == "junction")
check("satnav: junction mode starts 20 m before the junction", abs(first_junction - (j["start_m"] - 20.)) <= 1.)
msg = nav.update(*nav.route.at(10.), 2.)
check("satnav: the guide runs 30 m ahead in the car's frame", 28. <= np.hypot(*msg["guide"][-1]) <= 31. and abs(msg["guide"][0][1]) < .5)
check("satnav: arrive_in_m only within 60 m", msg["arrive_in_m"] is None and nav.update(*nav.route.at(nav.route.length - 50.), 99.)["arrive_in_m"] <= 51.)
x, y, yaw = nav.route.at(30.); nav.update(x + 6 * sin(yaw), y - 6 * cos(yaw), yaw, 9.)
check("satnav: 6 m off the route re-plans (left_route)", nav.events[-1]["event"] == "replanned" and nav.events[-1]["why"] == "left_route")
nav.update(*nav.route.at(nav.route.length * .6), 10.)
check("satnav: a 50 m jump re-plans (reset)", nav.events[-1]["why"] == "reset")
inside = nav.route.at(j["start_m"] + 8.); nav.set_destination("Village gate", *inside, 11.)
check("satnav: a new destination mid-junction continues on that connector", nav.route.parts[0]["road_id"] == j["road_id"])
check("satnav: no turn given: hold_at_m set", nav.update(*nav.route.at(j["start_m"] - 15.), 12., turn_granted=False)["hold_at_m"] is not None)
check("satnav: parse_destination", parse_destination("Colony Junction") == "Colony junction" and parse_destination("12.5,-40") == (12.5, -40.))
```

- [ ] **Step 2: Run them and see them fail.** Expected: an ImportError for `india_satnav`.
- [ ] **Step 3: Implement.** `update` projects the pose with `route.progress`. The mode is `"junction"` when
  `start_m − 20 ≤ s ≤ end_m` for some junction. The guide is `route.at(s + k)` for k in 0..30 m (step 1 m), converted to the
  car's frame. `hold_at_m` is `junction.start_m − s − front_m` (front_m = 1.9 by default; the scene passes the real value)
  when `turn_granted is False` and the junction is within 35 m.
- [ ] **Step 4: Run the checks and see them pass.**
- [ ] **Step 5: Commit:** `git commit -m "Add a sat-nav that turns the car's pose and route into driver messages"`

### Task 5: Turn-taking for your car

**Files:**
- Modify: `integration/india_combined_traffic.py`
- Modify: `setup/verify_camera_destinations.py` (adds `route_interface_check()`)

**Interfaces:**
- Consumes: Task 1's `Route`.
- Produces:
  - `CombinedTraffic.set_camera_route(route: Route | None) -> None`
  - `CombinedTraffic.camera_turn() -> {"node": str | None, "granted": bool | None, "distance_m": float | None}`
  - `stats["camera_car_token_grants"]`
- In `update()`, when camera mode has a route:
  - **Position on the route:** `ego.route = route`, `ego.s = route.progress(x, y, yaw, ego.index)["s"]`.
  - **Next junction:** `near[ego] = route.next_junction(ego.s)`.
  - **Queueing:** the queue entry happens as for the other vehicles (`-5 < distance < 35`).
  - **Without a route:** it keeps today's `(1e5, …)` entry and stays out of the reservations.
  - **Unchanged:** the other vehicles' give-way rule for your car.

- [ ] **Step 1: Write the failing check.** A `Route` serves the reservation code's calls the way a `LoopPath` does:
  `at(s)[:2]` lies on the route; `next_junction(s)` returns a distance and a dict with `node`, `start_m`, `end_m` and
  `road_id`; `part_at(s)` has `road_id`; `length` is a float.
- [ ] **Step 2: Run it and see it fail** (if Task 1 left any of these out), then **implement** `set_camera_route`,
  `camera_turn` and the update branch.
- [ ] **Step 3: Run both verify scripts and see them pass.** Turn-taking itself is checked in the Task 7 town run (the
  `camera_car_token_grants` count and the time spent `waiting_turn`).
- [ ] **Step 4: Commit:** `git commit -m "Let the camera car take its turn at junctions when it has a route"`

### Task 6: Viewer, command line and records

**Files:**
- Modify: `integration/india_town_hud.py`
  - adds `MapPicker`
  - the route and pin on the minimap
  - the AI-panel line
  - the junction path (magenta) and goal in `ai_view_surface`
  - `HELP` entries for M and X
- Modify: `integration/india_combined_scene.py`
  - builds `TownRoutes` and `SatNav` in camera mode
  - each tick: `nav.update(...)`, then `camera_driver.navigate(...)`, and `traffic.set_camera_route(nav.route)`
  - keys: M (map) and X (clear)
  - mouse clicks while the map is open; a refused click shows "Not on a street" for 2 s
  - `--destination` / `--destinations`
  - records:
    - `camera_driving/navigation.jsonl` (each sat-nav event)
    - `decisions.jsonl` fields `mode`, `plan_found`, `plan_ms`, `goal`
    - `result.json` → `camera_driving.navigation`
- Modify: `setup/carla/run_session.py`, `setup/run_camera_driving.py`, `Start_Camera_Driving.ps1`. The new options
  `--destination`, `--destinations` (and `-Destination`, `-Destinations`) pass through, only alongside `--camera-drive`.
- Modify: `setup/verify_camera_destinations.py` (adds `viewer_checks()`)

**Interfaces:**
- Consumes: Task 4's `SatNav`, `parse_destination` and `hud()`; Task 5's `set_camera_route` and `camera_turn`.
- Produces:
  - `MapPicker(lines: list[tuple[str, list[tuple[float, float]]]], places: dict[str, tuple[float, float]], size=(1280, 720))`
    with `.to_pixel(x, y) -> (int, int)`, `.to_world(px, py) -> (float, float)`, `.hit_place(px, py) -> str | None`, and
    `.draw(surface, car_xy, route_xy, pin, status: str)`.
  - `result.json` → `camera_driving.navigation`:
    - `destinations` (a list of `{name, set_s, arrived_s | None, short_m | None}`)
    - `replans`
    - `junction_turns` (a list of `{node, turn, planned_frames, fallback_frames}`)
    - `plan_ms` (`{p50, p95}`)
    - `waiting_turn_s`
    - `token_grants`

- [ ] **Step 1: Write the failing checks.**
  - **`MapPicker` round trip:** `to_world(*to_pixel(x, y))` is within 0.6 m for every place. `hit_place` returns the place
    name at its label's pixel and None 40 px away.
  - **Command line:** `run_session.py --help` lists `--destination` and `--destinations`, and passing `--destination`
    without `--camera-drive` exits with an error (subprocess, exit code ≠ 0).
- [ ] **Step 2: Run them and see them fail; implement; run them and see them pass.**
- [ ] **Step 3: Smoke run** (also checks the viewer by eye):

```bash
.venv-carla/Scripts/python.exe setup/run_camera_driving.py --output runs/camera_driving_20261108_smoke01 --sim-seconds 120 --check --destination "Bazaar gate"
```

Expected:
- the run completes
- `navigation.jsonl` has `destination_set`, `route_planned` and `arrived`
- the AI view at a junction shows a magenta path
- an `M` screenshot shows the map with the route

Send the user that screenshot.
- [ ] **Step 4: Commit:** `git commit -m "Pick a destination on the town map and record the drive there"`

### Task 7: Town test and documentation

**Files:**
- Modify: `docs/CAMERA_DRIVING.md` ("Picking a destination": how to, and the results)
- Modify: `docs/FILE_GUIDE.md` (entries for the three new modules and the new verify script)
- Modify: `docs/INDEX.md`

- [ ] **Step 1: Choose the destination list offline.** In `setup/verify_camera_destinations.py --plan-test`, starting from
  the car's spawn pose (`runs/…/actors.json` of a recent run), greedily pick places until the routes together use:
  - a left, a right and a straight turn at C
  - each turn type that exists at D and at E
  - all five districts
  - one destination behind the car

  Print the list and the turns it covers.
- [ ] **Step 2: Run it** (needs CARLA free: ask the other session or the user first if a run is in progress):

```bash
.venv-carla/Scripts/python.exe setup/run_camera_driving.py --output runs/camera_driving_20261108_destinations01 --sim-seconds 900 --check --destinations "<list from step 1>"
```

- [ ] **Step 3: Score it against the spec's "done":**
  - every destination reached
  - 0 collisions involving your car and 0 resets
  - under 5 s on the wrong side
  - the share of junction frames planned by Hybrid A\* rather than by the fallback, reported

  Make a contact sheet of the AI view at each junction turn. If it fails, fix the cause in the owning task's file, add an
  offline check for it, and re-run.
- [ ] **Step 4: If the GPU allows, run the same thing with `--weather dusk`.**
- [ ] **Step 5: Write up the docs** (how to pick a destination, the M and X keys, the command-line options, the results
  table, and limits).
- [ ] **Step 6: Commit:** `git commit -m "Test driving to picked destinations across the town and document it"`
