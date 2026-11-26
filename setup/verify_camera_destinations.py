"""Offline checks for driving the camera car to picked destinations (no CARLA server, no GPU).

  .venv-carla/Scripts/python.exe setup/verify_camera_destinations.py

The route checks build the connected town's map offline (carla.Map from
india_combined_map.build_combined()), so run this with .venv-carla.
"""
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration"))

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)


# ---------------------------------------------------------------- the town, offline
_TOWN = {}


def town():
    """The connected town's network and a lane-centre sampler, built once."""
    if not _TOWN:
        import carla
        from india_combined_map import build_combined, NODES
        xml, network = build_combined()
        native = carla.Map("town", xml)

        def sample(road_id, lane, s):
            t = native.get_waypoint_xodr(int(road_id), int(lane), float(s)).transform
            return t.location.x, t.location.y, math.radians(t.rotation.yaw)
        _TOWN.update(network=network, sample=sample, native=native, nodes={k: (x, -y) for k, (x, y) in NODES.items()})
    return _TOWN


def point_beside(street, lane, s, offset):
    """A point `offset` m to the right (negative: left) of a lane centre, across its travel direction."""
    x, y, yaw = town()["sample"](street, lane, s)
    return x - offset * math.sin(yaw), y + offset * math.cos(yaw)


def connector_pose(connector, s):
    return town()["sample"](connector, 1, s)


def lane_point(street, lane, s):
    from india_route_planner import LanePoint
    return LanePoint(street, lane, s, *town()["sample"](street, lane, s))


# ---------------------------------------------------------------- checks
def route_checks():
    from india_route_planner import TownRoutes, UTURN_MARGIN
    t = town()
    routes = TownRoutes(t["network"], t["sample"])
    names = list(routes.places)
    check("routes: 11 named places", len(names) == 11, ", ".join(names))
    worst_gap, u_turns, flips, bad_turns, failures = 0., 0, 0, [], []
    for a in names:
        for b in names:
            if a == b:
                continue
            p = routes.places[a]
            try:
                r = routes.plan(p.x, p.y, p.yaw, routes.places[b])
            except Exception as error:   # noqa: BLE001 - reported as a failed pair
                failures.append(f"{a} -> {b}: {type(error).__name__}: {error}")
                continue
            gaps = np.hypot(*np.diff(r.points[:, :2], axis=0).T)
            worst_gap = max(worst_gap, float(gaps.max()))
            if r.length <= 0:
                failures.append(f"{a} -> {b}: empty route")
            # A change of direction on one street only through a U-turn leg, on a two-way street, clear of its ends.
            flips += sum(u["road_id"] == v["road_id"] and u["lane"] == -v["lane"] and not u.get("uturn")
                         for u, v in zip(r.parts, r.parts[1:]))
            for part in r.uturns:
                u_turns += 1
                street = routes.streets[part["road_id"]]
                start = routes.point_on(part["road_id"], part["lane"], *r.at(part["start_m"])[:2])
                if street["divided"] or not UTURN_MARGIN - 1. <= start.s <= street["length"] - UTURN_MARGIN + 1.:
                    bad_turns.append((a, b, part["road_id"], round(start.s, 1)))
    check("routes: every place reaches every other, continuous; it turns round only by a U-turn on a two-way street, "
          "25 m or more from its junctions",
          not failures and worst_gap < .65 and flips == 0 and not bad_turns,
          f"largest gap {worst_gap:.2f} m, {u_turns} U-turns in {len(names) * (len(names) - 1)} routes, other reversals {flips}, "
          f"misplaced {bad_turns[:3]}" + ("; " + "; ".join(failures[:3]) if failures else ""))
    c = routes.connector_turn
    check("routes: connector 109 (village road to colony street at C) is a left turn; C, D, E have left, right and straight",
          c.get(109) == "left" and {"left", "right", "straight"} <= {c[i] for i in range(108, 132)},
          f"109 {c.get(109)}; C/D/E labels {sorted({c[i] for i in range(108, 132)})}")
    bends = (100, 101, 102, 103, 104, 105, 106, 107, 132, 133, 134, 135)
    check("routes: A, B, G, H connectors are bends", all(c[i] == "bend" for i in bends), str([c[i] for i in bends]))
    left_side, right_side = point_beside(1, 1, 100., -2.), point_beside(1, -1, 100., -2.)
    a, b = routes.snap(*left_side), routes.snap(*right_side)
    check("routes: a click on each side of a two-way street snaps to that side's lane",
          a is not None and b is not None and a.lane == 1 and b.lane == -1, f"{a and a.lane}, {b and b.lane}")
    p = routes.snap(*t["nodes"]["C"])
    check("routes: a click inside a junction moves out to a street",
          p is not None and p.road_id in range(1, 10) and 5. <= p.s <= routes.streets[p.road_id]["length"] - 5., str(p))
    # Final review (9 Nov): clicks on a bend's arcs or inside a junction snapped to the nearest street sample,
    # often the opposite carriageway 10-27 m away (80 of 192 clicks). They must stay on the connector's own lanes.
    wrong = []
    route_connectors = sorted({cid for options in routes.turns.values() for cid, _, _ in options})
    for cid in route_connectors:
        c, samples = routes.connectors[cid], routes.lanes[(cid, 1)]
        own = {(c["incoming"], int(np.sign(c["from_lane"]))), (c["outgoing"], int(np.sign(c["to_lane"])))}
        for share in (.25, .5, .75):
            _, x, y, yaw = samples[int(share * (len(samples) - 1))]
            for kerb_shift in (0., .7):   # on the lane line, and towards its kerb (the left)
                q = routes.snap(x + kerb_shift * math.sin(yaw), y - kerb_shift * math.cos(yaw))
                inside = q is not None and 5. <= q.s <= routes.streets[q.road_id]["length"] - 5.
                # Off the lane line inside a multi-arm junction the click can be nearer a crossing connector
                # from another arm, where "the clicked side" means nothing: there it need only land on a street.
                strict = kerb_shift == 0. or routes.connector_turn[cid] == "bend"
                ok = inside and (not strict or (q.road_id, int(np.sign(q.lane))) in own)
                if not ok:
                    wrong.append((cid, share, kerb_shift, None if q is None else (q.road_id, q.lane)))
    check("routes: clicks on bend arcs and junction connectors snap to that connector's own lanes",
          not wrong, f"{len(wrong)} wrong of {6 * len(route_connectors)}: {wrong[:4]}")
    check("routes: a click far from any street is refused", routes.snap(650., 400.) is None)
    r = routes.plan(*connector_pose(110, 10.), routes.places["Outskirts bend"])
    check("routes: planning from inside a junction starts on that connector", r.parts[0]["road_id"] == 110,
          str([part["road_id"] for part in r.parts[:3]]))
    ahead = routes.places["Market street"]   # mid-street: room 40 m behind it
    behind = lane_point(1, ahead.lane, ahead.s + 40. * (1 if ahead.lane < 0 else -1))   # 40 m back along its travel
    r = routes.plan(ahead.x, ahead.y, ahead.yaw, behind)
    old = routes.plan(ahead.x, ahead.y, ahead.yaw, behind, u_turns=False, either_side=False).length
    check("routes: a destination 40 m behind the car turns round on its street (a U-turn) instead of going round the block",
          r.length < 150. and len(r.uturns) == 1 and old > 400., f"{r.length:.0f} m with a U-turn; {old:.0f} m round the block")
    # 18 Nov: a pin clicked on the far side of a two-way street ahead was routed 1.8 km round the town to arrive on
    # that side. Either side of the street will do; the route arrives on the nearer one.
    lane = routes.lanes[(1, 1)]
    car = lane[int(len(lane) * .75)]   # Market street, towards Bazaar gate
    pin = routes.snap(117., -76.)
    clicked = routes.plan(car[1], car[2], car[3], pin, either_side=False, u_turns=False)
    either = routes.plan(car[1], car[2], car[3], pin, u_turns=False)
    best = routes.plan(car[1], car[2], car[3], pin)
    check("routes: a pin on a two-way street is reached on either side of it, whichever is nearer",
          either.length < .6 * clicked.length and either.destination.lane == -pin.lane and best.length <= either.length,
          f"clicked side only {clicked.length:.0f} m; either side {either.length:.0f} m; with U-turns {best.length:.0f} m")
    divided = routes._point(2, 3, 150.)
    check("routes: on the divided main road the far carriageway is not 'the same spot'", routes.across(divided) is None)
    return routes


# ---------------------------------------------------------------- synthetic crossroads (vehicle frame)
# Main road y in [-1.5, 4.5] (left kerb at -1.5; the car at (0, 0) in the left lane) for x < 14 and
# x > 20; the cross road x in [14, 20] for all y. India drives on the left: after a left turn (towards
# -y) the car keeps to the near half (x 14-17), after a right turn to the far half (x 17-20).
FINE_XS = np.round(np.arange(2.5, 25. + .05, .1), 4)
FINE_YS = np.round(np.arange(-10., 10. + .05, .1), 4)
CAR_SIZE = {"half_width": .92, "half_length": 1.9}


def crossroads_road(x, y):
    return ((y >= -1.5) & (y <= 4.5)) | ((x >= 14.) & (x <= 20.))


def crossroads_grid():
    gx, gy = np.meshgrid(FINE_XS, FINE_YS, indexing="ij")
    return crossroads_road(gx, gy), np.ones(gx.shape, bool)


GUIDES = {"left": np.array([(0., 0.), (14., 0.), (15.5, -1.5), (15.5, -10.)]),
          "right": np.array([(0., 0.), (16., 0.), (18.5, 2.5), (18.5, 10.)]),
          "straight": np.array([(0., 0.), (25., 0.)])}


def astar_checks():
    from india_hybrid_astar import occupancy, plan
    road, valid = crossroads_grid()
    occ = occupancy(road, valid, FINE_XS, FINE_YS, [], **CAR_SIZE)
    args = (2.45, math.radians(70.))
    left = plan(occ, (15.5, -8., -math.pi / 2), GUIDES["left"], *args)
    ok = left.reason == "found" and left.ms < 150 and math.dist(left.path[-1][:2], (15.5, -8.)) <= 1. \
        and (left.path[left.path[:, 0] > 14., 0] < 17.3).all()
    check("astar: left turn found in time, stays in the near half of the cross road", ok,
          f"{left.reason}, {left.ms:.0f} ms, {left.expansions} expansions")
    right = plan(occ, (18.5, 8., math.pi / 2), GUIDES["right"], *args)
    ok = right.reason == "found" and (right.path[right.path[:, 2] > math.pi / 4, 0] > 16.7).all()
    check("astar: right turn found, ends in the far half (its own lane)", ok,
          f"{right.reason}, {right.ms:.0f} ms, {right.expansions} expansions")
    car = [{"x": 8., "length": 4., "y_left": -.9, "y_right": .9}]
    straight = plan(occupancy(road, valid, FINE_XS, FINE_YS, car, **CAR_SIZE), (24., 0., 0.), GUIDES["straight"], *args)
    check("astar: straight on round a stopped car", straight.reason == "found" and straight.path[:, 1].max() > 1.,
          f"{straight.reason}, max y {straight.path[:, 1].max() if straight.path is not None else float('nan'):.2f} m")
    wall = road & ~((FINE_XS[:, None] >= 10.) & (FINE_XS[:, None] <= 11.))
    walled = plan(occupancy(wall, valid, FINE_XS, FINE_YS, [], **CAR_SIZE), (24., 0., 0.), GUIDES["straight"], *args)
    check("astar: walled-off exit gives no_path", walled.reason == "no_path", walled.reason)
    tiny = plan(occ, (18.5, 8., math.pi / 2), GUIDES["right"], *args, max_expansions=50)
    check("astar: an expansion limit too small gives budget", tiny.reason == "budget", tiny.reason)
    # 8 Nov town runs: only 62 of 446 junction plans were found. The town's lanes are 2.8 m (roads 5.6 m)
    # and the road model's mask has stray not-road pixels inside the road.
    gx, gy = np.meshgrid(FINE_XS, FINE_YS, indexing="ij")
    town = ((gy >= -1.4) & (gy <= 4.2)) | ((gx >= 14.) & (gx <= 19.6))
    speckled = town & (np.random.default_rng(7).random(town.shape) >= .03)
    occ = occupancy(speckled, valid, FINE_XS, FINE_YS, [], **CAR_SIZE)
    found = {name: plan(occ, goal, guide, *args, budget_s=1.).reason for name, goal, guide in (
        ("left", (15.4, -8., -math.pi / 2), np.array([(0., 0.), (14., 0.), (15.4, -1.4), (15.4, -10.)])),
        ("right", (18.2, 8., math.pi / 2), np.array([(0., 0.), (16., 0.), (18.2, 2.2), (18.2, 10.)])),
        ("straight", (24., 0., 0.), GUIDES["straight"]))}
    check("astar: 2.8 m lanes with 3% stray not-road pixels: left, right and straight on all found",
          set(found.values()) == {"found"}, str(found))
    # 9 Nov (destinations05, Bazaar gate): a bicycle ahead on the route. The junction goal was the farthest free
    # route point, right behind it, where the car's centre fits but not its body: A* ran to its time limit, or
    # found a path that swung the car out (it turned right off a left bend into the sand).
    import verify_camera_driving as vcd
    from india_hybrid_astar import disc_offsets
    driver = vcd.fresh_driver()
    guide = np.array([(x, 0.) for x in np.arange(0., 22.01, .5)])
    results = []
    for xb in (7.4, 8.0, 8.6, 10.):
        bike = [{"x": xb, "length": 1.8, "y_left": -.35, "y_right": .35}]
        occ = occupancy(town, valid, FINE_XS, FINE_YS, bike, **CAR_SIZE)
        goal = driver._junction_goal(guide, occ, bike)
        body = [] if goal is None else [(goal[0] + d * math.cos(goal[2]), goal[1] + d * math.sin(goal[2]))
                                        for d in disc_offsets(occ.half_length)]
        fits = all(not occ.blocked[int((bx - occ.xs[0] + occ.cell / 2) / occ.cell), int((by - occ.ys[0] + occ.cell / 2) / occ.cell)]
                   for bx, by in body)
        behind = goal is None or max(bx for bx, _ in body) < xb
        results.append((xb, None if goal is None else round(goal[0], 2), behind and fits))
    check("astar: a vehicle ahead on the route: the junction goal stays behind it, with room for the whole car",
          all(ok for _, _, ok in results), str(results))


# ---------------------------------------------------------------- the driver on a route (closed loop)
def polyline(*pieces, step=.5):
    """Join straight segments and arcs into a dense world polyline (x, y)."""
    points = []
    for piece in pieces:
        points.extend(piece)
    points = np.array(points, float)
    keep = np.r_[True, np.hypot(*np.diff(points, axis=0).T) > 1e-6]
    points = points[keep]
    lengths = np.r_[0., np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    s = np.arange(0., lengths[-1] + 1e-9, step)
    return np.stack([np.interp(s, lengths, points[:, 0]), np.interp(s, lengths, points[:, 1])], axis=1)


def arc(cx, cy, radius, a0, a1, n=40):
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in np.linspace(a0, a1, n)]


def route_through(turn):
    """Lane-centre route from x = -30 through the crossroads (left lane y = 0; exits x 18.5 right, 15.5 left)."""
    if turn == "right":
        return polyline([(-30., 0.), (13., 0.)], arc(13., 5.5, 5.5, -math.pi / 2, 0.), [(18.5, 5.5), (18.5, 70.)])
    if turn == "left":
        return polyline([(-30., 0.), (13., 0.)], arc(13., -2.5, 2.5, math.pi / 2, 0.), [(15.5, -2.5), (15.5, -70.)])
    return polyline([(-30., 0.), (90., 0.)])


def route_to(x):
    return polyline([(-30., 0.), (float(x), 0.)])


Row = __import__("collections").namedtuple("Row", "t x y v why yaw")


def _junction_span(route):
    """Route distances where it enters and leaves the crossroads box (x 14-20, y -1.5..4.5), or None."""
    inside = (route[:, 0] >= 14.) & (route[:, 0] <= 20.) & (route[:, 1] >= -1.5) & (route[:, 1] <= 4.5)
    if not inside.any() or (route[:, 1].max() <= 4.5 and route[:, 1].min() >= -1.5):
        return None   # straight through: no turn, but the crossroads is still a junction
    lengths = np.r_[0., np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    idx = np.flatnonzero(inside)
    return lengths[idx[0]], lengths[idx[-1]] + 2.


def simulate_route(road, route, seconds=40., settings=None, world_objects=(), rear=False, hold_at_world_x=None,
                   astar_budget=None, start=(-30., 0., 0.), seen_road=None):
    """The kinematic car of verify_camera_driving.simulate, told its route every step (a stand-in sat-nav
    with the exact pose). Junction mode runs from 20 m before the crossroads to 2 m past its far edge."""
    import verify_camera_driving as vcd
    import india_hybrid_astar
    from india_camera_driver import Perception
    driver = vcd.fresh_driver(settings, rear=rear)
    lengths = np.r_[0., np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    span = _junction_span(route)
    if span is None and route[:, 0].max() > 20. and route[:, 0].min() < 14. and abs(route[:, 1]).max() < .1:
        span = (float(np.interp(14., route[:, 0], lengths)), float(np.interp(22., route[:, 0], lengths)))
    if span is not None and abs(route[-1, 1]) < .1 and route[-1, 0] < 22.:
        span = None   # route ends before the crossroads (arrival and hold checks)
    x, y, psi = start
    vel, r = 0., 0.
    due, log = {}, []
    saved = india_hybrid_astar.FORCE_BUDGET_S
    india_hybrid_astar.FORCE_BUDGET_S = astar_budget
    try:
        for it in range(int(seconds / .05)):
            t = it * .05
            if it in due:
                frame, mask, boxes, back = due.pop(it)
                driver.observe(Perception(frame, True, mask=mask, boxes=boxes, rear_boxes=back), t)
            i = int(np.argmin(np.hypot(route[:, 0] - x, route[:, 1] - y)))
            s = lengths[i]
            ahead = route[(lengths >= s) & (lengths <= s + 30.)]
            dx, dy = ahead[:, 0] - x, ahead[:, 1] - y
            guide = np.stack([dx * math.cos(psi) + dy * math.sin(psi), -dx * math.sin(psi) + dy * math.cos(psi)], axis=1)
            remaining = lengths[-1] - s
            junction = span is not None and span[0] - 20. <= s <= span[1]
            hold = None if hold_at_world_x is None else hold_at_world_x - (x + vcd.CAR["front_m"])
            driver.navigate({"mode": "junction" if junction else "lane", "guide": guide,
                             "junction": {"name": "Crossroads", "turn": "right"} if junction else None,
                             "hold_at_m": hold, "arrive_in_m": remaining if remaining < 60. else None})
            out = driver.control(abs(vel), r, t)
            delta = out["steer"] * vcd.CAR["max_steer"]
            r = vel * math.tan(delta) / vcd.CAR["wheelbase"]
            psi += r * .05
            x += vel * math.cos(psi) * .05
            y += vel * math.sin(psi) * .05
            gear = -1. if out.get("reverse") else 1.
            speed = abs(vel)
            if speed < 1e-6 or np.sign(vel) == gear:
                speed = max(0., speed + (3.5 * out["throttle"] - 7. * out["brake"] - .03 * speed) * .05)
                vel = gear * speed
            else:
                vel = np.sign(vel) * max(0., speed - (3.5 * out["throttle"] + 7. * out["brake"]) * .05)
            if it % 2 == 0:
                mask, boxes = vcd.render(seen_road or road, vcd.car_frame_objects(world_objects, (x, y, psi)), pose=(x, y, psi))
                back = vcd.rear_boxes(vcd.car_frame_all(world_objects, (x, y, psi))) if rear and driver.wants_rear else None
                driver.captured(it, abs(vel), r)
                due[it + 2] = (it, mask, boxes, back)
            log.append(Row(t, x, y, abs(vel), out["reason"], psi))
    finally:
        india_hybrid_astar.FORCE_BUDGET_S = saved
    return log


def pose_after_exit(log, turn, beyond=10.):
    """The first log row 10 m past the crossroads' far edge along the exit."""
    if turn == "right":
        return next((row for row in log if row.y >= 4.5 + beyond), log[-1])
    return next((row for row in log if row.y <= -1.5 - beyond), log[-1])


def junction_message():
    return {"mode": "junction", "guide": GUIDES["right"].copy(), "junction": {"name": "Crossroads", "turn": "right"},
            "hold_at_m": None, "arrive_in_m": None}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def driver_checks():
    import verify_camera_driving as vcd
    from india_camera_driver import DriverSettings
    straight_road, divided_road = vcd.straight(-1.5, 4.5), vcd.straight(-1.6, 4.9)
    bus = (60., 70.3, .14 - 1.97, .14 + 1.97, 3.2, "bus", .45)
    for turn, (gx, heading) in {"right": (18.5, math.pi / 2), "left": (15.5, -math.pi / 2)}.items():
        log = simulate_route(crossroads_road, route_through(turn), seconds=40.)
        end = pose_after_exit(log, turn)
        check(f"route: {turn} turn at a crossroads ends in its exit lane",
              abs(end.x - gx) <= .3 and abs(wrap(end.yaw - heading)) <= math.radians(10.),
              f"10 m after the exit: x {end.x:.2f} (lane {gx}), heading {math.degrees(end.yaw):.0f} deg; "
              f"planned {sum(r.why == 'junction_turn' for r in log)} / fallback {sum(r.why == 'junction_fallback' for r in log)} steps")
    log = simulate_route(crossroads_road, route_through("straight"), seconds=30.)
    worst = max((abs(row.y) for row in log if 20. < row.x < 35.), default=float("nan"))
    check("route: straight on through a crossroads stays in lane", worst <= .3, f"max |y| {worst:.2f} m")
    log = simulate_route(straight_road, route_to(80.), seconds=40.)
    end = log[-1]
    check("route: arrives within 2 m of the pin, at most 1.3 m from the kerb, and stops",
          abs(end.x - 80.) <= 2. and end.y + 1.5 <= 1.3 and end.v < .05 and end.why == "arrived",
          f"centre at x {end.x:.2f} (pin 80), {end.y + 1.5:.2f} m from the kerb, {end.v:.2f} m/s, {end.why}")
    log = simulate_route(straight_road, route_to(200.), seconds=20., hold_at_world_x=12.)
    front = log[-1].x + vcd.CAR["front_m"]
    check("route: told to wait its turn, stops 1-2.5 m short of the junction", 9.5 <= front <= 11. and log[-1].why == "waiting_turn",
          f"front at {front:.2f} m (junction at 12), {log[-1].why}")
    log = simulate_route(divided_road, route_to(bus[1] + 15.), seconds=60., world_objects=[bus], rear=True)
    check("route: destination just past a stopped bus: passes it and arrives",
          log[-1].why == "arrived" and log[-1].x > bus[1], f"ended at x {log[-1].x:.1f}, {log[-1].why}")
    log = simulate_route(divided_road, route_to(bus[0] - 2.), seconds=40., world_objects=[bus], rear=True)
    check("route: destination behind a stopped bus: stops behind it and reports arrived short",
          log[-1].why == "arrived" and log[-1].x < bus[0], f"ended at x {log[-1].x:.1f}, {log[-1].why}")
    # Final review (9 Nov): any vehicle it waited behind for 5 s within 25 m of the pin counted as "arrived short",
    # so a queue 15-20 m before the pin (the junction places sit 15 m past a junction) ended the trip early.
    queued = (65., 69., -.9, .9, 1.5, "car")
    log = simulate_route(straight_road, route_to(80.), seconds=30., world_objects=[queued])
    check("route: a car queuing 11 m before the pin: waits behind it, does not report arrived",
          not any(row.why == "arrived" for row in log) and log[-1].x < queued[0], f"ended at x {log[-1].x:.1f}, {log[-1].why}; "
          f"arrived reported: {any(row.why == 'arrived' for row in log)}")
    log = simulate_route(crossroads_road, route_through("right"), seconds=40., settings=DriverSettings(), astar_budget=1e-6)
    end = pose_after_exit(log, "right")
    check("route: Hybrid A* always over budget: the route-line fallback still makes the right turn",
          abs(end.x - 18.5) <= .5 and any(row.why == "junction_fallback" for row in log),
          f"x {end.x:.2f}; fallback steps {sum(row.why == 'junction_fallback' for row in log)}")
    d = vcd.fresh_driver()
    d.navigate(junction_message())
    d.reset()
    check("route: reset drops the route message and junction path", d.nav is None and d.junction is None)
    # The road model cannot see the mouth of the turn (8 Nov smoke run, junction B): the only goals in view are
    # a few metres straight ahead. The car must still take the turn along the route line, not drive on.
    hidden = lambda x, y: crossroads_road(x, y) & ~((x >= 14.) & (x <= 20.) & (y <= -1.5) & (y >= -8.))
    log = simulate_route(crossroads_road, route_through("left"), seconds=40., seen_road=hidden)
    end = pose_after_exit(log, "left")
    check("route: a left turn the camera cannot see into still follows the route into its exit lane",
          abs(end.x - 15.5) <= .5 and abs(wrap(end.yaw + math.pi / 2)) <= math.radians(15.),
          f"10 m after the exit: x {end.x:.2f} (lane 15.5), heading {math.degrees(end.yaw):.0f} deg; "
          f"fallback steps {sum(r.why == 'junction_fallback' for r in log)}")


def simulate_uturn(seconds=60., road=None, edges=(-1.5, 4.5), across=3., world_objects=(), movers=(), at=20., ghosts=None,
                   sides=True, walls=()):
    """A route that turns round at x = `at` on a straight two-way street (left lane centre y = 0, the other lane's
    `across` to the right), back to a pin 60 m behind; the kinematic car of simulate_route, but turning about its
    rear axle (half a wheelbase behind the centre) as a car does on full lock. `ghosts(t)`: extra rear-camera boxes
    (pixels, detector rows) with nothing there. The rear and side cameras send one picture a frame in turn, as in
    the town (sides=False: the rear one only). `walls`: world boxes the car cannot enter that no camera shows (a
    kerb): a step into one is undone and the car stops dead. Returns the log and the driver."""
    import verify_camera_driving as vcd
    from india_camera_driver import Perception
    road = road or vcd.straight(*edges)
    r = across / 2
    route = polyline([(-30., 0.), (at, 0.)], arc(at, r, r, -math.pi / 2, math.pi / 2, 24), [(at, across), (at - 60., across)])
    turn_s = at + 30.
    driver = vcd.fresh_driver(rear=True, sides=sides)
    lengths = np.r_[0., np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    x, y, psi = -30., 0., 0.
    vel, rate = 0., 0.
    due, log = {}, []
    half = vcd.CAR["wheelbase"] / 2
    for it in range(int(seconds / .05)):
        t = it * .05
        objects = list(world_objects) + [m(t) for m in movers]
        if it in due:
            frame, mask, boxes, back, side = due.pop(it)
            driver.observe(Perception(frame, True, mask=mask, boxes=boxes, rear_boxes=back, side_boxes=side), t)
        i = int(np.argmin(np.hypot(route[:, 0] - x, route[:, 1] - y)))
        s = lengths[i]
        ahead = route[(lengths >= s) & (lengths <= s + 30.)]
        c, sn = math.cos(psi), math.sin(psi)
        dx, dy = ahead[:, 0] - x, ahead[:, 1] - y
        guide = np.stack([dx * c + dy * sn, -dx * sn + dy * c], axis=1)
        remaining = lengths[-1] - s
        uturn = None
        if -3. <= turn_s - s <= 60. and abs(math.remainder(psi, 2 * math.pi)) < math.radians(45.):
            ddx, ddy = at - x, -y
            uturn = {"in_m": turn_s - s, "pose": (ddx * c + ddy * sn, -ddx * sn + ddy * c, -psi),
                     "across_m": across, "left_m": -edges[0], "right_m": edges[1]}
        driver.navigate({"mode": "lane", "guide": guide, "junction": None, "hold_at_m": None,
                         "arrive_in_m": remaining if remaining < 60. else None, "uturn": uturn})
        out = driver.control(abs(vel), rate, t)
        rate = vel * math.tan(out["steer"] * vcd.CAR["max_steer"]) / vcd.CAR["wheelbase"]
        before = (x, y, psi)
        rx, ry = x - half * c, y - half * sn
        psi += rate * .05
        rx, ry = rx + vel * math.cos(psi) * .05, ry + vel * math.sin(psi) * .05
        x, y = rx + half * math.cos(psi), ry + half * math.sin(psi)
        gear = -1. if out.get("reverse") else 1.
        speed = abs(vel)
        if speed < 1e-6 or np.sign(vel) == gear:
            speed = max(0., speed + (3.5 * out["throttle"] - 7. * out["brake"] - .03 * speed) * .05)
            vel = gear * speed
        else:
            vel = np.sign(vel) * max(0., speed - (3.5 * out["throttle"] + 7. * out["brake"]) * .05)
        if any(overlaps(Row(t, x, y, 0., "", psi), wall) for wall in walls):
            (x, y, psi), vel = before, 0.   # held by the kerb
        if it % 2 == 0:
            mask, boxes = vcd.render(road, vcd.car_frame_objects(objects, (x, y, psi)), pose=(x, y, psi))
            back, side = vcd.extra_view(driver, vcd.car_frame_all(objects, (x, y, psi)), it // 2)
            if back is not None and ghosts is not None:
                back = np.vstack([back, np.asarray(ghosts(t), np.float32).reshape(-1, 6)])
            driver.captured(it, abs(vel), rate)
            due[it + 2] = (it, mask, boxes, back, side)
        log.append(Row(t, x, y, abs(vel), out["reason"], psi))
    return log, driver


def body_corners(row):
    import verify_camera_driving as vcd
    hl, hw = vcd.CAR["half_length"], vcd.CAR["half_width"]
    c, s = math.cos(row.yaw), math.sin(row.yaw)
    return [(row.x + a * c - b * s, row.y + a * s + b * c) for a, b in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw))]


def overlaps(row, box):
    """The car's body (at a log row) and a world box (x0, x1, y0, y1, ...) overlap (separating axes)."""
    x0, x1, y0, y1 = box[:4]
    corners = np.array(body_corners(row))
    other = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    for axis in ((1., 0.), (0., 1.), (math.cos(row.yaw), math.sin(row.yaw)), (-math.sin(row.yaw), math.cos(row.yaw))):
        a, b = corners @ axis, other @ axis
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def uturn_checks():
    import verify_camera_driving as vcd

    def inside(log):
        ys = [cy for row in log for _, cy in body_corners(row)]
        return min(ys), max(ys)

    def moves(driver):
        return sum(e[1] == "uturn_move" for e in driver.events)
    log, d = simulate_uturn()
    lo, hi = inside(log)
    end = log[-1]
    done = [e for e in d.events if e[1] == "uturn_done"]
    check("uturn: turns round on a 6 m two-way street in three moves, the body inside the road, then drives to the pin",
          done and moves(d) == 3 and lo >= -1.5 and hi <= 4.5 and end.why == "arrived" and abs(end.y - 3.) < .7
          and abs(wrap(end.yaw - math.pi)) < math.radians(8.),
          f"{moves(d)} moves; body y {lo:.2f}..{hi:.2f} (road -1.5..4.5); end x {end.x:.1f} y {end.y:.2f} "
          f"heading {math.degrees(wrap(end.yaw)):.0f}, {end.why}")
    saved = vcd.CAR["max_steer"]
    vcd.CAR["max_steer"] = math.radians(35.)
    try:
        log, d = simulate_uturn(seconds=75.)
    finally:
        vcd.CAR["max_steer"] = saved
    lo, hi = inside(log)
    check("uturn: with a 35 degree steering lock it takes more moves, still inside the road",
          any(e[1] == "uturn_done" for e in d.events) and moves(d) >= 4 and lo >= -1.5 and hi <= 4.5,
          f"{moves(d)} moves; body y {lo:.2f}..{hi:.2f}")
    # A car coming the other way in the far lane: wait for it to go by, then turn.
    oncoming = lambda t: (75. - 6. * t, 79. - 6. * t, 2.1, 3.9, 1.5, "car")
    log, d = simulate_uturn(movers=[oncoming], seconds=70.)
    started = next((e[0] for e in d.events if e[1] == "uturn_move"), None)
    hit = [round(row.t, 1) for row in log if overlaps(row, oncoming(row.t))]
    passed = (79. - 20.) / 6.   # its rear passes the turn's start
    check("uturn: an oncoming car in the far lane: waits for it to pass, then turns round, no contact",
          started is not None and started > passed and not hit and any(e[1] == "uturn_done" for e in d.events),
          f"first move at {started} s (car past at {passed:.1f} s); contact at {hit[:3]}")
    parked = (45., 49., 2.1, 3.9, 1.5, "car")
    log, d = simulate_uturn(world_objects=[parked])
    stopped = next((e[0] for e in d.events if e[1] == "uturn_started"), None)
    started = next((e[0] for e in d.events if e[1] == "uturn_move"), None)
    check("uturn: a car parked 25 m ahead in the far lane does not hold the turn up",
          stopped is not None and started is not None and started - stopped < 5.,
          f"stopped at {stopped} s, first move at {started} s")
    # 18 Nov CARLA run: things off the carriageway held the turn. A food stall on the pavement behind (read as
    # "truck 1.0 m behind") stopped the backing move for good; people standing on the pavement held the gap.
    # It stands there only once the car has stopped to turn (11 s): the kinematic approach would otherwise try to
    # pass it, and start the turn from the far lane.
    stall = (14., 20., -4.5, -1.9, 2.5, "truck")
    log, d = simulate_uturn(movers=[lambda t: stall if t > 11.5 else (500., 506., -4.5, -1.9, 2.5, "truck")], seconds=70.)
    lo, hi = inside(log)
    hit = [round(row.t, 1) for row in log if overlaps(row, stall)]
    check("uturn: a stall on the pavement beside the turn does not stop the backing move; no contact",
          any(e[1] == "uturn_done" for e in d.events) and not hit and lo >= -1.5 and hi <= 4.5,
          f"{moves(d)} moves; body y {lo:.2f}..{hi:.2f}; contact at {hit[:3]}; "
          f"events {[e[1] for e in d.events if e[1].startswith('uturn_') and e[1] != 'uturn_move']}")

    def first_move(objects, seconds=40.):
        log, d = simulate_uturn(world_objects=objects, seconds=seconds)
        stopped = next((e[0] for e in d.events if e[1] == "uturn_started"), None)
        started = next((e[0] for e in d.events if e[1] == "uturn_move"), None)
        return stopped, started
    stopped, started = first_move([(27., 27.5, 4.8, 5.3, 1.7, "person")])
    check("uturn: a person standing on the far pavement 7 m ahead does not hold the turn up",
          stopped is not None and started is not None and started - stopped < 5.,
          f"stopped at {stopped} s, first move at {started} s")
    stopped, started = first_move([(27., 27.5, 2.8, 3.3, 1.7, "person")], seconds=30.)
    check("uturn: a person standing in the far lane 7 m ahead is waited for",
          stopped is not None and started is None, f"stopped at {stopped} s, first move at {started}")
    stopped, started = first_move([(6., 14., -1.4, 1.4, 3., "bus")])
    check("uturn: a bus standing straight behind it is not taken for one overtaking on its right",
          stopped is not None and started is not None and started - stopped < 5.,
          f"stopped at {stopped} s, first move at {started} s")
    # Across the street, traffic along it is beside the car, where only the side cameras see it. A car in the far
    # lane, too far off to wait for when the turn starts (6 m/s from 150 m), comes up during the backing move and
    # stops 7 m short: the car holds where it is until it has stopped, then carries on.
    def coming_car(t):
        front = max(31., 150. - 6. * t)
        return (front, front + 4.2, 2.1, 3.9, 1.5, "car")
    log, d = simulate_uturn(movers=[coming_car], seconds=70.)
    hit = [round(row.t, 1) for row in log if overlaps(row, coming_car(row.t))]
    held = [e for e in d.events if e[1] == "uturn_move"]
    near = [row for row in log if 150. - 6. * row.t > 31. and coming_car(row.t)[0] - max(cx for cx, _ in body_corners(row)) < 12.]
    moving = [round(row.t, 1) for row in near if row.v > .15]
    took = next((e[0] for e in d.events if e[1] == "uturn_done"), None)
    check("uturn: a car coming along the far lane mid-turn (seen by the side camera): holds where it is until it stops",
          took is not None and not hit and near and not moving,
          f"turned round at {took} s (without it {done[0][0]} s); moving while it came within 12 m at {moving[:3]}; "
          f"contact at {hit[:3]}")
    parked = (37., 41., 2.1, 3.9, 1.5, "car")   # 14 m along the street from the turn
    log, d = simulate_uturn(world_objects=[parked], seconds=60.)
    took = next((e[0] for e in d.events if e[1] == "uturn_done"), None)
    check("uturn: a car parked 14 m along the far lane, beside the turn, is not taken for one coming",
          took is not None and took < done[0][0] + 3. and not any(overlaps(row, parked) for row in log),
          f"turned round at {took} s (without it {done[0][0]} s)")
    # A kerb 0.5 m inside the road's edge on the map, which no camera shows (26 Nov: a car wedged on a kerb pushed
    # for 29 s): a move that reaches it ends there, as at the edge, and the turn still finishes.
    kerb = (5., 40., 4., 6., .15, "kerb")
    log, d = simulate_uturn(seconds=70., walls=[kerb])
    stuck = [e[2] for e in d.events if e[1] == "uturn_move" and "stuck" in e[2]]
    took = next((e[0] for e in d.events if e[1] == "uturn_done"), None)
    check("uturn: a kerb the map does not show (0.5 m inside its edge): the move ends at it and the turn still finishes",
          took is not None and stuck, f"turned round at {took} s in {moves(d)} moves; ended by the kerb: {stuck}")
    # 18 Nov CARLA rerun: boxes over most of the rear picture with nothing there held the backing moves for about
    # 20 s. These two read exactly as they did there: "bus 0.0 m behind" (cut at the bottom), "autorickshaw 0.7 m".
    from india_camera_driver import Perception
    ghost = np.array([[20., 10., 740., 539., .8, vcd.CLASSES.index("bus")]], np.float32)
    rick = np.array([[80., 20., 900., 525., .8, vcd.CLASSES.index("autorickshaw")]], np.float32)
    log, d = simulate_uturn(seconds=70., ghosts=lambda t: ghost if int(t * 10) % 3 else rick)
    took = next((e[0] for e in d.events if e[1] == "uturn_done"), None)
    check("uturn: boxes over most of the rear picture with nothing there (bus 0 m, autorickshaw 0.7 m behind) do not hold it",
          took is not None and took < done[0][0] + 3. and moves(d) == 3,
          f"turned round at {took} s (without them {done[0][0]} s), {moves(d)} moves")
    # Nothing gets that close behind unseen, so a box that close counts only if something was read farther out
    # first (or the camera has just started), and the car has not driven half a metre forward since.
    mask, boxes = vcd.render(vcd.straight(), [])

    def look(d, k, objects, extra=None):
        back = vcd.rear_boxes(objects)
        back = back if extra is None else np.vstack([back, extra])
        d.observe(Perception(k, True, mask=mask, boxes=boxes, rear_boxes=back), k * .1)
        return [d._rear_believed(o) for o in d.rear_obstacles if o["behind_m"] < d.settings.uturn_rear_seen_m]
    d = vcd.fresh_driver(rear=True)
    for k in range(15):
        look(d, k, [])
    d.pose = (1., 0., 0.)   # drives a metre forward
    fake = look(d, 15, [], ghost)
    for k in range(16, 40):   # a car comes up from 8 m behind and stops 0.5 m from the tail
        gap = max(.5, 8. - .4 * (k - 16))
        tail = -vcd.CAR["half_length"] - gap
        real = look(d, k, [(tail - 4.2, tail, -.9, .9, 1.5, "car")])
    d = vcd.fresh_driver(rear=True)
    at_start = look(d, 0, [(-2.4 - 4.2, -2.4, -.9, .9, 1.5, "car")])
    check("uturn: a car seen coming up behind still counts at 0.5 m (or right behind when the camera starts); "
          "a box that close out of nowhere does not",
          fake and not any(fake) and real and all(real) and at_start and all(at_start),
          f"out of nowhere {fake}, came up {real}, at the start {at_start}")


def satnav_checks(routes):
    from india_satnav import SatNav, parse_destination
    nav = SatNav(routes)
    start = routes.places["Market junction"]
    nav.set_destination("colony junction", start.x, start.y, start.yaw, 0.)
    route = nav.route
    modes = [(s, nav.update(*route.at(s), s / 5.)["mode"]) for s in np.arange(0., route.length, .5)]
    j = route.junctions[0]
    first = next(s for s, m in modes if m == "junction")
    check("satnav: junction mode starts 20 m before the junction", abs(first - (j["start_m"] - 20.)) <= 1.,
          f"junction mode from {first:.1f} m, junction at {j['start_m']:.1f} m")
    check("satnav: arrival reported at the end of the route", nav.arrived and nav.events[-1]["event"] == "arrived")
    # 8 Nov town run: the driver still reported "arrived" on the step after the next destination was set,
    # so a destination 1 km away was marked reached at once.
    end = nav.route.at(nav.route.length)
    nav.set_destination("Market junction", *end, 1.)
    nav.update(*end, 1.05, driver_arrived=True)
    check("satnav: a stale 'arrived' from the driver does not reach a new, far destination",
          not nav.arrived and nav.events[-1]["event"] != "arrived", f"last event {nav.events[-1]['event']}")
    nav.set_destination("Colony junction", start.x, start.y, start.yaw, 1.)
    msg = nav.update(*nav.route.at(10.), 2.)
    reach = float(np.hypot(*np.diff(msg["guide"], axis=0).T).sum())   # along the line (it may fold back at a U-turn)
    check("satnav: the guide runs 30 m ahead in the car's frame", 28. <= reach <= 31. and abs(msg["guide"][0][1]) < .5,
          f"guide {reach:.1f} m long, first point {msg['guide'][0][1]:+.2f} m to the side")
    near_end = nav.update(*nav.route.at(nav.route.length - 50.), 99.)   # a jump: re-planned from there
    check("satnav: arrive_in_m only within 60 m", msg["arrive_in_m"] is None and near_end["arrive_in_m"] is not None
          and near_end["arrive_in_m"] <= 51., f"{msg['arrive_in_m']}, {near_end['arrive_in_m']}")
    nav.set_destination("Colony junction", start.x, start.y, start.yaw, 100.)
    # A wrong turn: on the village road towards C, routed straight on (connector 110), the car takes the
    # tight left (connector 109) onto the colony street instead.
    near_c = routes._point(3, 1, routes.streets[3]["length"] - 30.)
    nav.set_destination("Colony junction", near_c.x, near_c.y, near_c.yaw, 100.)
    wrong = np.vstack([routes._travel(3, 1, near_c.s, None)[:, 1:4], routes.lanes[(109, 1)][:, 1:4],
                       routes._travel(7, -1)[:40, 1:4]])
    for k, (x, y, yaw) in enumerate(wrong):
        nav.update(x, y, yaw, 100. + .05 * k)
    whys = [e.get("why") for e in nav.events if e["event"] == "replanned" and e["sim_s"] >= 100. and e.get("why") != "new_destination"]
    check("satnav: a wrong turn re-plans once (left_route)", whys == ["left_route"], str(whys))
    nav.update(*nav.route.at(nav.route.length * .6), 102.)
    jumped = [e.get("why") for e in nav.events if e["event"] == "replanned" and e["sim_s"] == 102.]
    check("satnav: a 50 m jump re-plans (reset)", jumped == ["reset"], str(jumped))
    nav.set_destination("Colony junction", start.x, start.y, start.yaw, 103.)
    j = nav.route.junctions[0]
    inside = nav.route.at(j["start_m"] + 8.)
    nav.update(*nav.route.at(j["start_m"] + 7.5), 104.)
    nav.set_destination("Village gate", *inside, 104.1)
    check("satnav: a new destination mid-junction continues on that connector", nav.route.parts[0]["road_id"] == j["road_id"],
          f"{nav.route.parts[0]['road_id']} vs {j['road_id']}")
    # Final review (9 Nov): 0.5-1 m off the connector's centre line (routine in a turn), the planner located the car on
    # a sibling connector to another exit, so the guide jumped. The new route must start on the connector being driven.
    jumped = []
    for cid in (108, 109, 110, 122, 123, 125):
        for s_in in (1., 2., 4.):
            x, y, yaw = connector_pose(cid, s_in)
            probe = SatNav(routes)
            probe.set_destination("Village gate", x, y, yaw, 0.)
            if probe.route.parts[0]["road_id"] != cid:
                continue   # (a pose exactly on it always starts there: checked above)
            for side in (-1., -.5, .5, 1.):
                ox, oy = x - side * math.sin(yaw), y + side * math.cos(yaw)
                probe.update(ox, oy, yaw, 1.)
                probe.set_destination("Bazaar gate", ox, oy, yaw, 1.1)
                if probe.route.parts[0]["road_id"] != cid:
                    jumped.append((cid, s_in, side, probe.route.parts[0]["road_id"]))
                probe.set_destination("Village gate", x, y, yaw, 1.2)
    check("satnav: a new destination 0.5-1 m off a junction connector still starts on that connector",
          not jumped, f"{len(jumped)} jumped: {jumped[:5]}")
    nav.set_destination("Colony junction", start.x, start.y, start.yaw, 105.)
    j = nav.route.junctions[0]
    held = nav.update(*nav.route.at(j["start_m"] - 15.), 106., turn_granted=False)
    check("satnav: no turn given: hold_at_m set", held["hold_at_m"] is not None and 10. <= held["hold_at_m"] <= 15.,
          str(held["hold_at_m"]))
    check("satnav: parse_destination", parse_destination("Colony Junction") == "Colony junction"
          and parse_destination("12.5,-40") == (12.5, -40.))
    check("satnav: an unknown place is refused", nav.set_destination("Nowhere", start.x, start.y, start.yaw, 107.) is not None)
    hud = nav.hud()
    check("satnav: the HUD names the destination and the next turn", hud["destination"] == "Colony junction"
          and hud["next"] and hud["route_xy"].shape[1] == 2, str({k: hud[k] for k in ("destination", "next", "remaining_m")}))
    # Passing the parked bus puts the car in the divided road's other lane, same direction (8 Nov smoke run:
    # 22 re-plans in a row). That is still the route.
    start = routes._point(2, 3, 40.)
    nav.set_destination("Bus stop", start.x, start.y, start.yaw, 200.)
    planned = nav.route
    for k in range(60):   # drive the first 30 m on the route itself
        nav.update(*planned.at(.5 * k), 199. + .02 * k)
    before = len(nav.events)
    for k in range(40):
        x, y, yaw = planned.at(30. + .3 * k)
        nav.update(x - 4.5 * math.sin(yaw), y + 4.5 * math.cos(yaw), yaw, 201. + .05 * k)   # in the other lane, passing
    replans = [e for e in nav.events[before:] if e["event"] == "replanned"]
    check("satnav: driving in the divided road's other lane (passing) is not leaving the route", not replans, str(replans[:2]))
    # U-turns: the driver is told where the route turns round, while it still faces the way it came.
    ahead = routes.places["Market street"]
    behind = lane_point(1, ahead.lane, ahead.s + 40. * (1 if ahead.lane < 0 else -1))
    nav.set_destination(f"{behind.x:.2f},{behind.y:.2f}", ahead.x, ahead.y, ahead.yaw, 300.)
    turn = nav.route.uturns[0] if nav.route.uturns else None
    before = nav.update(*nav.route.at(turn["start_m"] - 8.), 301.) if turn else {}
    hud = nav.hud()
    x, y, yaw = nav.route.at(turn["end_m"] + 2.) if turn else (0., 0., 0.)
    after = nav.update(x, y, yaw, 302.) if turn else {}
    told = before.get("uturn") or {}
    check("satnav: a U-turn ahead is in the driver's message (where, the road's edges) and on the HUD; not once turned round",
          turn is not None and abs(told.get("in_m", 99.) - 8.) < .6 and abs(told["pose"][0] - 8.) < 1. and told["left_m"] > 1.
          and told["right_m"] > told["across_m"] and "turn round" in (hud["next"] or "") and after.get("uturn") is None,
          f"in {told.get('in_m')}, pose {told.get('pose')}; HUD {hud['next']!r}; after: {after.get('uturn')}")
    # Turning round goes across the road and back: still the route (no re-plan), even 3-4 m off its line.
    before = len(nav.events)
    sx, sy, syaw = nav.route.at(turn["start_m"])
    for k, (fwd, right, dyaw) in enumerate(((1., 1., .6), (2., 2.5, 1.4), (1.2, 3.8, 1.9), (0.5, 3.5, 2.6), (1., 3., 3.1))):
        nav.update(sx + fwd * math.cos(syaw) - right * math.sin(syaw), sy + fwd * math.sin(syaw) + right * math.cos(syaw),
                   syaw + dyaw, 303. + 1.5 * k)
    replans = [e for e in nav.events[before:] if e["event"] == "replanned"]
    check("satnav: turning round across the road is not leaving the route", not replans, str(replans[:2]))
    nav.update(sx, sy, syaw, 311., uturn_failed=True)
    check("satnav: a U-turn the driver gives up is re-planned without turning round",
          nav.events[-1].get("why") == "uturn_given_up" and not nav.route.uturns, str(nav.events[-1]))


def route_interface_check(routes):
    """A Route serves the town's junction reservations the way the other vehicles' LoopPath does."""
    start, goal = routes.places["Market junction"], routes.places["Colony junction"]
    route = routes.plan(start.x, start.y, start.yaw, goal)
    s = route.length / 3
    point = route.at(s)
    distance, junction = route.next_junction(s)
    after = route.next_junction(route.length)
    ok = (len(point) == 3 and float(np.min(np.hypot(route.points[:, 0] - point[0], route.points[:, 1] - point[1]))) < .5
          and isinstance(distance, float) and {"node", "start_m", "end_m", "road_id"} <= set(junction)
          and "road_id" in route.part_at(s) and isinstance(route.length, float)
          and after[0] >= 1e5 and after[1]["node"] is None)
    check("routes: a Route serves the junction reservations like a LoopPath (at, next_junction, part_at, length)", ok,
          f"next junction {junction.get('node')} in {distance:.0f} m")


def viewer_checks(routes):
    import subprocess
    from india_town_hud import MapPicker
    lines = [(str(road_id), [(float(x), float(y)) for x, y in v[::20, 1:3]])
             for (road_id, lane), v in routes.lanes.items() if road_id in routes.streets and lane > 0]
    places = {name: (p.x, p.y) for name, p in routes.places.items()}
    picker = MapPicker(lines, places)
    worst = max(math.dist(picker.to_world(*picker.to_pixel(x, y)), (x, y)) for x, y in places.values())
    check("viewer: map picker pixel <-> world round trip within 0.6 m", worst <= .6, f"worst {worst:.2f} m")
    name = "Colony junction"
    lx, ly = picker.label_pixel(name)
    hit, miss = picker.hit_place(lx, ly), picker.hit_place(lx, ly - 40)
    check("viewer: clicking a place label picks it; 40 px away does not", hit == name and miss != name, f"{hit!r}, {miss!r}")
    python = str(ROOT / ".venv-carla/Scripts/python.exe")
    help_text = subprocess.run([python, "setup/carla/run_session.py", "--help"], cwd=ROOT, capture_output=True, text=True).stdout
    bad = subprocess.run([python, "setup/carla/run_session.py", "--india-world", "combined", "--destination", "Bazaar gate"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    check("viewer: run_session takes --destination/--destinations, only with --camera-drive",
          "--destination" in help_text and "--destinations" in help_text and bad.returncode != 0,
          f"exit {bad.returncode}: {bad.stderr.strip().splitlines()[-1] if bad.stderr.strip() else ''}")
    # Final review (9 Nov): "x,y" with a negative x (about half the town) was read by argparse as an option at every
    # hop that passed the value as its own argument. Each hop must attach it with "=".
    negative = subprocess.run([python, "setup/carla/run_session.py", "--india-world", "combined", "--destination=-220,160"],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)
    import re
    separate = [f"{name}:{n}" for name in ("setup/run_camera_driving.py", "setup/carla/run_session.py", "Start_Camera_Driving.ps1")
                for n, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1)
                if ("extend(" in line or "+=" in line) and re.search(r"""["']--destinations?["']\s*,""", line)]
    check("viewer: a destination with a negative x reaches the scene (passed as --destination=x,y at every hop)",
          "--camera-drive" in negative.stderr and not separate,
          f"run_session: {negative.stderr.strip().splitlines()[-1] if negative.stderr.strip() else ''}; passed separately at {separate}")


def junction_sharing_check():
    """9 Nov (user, driving it): at Bazaar gate the car waited for its turn because a motorcycle was coming the other
    way round the bend. Two movements conflicted whenever their lane centres came within 3.6 m, and a bend's two
    directions are 2.8-2.95 m apart, side by side, never crossing."""
    import math as _math
    from types import SimpleNamespace
    from india_combined_traffic import CombinedTraffic
    t = town()
    holder = SimpleNamespace(network=t["network"])
    conflicts = CombinedTraffic._junction_conflicts(holder, t["native"])
    bends_shared = {pair: conflicts[pair] for pair in ((101, 103), (100, 102), (105, 107), (104, 106), (132, 133))}
    # Movements whose centre lines come within 2.4 m, side by side or not, must still conflict: H's two directions
    # (1.06 m apart where its bend overlaps), E's close pairs, and movements sharing an exit.
    still = {pair: conflicts[pair] for pair in ((134, 135), (127, 130), (129, 131), (100, 101), (102, 103))}
    check("junctions: a bend's two directions may go at once; movements closer than a vehicle's width still conflict",
          not any(bends_shared.values()) and all(still.values()),
          f"bend pairs conflicting: {[p for p, c in bends_shared.items() if c]}; close pairs sharing: {[p for p, c in still.items() if not c]}")


def plan_test(routes, run_folder):
    """Choose the town test's destination list: from the car's spawn, greedily add places until the
    routes make every turn type available at C, D and E and pass all five districts; then one point
    behind the last stop (a loop). Prints the list for --destinations."""
    import json
    first = json.loads(next(open(Path(run_folder) / "actor_trace.jsonl")))["actors"][0]
    pose = (first["x"], first["y"], first["yaw"])
    wanted = {(c["node"], routes.connector_turn[c["id"]]) for c in routes.connectors.values() if c["node"] in ("C", "D", "E")}
    districts = {s["district"] for s in routes.streets.values()}
    names = list(routes.places)

    def summary(route):
        return (route.length, frozenset((j["node"], j["turn"]) for j in route.junctions if j["node"] in ("C", "D", "E")),
                frozenset(p["district"] for p in route.parts if not p["node"]))
    legs = {("START", b): summary(routes.plan(*pose, routes.places[b])) for b in names}
    for a in names:
        p = routes.places[a]
        for b in names:
            if a != b:
                legs[(a, b)] = summary(routes.plan(p.x, p.y, p.yaw, routes.places[b]))
    # No U-turns: covering every turn type takes ~5.8 km, too far for one 900 s run (~3 km). Find two
    # runs from the same start, each at most `budget` long, that together cover everything.
    budget = 3000.
    goals = sorted(wanted) + sorted(districts)
    bit = {g: 1 << i for i, g in enumerate(goals)}
    full = (1 << len(goals)) - 1
    shortest = {}   # coverage mask -> (length, stops)

    def walk(at, length, mask, stops):
        if stops:
            if mask not in shortest or length < shortest[mask][0]:
                shortest[mask] = (length, list(stops))
        if len(stops) >= 4:
            return
        for b in names:
            if b == at:
                continue
            leg = legs[(at, b)]
            if length + leg[0] > budget:
                continue
            extra = sum(bit[g] for g in leg[1] | leg[2] if g in bit)
            walk(b, length + leg[0], mask | extra, stops + [b])
    walk("START", 0., 0, [])
    best = None
    masks = list(shortest.items())
    for i, (m1, (l1, s1)) in enumerate(masks):
        for m2, (l2, s2) in masks[i:]:
            if m1 | m2 == full and (best is None or max(l1, l2) < best[0]):
                best = (max(l1, l2), (l1, s1), (l2, s2))
    if best is None:
        print("no pair of runs within", budget, "m covers everything")
        return None
    runs = [list(best[1][1]), list(best[2][1])]
    # The second run also ends with a point 60 m behind its last stop: reaching it needs a loop.
    last = routes.places[runs[1][-1]]
    lane = routes.lanes[(last.road_id, last.lane)]
    back = last.s + 60. * (1 if last.lane < 0 else -1)
    row = lane[int(np.argmin(np.abs(lane[:, 0] - back)))]
    loop = routes.plan(last.x, last.y, last.yaw, routes.snap(row[1], row[2])).length
    if best[2][0] + loop <= budget:
        runs[1].append(f"{row[1]:.1f},{row[2]:.1f}")
    else:
        print(f"(the loop to a point behind the last stop adds {loop:.0f} m: left to the offline route checks)")
        loop = 0.
    for k, (stops, length) in enumerate(zip(runs, (best[1][0], best[2][0] + loop))):
        print(f"run {k + 1} stops: {';'.join(stops)}   ({length:.0f} m)")
    return runs


if __name__ == "__main__":
    if "--plan-test" in sys.argv:
        t = town()
        from india_route_planner import TownRoutes
        plan_test(TownRoutes(t["network"], t["sample"]), sys.argv[sys.argv.index("--plan-test") + 1])
        raise SystemExit(0)
    routes = route_checks()
    junction_sharing_check()
    route_interface_check(routes)
    viewer_checks(routes)
    astar_checks()
    sys.path.insert(0, str(ROOT / "setup"))
    driver_checks()
    uturn_checks()
    satnav_checks(routes)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + ("" if not failed else "; failed: " + "; ".join(failed)))
    raise SystemExit(1 if failed else 0)
