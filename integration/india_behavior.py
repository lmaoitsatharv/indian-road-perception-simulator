"""Observable, seeded rule-based traffic on the small LHT junction.

Vehicles use CARLA physics and throttle/brake/steering. Routes and interactions
are explicit scenario rules, not a learned policy or a camera-based controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import xml.etree.ElementTree as ET

import carla
import numpy as np

from india_road import ARMS, ARM_LENGTH, LANE_WIDTH
from india_pedestrian_spacing import entry_blockers

DT = 0.05
CASES = ("clear_overtake", "oncoming", "pedestrian", "return_gap", "junction", "mixed")


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rectangle(x, y, yaw, ex, ey):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + a * c - b * s, y + a * s + b * c)
            for a, b in ((-ex, -ey), (ex, -ey), (ex, ey), (-ex, ey))]


def body(actor, snapshot, bounds=None):
    transform = snapshot.get_transform()
    b = bounds if bounds is not None else actor.bounding_box
    loc = transform.transform(carla.Location(b.location))
    yaw = math.radians(transform.rotation.yaw + b.rotation.yaw)
    return {"x": loc.x, "y": loc.y, "z": loc.z, "yaw": yaw,
            "ex": b.extent.x, "ey": b.extent.y,
            "polygon": rectangle(loc.x, loc.y, yaw, b.extent.x, b.extent.y)}


def polygon_gap(a, b):
    """2D convex-rectangle overlap and exact edge separation in this snapshot."""
    separated = False
    for poly in (a, b):
        for p, q in zip(poly, poly[1:] + poly[:1]):
            nx, ny = p[1]-q[1], q[0]-p[0]
            if math.hypot(nx, ny) < 1e-10:
                continue
            pa = [x*nx + y*ny for x, y in a]
            pb = [x*nx + y*ny for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                separated = True
                break
        if separated:
            break
    if not separated:
        return True, 0.0
    closest_sq = float("inf")
    for points, poly in ((a, b), (b, a)):
        for x, y in points:
            for (u, v), (w, z) in zip(poly, poly[1:] + poly[:1]):
                dx, dy = w-u, z-v
                t = max(0., min(1., ((x-u)*dx + (y-v)*dy) / max(dx*dx+dy*dy, 1e-12)))
                closest_sq = min(closest_sq, (x-u-t*dx)**2 + (y-v-t*dy)**2)
    return False, math.sqrt(closest_sq)


@dataclass
class Route:
    incoming: str
    outgoing: str
    points: np.ndarray
    distances: np.ndarray
    junction_end: float

    @classmethod
    def build(cls, road_map, xml, incoming, outgoing):
        roads = ET.fromstring(xml).findall("road")
        connector = next(r for r in roads if r.attrib["name"] == f"{incoming}_to_{outgoing}")
        cid, length = int(connector.attrib["id"]), float(connector.attrib["length"])
        xyz = []
        parts = ((ARMS[incoming][0], 1, 0.1, ARM_LENGTH - .01),
                 (cid, 1, .01, length - .01),
                 (ARMS[outgoing][0], -1, ARM_LENGTH - .01, .1))
        junction_end = None
        for index, (rid, lane, a, b) in enumerate(parts):
            for s in np.linspace(a, b, max(2, int(abs(b - a) / .4) + 1)):
                wp = road_map.get_waypoint_xodr(rid, lane, float(s))
                if wp is None:
                    raise RuntimeError(f"Missing route waypoint {rid}/{lane}/{s}")
                t = wp.transform
                p = [t.location.x, t.location.y, math.radians(t.rotation.yaw)]
                if not xyz or math.hypot(p[0] - xyz[-1][0], p[1] - xyz[-1][1]) > .02:
                    xyz.append(p)
            if index == 1:
                junction_end = len(xyz) - 1
        points = np.array(xyz)
        distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))))
        return cls(incoming, outgoing, points, distances, float(distances[junction_end]))

    def at(self, s, offset=0.0):
        i = min(int(np.searchsorted(self.distances, s)), len(self.points) - 1)
        if i == 0:
            p = self.points[0].copy()
        else:
            fraction = np.clip((s - self.distances[i - 1]) / (self.distances[i] - self.distances[i - 1]), 0, 1)
            p = self.points[i - 1].copy()
            p[:2] += fraction * (self.points[i, :2] - p[:2])
            p[2] += fraction * wrap(self.points[i, 2] - p[2])
        p[:2] += offset * np.array([-math.sin(p[2]), math.cos(p[2])])
        return p

    def project(self, x, y, last=0):
        low, high = max(0, last - 5), min(len(self.points), last + 100)
        i = low + int(np.argmin(np.sum((self.points[low:high, :2] - [x, y]) ** 2, axis=1)))
        return i, float(self.distances[i])


@dataclass
class Driver:
    actor: object
    route: Route
    name: str
    desired: float
    offset: float = 0.0
    parked: bool = False
    index: int = 0
    s: float = 0
    target: float = 0
    reason: str = "accelerate"
    accel_command: float = 0
    speed_integral: float = 0
    previous_speed: float = 0
    previous_target: float = 0
    smoothed_accel: float = 0
    steer: float = 0
    requested: float | None = None
    stop_until: float | None = None
    bus_stopped: bool = False
    pass_state: str = "follow"
    pass_leader: int | None = None
    pass_offset: float = 1.08
    blocked_reason: str | None = None
    previous_reason: str | None = None
    finished: bool = False
    half_length: float = 2.0
    half_width: float = 1.0
    wheelbase: float = 2.7
    max_steer: float = .7
    goal_s: float = 0
    wait_started: float | None = None
    wait_reported: bool = False
    stuck_started: float | None = None

    def calibrate(self):
        self.half_length, self.half_width = self.actor.bounding_box.extent.x, self.actor.bounding_box.extent.y
        physics = self.actor.get_physics_control()
        xs = [wheel.position.x for wheel in physics.wheels]
        self.wheelbase = max(.8, (max(xs) - min(xs)) / 100)
        angles = [wheel.max_steer_angle for wheel in physics.wheels if wheel.max_steer_angle > 1]
        self.max_steer = math.radians(max(angles, default=40))

    def control(self, state):
        if self.parked:
            return carla.VehicleControl(brake=1, hand_brake=True)
        x, y, yaw, speed = state["x"], state["y"], state["yaw"], state["speed"]
        lookahead = 3.8 + .5 * speed
        target = self.route.at(self.s + lookahead, self.offset)
        alpha = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        steer = math.atan2(2 * self.wheelbase * math.sin(alpha), lookahead) / self.max_steer
        self.steer += float(np.clip(steer - self.steer, -.07, .07))
        self.steer = float(np.clip(self.steer, -.7, .7))
        error = self.target - speed
        self.smoothed_accel = .8 * self.smoothed_accel + .2 * (speed - self.previous_speed) / DT
        if self.target < self.previous_target - .3 or error < -.5:
            self.speed_integral = min(0., self.speed_integral)
        self.speed_integral = float(np.clip(self.speed_integral + error * DT, -3, 3))
        requested = float(np.clip(.85 * error + .35 * self.speed_integral - .4 * self.smoothed_accel, -3.5, 2.1))
        self.previous_speed, self.previous_target = speed, self.target
        emergency = self.reason.startswith("emergency")
        self.accel_command += float(np.clip(requested - self.accel_command, -.15 if not emergency else -8, .12))
        if self.target < .08 and speed < .2:
            throttle, brake = 0., .7
        elif emergency:
            throttle, brake = 0., 1.
        elif self.accel_command >= 0:
            throttle = float(np.clip(.16 + .32 * self.accel_command, 0, .85))
            brake = 0.
        else:
            throttle = 0.
            brake = float(np.clip(-self.accel_command * .06, 0, .25))
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=self.steer)


@dataclass
class Pedestrian:
    actor: object
    name: str
    a: tuple
    b: tuple
    speed: float
    crossing: bool = False
    start: float = 0
    active: bool = False
    completed: bool = False
    reverse: bool = False
    gap_seconds: float = 4.5
    gap_wait_started: float | None = None
    people_wait_started: float | None = None

    def control(self, sim_time):
        location = self.actor.get_location()
        if self.crossing and (sim_time < self.start or self.completed):
            return carla.WalkerControl(speed=0)
        self.active = True
        target = self.a if self.reverse else self.b
        dx, dy = target[0] - location.x, target[1] - location.y
        norm = math.hypot(dx, dy)
        if norm < .18:
            if self.crossing:
                self.completed, self.active = True, False
                return carla.WalkerControl(speed=0)
            self.reverse = not self.reverse
            target = self.a if self.reverse else self.b
            dx, dy = target[0] - location.x, target[1] - location.y
            norm = math.hypot(dx, dy)
        return carla.WalkerControl(direction=carla.Vector3D(dx / norm, dy / norm, 0), speed=self.speed)


class Traffic:
    def __init__(self, world, xml, owned, case, seed, emit, dataset_variation=False):
        self.world, self.owned, self.case, self.emit = world, owned, case, emit
        self.rng = random.Random(seed)
        self.dataset_variation = dataset_variation
        self.variation_rng = random.Random(seed + 90000)
        names = {road.attrib["name"] for road in ET.fromstring(xml).findall("road")}
        self.routes = {(a, b): Route.build(world.get_map(), xml, a, b)
                       for a in ARMS for b in ARMS if a != b and f"{a}_to_{b}" in names}
        self.drivers, self.walkers = [], []
        self.holders = set()
        self.conflicts = {}
        self.overtakes, self.blocked = 0, set()
        self._spawn()
        self.bounds = {a.id: a.bounding_box for a in [d.actor for d in self.drivers] + [p.actor for p in self.walkers]}

    def vehicle(self, blueprint, name, incoming, outgoing, start, desired, offset=0., parked=False, color=None):
        if (incoming, outgoing) not in self.routes:
            return None
        if self.dataset_variation:
            start += self.variation_rng.uniform(-3.0, 3.0)
            if not parked:
                desired *= self.variation_rng.uniform(.90, 1.10)
            color = self.variation_rng.choice(["210,220,230", "190,50,35", "40,95,160", "230,220,185", "45,50,55"])
        route = self.routes[incoming, outgoing]
        bp = self.world.get_blueprint_library().find(blueprint)
        bp.set_attribute("role_name", "india_" + name)
        if color and bp.has_attribute("color"):
            bp.set_attribute("color", color)
        p = route.at(start, offset)
        actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=p[0], y=p[1], z=.45),
                                                              carla.Rotation(yaw=math.degrees(p[2]))))
        if actor is None:
            raise RuntimeError(f"Unable to spawn required actor {name}; episode refused")
        self.owned.append(actor)
        d = Driver(actor, route, name, desired, offset, parked)
        d.index = int(np.searchsorted(route.distances, start))
        d.s = start
        d.goal_s = float(route.distances[-1] - (40 if name == "ego" else 10))
        d.calibrate()
        actor.apply_control(carla.VehicleControl(brake=1))
        if name == "ego":
            actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        self.drivers.append(d)
        return d

    def pedestrian(self, name, a, b, speed=1.2, crossing=False, start=0):
        if self.dataset_variation:
            speed *= self.variation_rng.uniform(.9, 1.1)
            if crossing:
                start = max(0, start + self.variation_rng.uniform(-2, 3))
        bps = sorted(self.world.get_blueprint_library().filter("walker.pedestrian.*"), key=lambda x: x.id)
        bp = bps[self.rng.randrange(len(bps))]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=a[0], y=a[1], z=.5)))
        if actor is None:
            raise RuntimeError(f"Unable to spawn required pedestrian {name}")
        self.owned.append(actor)
        p = Pedestrian(actor, name, a, b, speed, crossing, start)
        self.walkers.append(p)
        return p

    def _spawn(self):
        self.ego = self.vehicle("vehicle.tesla.model3", "ego", "west", "east", 27, 6.8, color="210,220,230")
        if self.case not in ("junction", "pedestrian"):
            self.vehicle("vehicle.bh.crossbike", "slow_cycle", "west", "east", 54, 2.4, -1.28)
        if self.case in ("oncoming", "mixed"):
            self.vehicle("vehicle.nissan.micra", "oncoming_car", "east", "west", 112, 7.0, color="220,70,50")
        if self.case == "return_gap":
            self.vehicle("vehicle.vespa.zx125", "return_guard", "west", "east", 67, 3.0, -1.25)
        if self.case in ("pedestrian", "mixed"):
            self.pedestrian("crossing_1", (-78, -6.55), (-78, 6.75), 1.2, True, 5.0)
        if self.case in ("junction", "mixed"):
            self.vehicle("vehicle.citroen.c3", "turning_car", "north", "west", 100, 5.1, color="40,140,210")
            self.vehicle("vehicle.vespa.zx125", "south_scooter", "south", "north", 78, 5.3, -.5)
            self.vehicle("vehicle.mercedes.sprinter", "delivery_van", "east", "south", 67, 5.8, color="220,220,210")
        if self.case == "mixed":
            self.vehicle("vehicle.mitsubishi.fusorosa", "bus", "west", "east", 89, 4.8, 0)
            self.vehicle("vehicle.yamaha.yzf", "motorcycle", "east", "west", 20, 6.2, -.55)
            self.vehicle("vehicle.audi.a2", "parked_car", "west", "east", 224, 0, -3.35, True, "180,80,35")
            self.vehicle("vehicle.nissan.micra", "north_car", "north", "south", 35, 6.0 + self.rng.uniform(-.5, .5))
            self.vehicle("vehicle.gazelle.omafiets", "south_cycle", "south", "east", 27, 2.6, -.9)
            self.pedestrian("crossing_2", (47, 6.55), (47, -8.0), 1.15, True, 27 + self.rng.uniform(-2, 2))
            # Distinct pavement tracks avoid walking people occupying the same body space.
            for i, (x, side) in enumerate(((-135, -1), (-108, 1), (-52, -1), (30, 1), (64, -1), (113, 1))):
                direction = 1 if i % 2 == 0 else -1
                end = min(-20, x + 35) if x < 0 else min(136, x + 22)
                track = 7.8 if x > 12 and side < 0 else 5.2 + .25 * (i % 2)
                a, b = (x, side * track), (end, side * track)
                self.pedestrian(f"footpath_{i}", a, b, .95 + self.rng.uniform(0, .35))

    def metadata(self):
        return {"vehicles": [{"id": d.actor.id, "name": d.name, "blueprint": d.actor.type_id,
                              "route": [d.route.incoming, d.route.outgoing], "desired_mps": d.desired,
                              "half_length_m": d.half_length, "half_width_m": d.half_width,
                              "wheelbase_m": d.wheelbase, "max_steer_rad": d.max_steer,
                              "goal_progress_m": d.goal_s, "goal_xy": d.route.at(d.goal_s)[:2].tolist(),
                              "initial_lateral_offset_m": d.offset, "parked": d.parked} for d in self.drivers],
                "pedestrians": [{"id": p.actor.id, "name": p.name, "a": p.a, "b": p.b,
                                 "speed_mps": p.speed, "crossing": p.crossing, "start_s": p.start,
                                 "gap_acceptance_s": p.gap_seconds if self.case == "mixed" and p.crossing else None}
                                for p in self.walkers]}

    def _overtake(self, states, t):
        d, ego = self.ego, states[self.ego.actor.id]
        # Observation only: none of these fields are read by the driving policy.
        self.pass_explanation = {"state": d.pass_state, "reason": d.pass_state,
                                 "leader": None, "blockers": [],
                                 "junction_entry_remaining_m": max(0., ARM_LENGTH - d.s),
                                 "permitted_section_remaining_m": max(0., d.route.junction_end + 25 - d.s)}
        c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
        def longitudinal(other):
            return (other["x"] - ego["x"]) * c + (other["y"] - ego["y"]) * s
        if d.pass_state == "follow":
            leaders = [v for v in self.drivers if v.name in ("slow_cycle", "return_guard", "parked_car")
                       and 4 < longitudinal(states[v.actor.id]) < 30
                       and abs(v.offset - d.offset) < d.half_width + v.half_width + .25]
            if not leaders or not (d.s < ARM_LENGTH - 35 or d.s > d.route.junction_end + 25):
                self.pass_explanation.update(reason="no_eligible_leader" if not leaders else "junction_zone",
                                             leader=min(leaders, key=lambda v: longitudinal(states[v.actor.id])).name if leaders else None)
                return
            lead = min(leaders, key=lambda v: longitudinal(states[v.actor.id]))
            reason = None
            # Required oncoming gap derives from conservative pass duration, not a coin flip.
            pass_time = (longitudinal(states[lead.actor.id]) + 2 * (d.half_length + lead.half_length) + 8) / max(d.desired - lead.desired, 1) + 3
            self.pass_explanation.update(leader=lead.name, leader_center_distance_m=longitudinal(states[lead.actor.id]),
                                         pass_time_budget_s=pass_time, required_junction_approach_m=d.desired * pass_time + 12)
            for other in self.drivers:
                v = states[other.actor.id]
                approach_speed = max(v["speed"], other.desired)
                if math.cos(v["yaw"] - ego["yaw"]) < -.6 and 0 < longitudinal(v) < (d.desired + approach_speed) * pass_time + 12:
                    reason = "oncoming_gap"
                    self.pass_explanation["blockers"].append({"reason": reason, "actor": other.name,
                        "available_m": longitudinal(v), "required_m": (d.desired + approach_speed) * pass_time + 12})
            for p in self.walkers:
                if p.crossing and not p.completed and -2 < longitudinal(states[p.actor.id]) < 55 and p.start < t + pass_time:
                    reason = "pedestrian_conflict"
                    self.pass_explanation["blockers"].append({"reason": reason, "actor": p.name,
                        "ahead_m": longitudinal(states[p.actor.id]), "scheduled_start_s": p.start})
            passing_offset = max(1.08, lead.offset + d.half_width + lead.half_width + .55)
            separation = passing_offset - lead.offset
            if separation < d.half_width + lead.half_width + .45:
                reason = "body_clearance"
                self.pass_explanation["blockers"].append({"reason": reason, "metric": "side_separation",
                    "available_m": separation, "required_m": d.half_width + lead.half_width + .45})
            if passing_offset > 1.65:
                reason = "body_clearance"
                self.pass_explanation["blockers"].append({"reason": reason, "metric": "lateral_offset",
                    "available_m": 1.65, "required_m": passing_offset})
            if d.s < ARM_LENGTH and ARM_LENGTH - d.s < d.desired * pass_time + 12:
                reason = "junction_too_close"
                self.pass_explanation["blockers"].append({"reason": reason,
                    "available_m": ARM_LENGTH - d.s, "required_m": d.desired * pass_time + 12})
            self.pass_explanation.update(reason=reason or "pass_allowed")
            if reason:
                self.blocked.add(reason)
                if d.blocked_reason != reason:
                    self.emit(t, "overtake_blocked", d.name, reason=reason, leader=lead.name)
                d.blocked_reason = reason
            else:
                d.pass_state, d.pass_leader, d.blocked_reason = "passing", lead.actor.id, None
                d.pass_offset = passing_offset
                self.emit(t, "overtake_started", d.name, leader=lead.name, pass_time_budget_s=pass_time,
                          opposing_lane_borrow_m=max(0, d.half_width + passing_offset - LANE_WIDTH / 2))
        if d.pass_state == "passing":
            self.pass_explanation.update(reason="passing" if self.pass_explanation["reason"] in ("passing", "pass_allowed") else self.pass_explanation["reason"])
            d.offset += min(.45 * DT, d.pass_offset - d.offset)
            lead = next(v for v in self.drivers if v.actor.id == d.pass_leader)
            self.pass_explanation.update(leader=lead.name,
                leader_center_distance_m=longitudinal(states[lead.actor.id]),
                required_leader_behind_m=d.half_length + lead.half_length + 3.5)
            if longitudinal(states[lead.actor.id]) < -(d.half_length + lead.half_length + 3.5):
                guards = [v for v in self.drivers if v.actor.id != d.actor.id
                          and math.cos(states[v.actor.id]["yaw"] - ego["yaw"]) > .7
                          and -(d.half_length + v.half_length + 4) < longitudinal(states[v.actor.id]) < 15
                          and abs(v.offset) < 1.6]
                if guards:
                    self.pass_explanation.update(reason="return_gap", blocking_actors=[g.name for g in guards])
                    if d.blocked_reason != "return_gap":
                        self.emit(t, "return_blocked", d.name, reason="return_gap", actors=[g.name for g in guards])
                    d.blocked_reason = "return_gap"
                    self.blocked.add("return_gap")
                else:
                    self.pass_explanation.update(reason="returning")
                    d.pass_state, d.blocked_reason = "returning", None
                    self.emit(t, "overtake_return", d.name)
        elif d.pass_state == "returning":
            d.offset = max(0, d.offset - .4 * DT)
            if d.offset < .005 and abs(ego["y"] + 2.25) < .3:
                self.pass_explanation.update(reason="overtake_completed")
                d.pass_state, d.pass_leader = "follow", None
                self.overtakes += 1
                self.emit(t, "overtake_completed", d.name, count=self.overtakes)
        self.pass_explanation.update(state=d.pass_state)

    def _wait_for_footpath_gap(self, person, states, t):
        blockers = entry_blockers(person, self.walkers, states)
        if blockers:
            if person.people_wait_started is None:
                person.people_wait_started = t
                self.emit(t, "pedestrian_waiting_for_person", person.name, blockers=blockers)
            person.actor.apply_control(carla.WalkerControl(speed=0))
            return True
        if person.people_wait_started is not None:
            self.emit(t, "pedestrian_person_gap_accepted", person.name, waited_s=t-person.people_wait_started)
            person.people_wait_started = None
        return False

    def step(self, snapshot, t):
        states = {}
        for actor in [d.actor for d in self.drivers] + [p.actor for p in self.walkers]:
            snap = snapshot.find(actor.id)
            b = body(actor, snap, self.bounds[actor.id])
            velocity = snap.get_velocity()
            b["speed"] = math.hypot(velocity.x, velocity.y)
            states[actor.id] = b
        for d in self.drivers:
            state = states[d.actor.id]
            d.index, d.s = d.route.project(state["x"], state["y"], d.index)
        self._overtake(states, t)
        for held in list(self.holders):
            h = next(d for d in self.drivers if d.actor.id == held)
            if h.s > h.route.junction_end + h.half_length + 3:
                self.emit(t, "junction_released", h.name)
                self.holders.remove(held)
        candidates = []
        for d in self.drivers:
            if not d.parked and ARM_LENGTH - 26 < d.s < d.route.junction_end + d.half_length + 3:
                if d.requested is None:
                    d.requested = t
                candidates.append(d)
        for chosen in sorted(candidates, key=lambda d: (d.requested, -d.s, d.actor.id)):
            if chosen.actor.id in self.holders:
                continue
            # Do not give a rear vehicle priority over an ungranted vehicle ahead in its queue.
            queued_ahead = any(d.route.incoming == chosen.route.incoming and d.s > chosen.s + 1
                               and d.actor.id not in self.holders for d in candidates if d is not chosen)
            if queued_ahead:
                continue
            compatible = True
            for held in self.holders:
                h = next(d for d in self.drivers if d.actor.id == held)
                if h.route is chosen.route:
                    if h.s - chosen.s < h.half_length + chosen.half_length + 6:
                        compatible = False
                    continue
                key = tuple(sorted((chosen.actor.id, held)))
                if key not in self.conflicts:
                    a = chosen.route.points[(chosen.route.distances >= ARM_LENGTH - 3) &
                                            (chosen.route.distances <= chosen.route.junction_end + 3), :2]
                    b = h.route.points[(h.route.distances >= ARM_LENGTH - 3) &
                                       (h.route.distances <= h.route.junction_end + 3), :2]
                    closest = float(np.sqrt(np.min(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))))
                    self.conflicts[key] = closest < chosen.half_width + h.half_width + .8
                if self.conflicts[key]:
                    compatible = False
            if compatible:
                self.holders.add(chosen.actor.id)
                self.emit(t, "junction_granted", chosen.name, waited_s=t - chosen.requested,
                          concurrent_reservations=len(self.holders))
        for d in self.drivers:
            state = states[d.actor.id]
            desired, reason = d.desired, "cruise"
            if d.parked:
                desired, reason = 0, "parked"
            elif d.s > d.goal_s - 32:
                remaining = d.goal_s - d.s
                desired = min(desired, math.sqrt(2 * 1.6 * max(0, remaining)))
                if remaining <= .6:
                    desired, reason, d.finished = 0, "route_goal", True
            elif ARM_LENGTH - 30 < d.s < d.route.junction_end + 4:
                desired, reason = min(desired, 3.8), "junction_approach"
                if d.actor.id not in self.holders:
                    gap = ARM_LENGTH - 7 - d.s
                    desired, reason = min(desired, max(0, (gap - .5) / 1.5)), "junction_yield"
            if d.name == "bus" and not d.bus_stopped and d.s >= 88:
                d.stop_until, d.bus_stopped = t + 6, True
                self.emit(t, "bus_stop", d.name, stop_duration_s=6)
            if d.stop_until and t < d.stop_until:
                desired, reason = 0, "bus_stop"
            c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
            for other in self.drivers:
                if other.actor.id == d.actor.id:
                    continue
                v = states[other.actor.id]
                dx, dy = v["x"] - state["x"], v["y"] - state["y"]
                ahead, lateral = dx * c + dy * s, -dx * s + dy * c
                # Use orientation-projected body extent for crossing/turning vehicles.
                da = v["yaw"] - state["yaw"]
                width = abs(math.cos(da)) * v["ey"] + abs(math.sin(da)) * v["ex"]
                length = abs(math.cos(da)) * v["ex"] + abs(math.sin(da)) * v["ey"]
                if 0 < ahead < 55 and abs(lateral) < d.half_width + width + .25:
                    gap = ahead - d.half_length - length
                    lead_speed = max(0, v["speed"] * math.cos(da))
                    cap = max(0, lead_speed + (gap - 4 - 1.5 * state["speed"]) / 2.)
                    if cap < desired:
                        desired, reason = cap, "headway"
                    if gap < 1.3 and state["speed"] > lead_speed + .3:
                        desired, reason = 0, "emergency_vehicle"
            for p in self.walkers:
                if not p.crossing or not p.active or p.completed:
                    continue
                v = states[p.actor.id]
                dx, dy = v["x"] - state["x"], v["y"] - state["y"]
                ahead, lateral = dx * c + dy * s, -dx * s + dy * c
                # A moving person within the road corridor gets priority before entering the vehicle body path.
                if -1 < ahead < 45 and abs(lateral) < 5.8:
                    gap = ahead - d.half_length - .5
                    cap = max(0, (gap - 3) / 2.5)
                    if cap < desired:
                        desired, reason = cap, "pedestrian_yield"
                    if gap < 2 and abs(lateral) < d.half_width + .8 and state["speed"] > .2:
                        desired, reason = 0, "emergency_pedestrian"
            if d.previous_reason != reason and reason in ("pedestrian_yield", "emergency_pedestrian", "emergency_vehicle", "junction_yield", "route_goal"):
                self.emit(t, reason, d.name)
            waiting = state["speed"] < .15 and not d.parked and not d.finished
            if waiting:
                if d.wait_started is None:
                    d.wait_started = t
                if t - d.wait_started > 12 and not d.wait_reported:
                    self.emit(t, "extended_wait", d.name, reason=reason, seconds=t - d.wait_started)
                    d.wait_reported = True
            elif d.wait_started is not None:
                if d.wait_reported:
                    self.emit(t, "wait_ended", d.name, seconds=t - d.wait_started)
                d.wait_started, d.wait_reported = None, False
            if waiting and desired > 1 and reason == "cruise":
                if d.stuck_started is None:
                    d.stuck_started = t
                if t - d.stuck_started > 8:
                    self.emit(t, "vehicle_stuck", d.name, desired_mps=desired)
                    d.stuck_started = t
            else:
                d.stuck_started = None
            d.target, d.reason, d.previous_reason = desired, reason, reason
            d.actor.apply_control(d.control(state))
        for p in self.walkers:
            was_active, was_completed = p.active, p.completed
            if (self.case == "mixed" or self.dataset_variation) and p.crossing and not p.active and not p.completed and t >= p.start:
                # Waiting people choose a gap; once in the road they keep walking and drivers yield.
                if self._wait_for_footpath_gap(p, states, t):
                    continue
                blocked = False
                for d in self.drivers:
                    v = states[d.actor.id]
                    if abs(math.cos(v["yaw"])) < .8:
                        continue
                    ahead = (p.a[0] - v["x"]) * math.cos(v["yaw"])
                    ttc = (ahead - d.half_length - .5) / max(v["speed"], .1)
                    if abs(ahead) < d.half_length + .8 or 0 < ttc < p.gap_seconds:
                        blocked = True
                if blocked:
                    if p.gap_wait_started is None:
                        p.gap_wait_started = t
                        self.emit(t, "pedestrian_waiting_for_gap", p.name)
                    p.actor.apply_control(carla.WalkerControl(speed=0))
                    continue
                if p.gap_wait_started is not None:
                    self.emit(t, "pedestrian_gap_accepted", p.name, waited_s=t - p.gap_wait_started)
            p.actor.apply_control(p.control(t))
            if p.crossing and p.active and not was_active:
                self.emit(t, "pedestrian_started_crossing", p.name)
            if p.completed and not was_completed:
                self.emit(t, "pedestrian_finished_crossing", p.name)
        return states
