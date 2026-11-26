"""Sat-nav for the camera car: destination, route, and one message per control step.

Runs in the viewer process with the car's exact pose (the design's choice: no
GPS noise) and the town's route planner. Each step it tells the camera driver
whether it is on a street ("lane") or at a junction ("junction"), the route
line for the next 30 m in the car's own frame, where to wait for its turn,
where the route turns round (a U-turn) and how far the destination is. It re-plans after a missed turn or a reset. No
CARLA import. See docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md.
"""
from __future__ import annotations

import math
import re

import numpy as np

from india_combined_map import NODE_NAMES
from india_route_planner import DISTRICT_PLACES, LanePoint, Route, TownRoutes
from india_rl_speed import cruise_limit, legal_profile, limits_at, posted_limits

PLACE_NAMES = tuple(NODE_NAMES.values()) + tuple(name for name, *_ in DISTRICT_PLACES)
_XY = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")
TURN_WORDS = {"left": "left", "right": "right", "straight": "straight on"}


def parse_destination(text, names=PLACE_NAMES):
    """A place name (any case) -> its proper name; "x,y" -> (x, y). ValueError otherwise."""
    match = _XY.match(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    wanted = text.strip().lower()
    for name in names:
        if name.lower() == wanted:
            return name
    raise ValueError(f"Unknown place {text!r}; places: {', '.join(names)}")


class SatNav:
    def __init__(self, routes: TownRoutes, junction_lead_m=20., guide_m=30., off_route_m=4., arrive_report_m=60.,
                 front_m=1.9, jump_m=10., hold_within_m=35., arrived_m=2., driver_arrived_within_m=25.):
        self.routes = routes
        self.junction_lead_m, self.guide_m, self.off_route_m = junction_lead_m, guide_m, off_route_m
        self.arrive_report_m, self.front_m, self.jump_m = arrive_report_m, front_m, jump_m
        self.hold_within_m, self.arrived_m = hold_within_m, arrived_m
        self.driver_arrived_within_m = driver_arrived_within_m   # the driver's "arrived short" window
        self.destination: LanePoint | None = None
        self.destination_name = ""
        self.route: Route | None = None
        self.arrived = False
        self.events: list[dict] = []
        self._last = None          # last pose seen (x, y)
        self._index = 0
        self._inside = None        # the junction (node, start_m) the car is in
        self._progress = None
        self._off_since = None     # when the car first looked off its route
        self._no_uturns = False    # the driver gave a U-turn up: routes to this destination go round instead

    # ---- destination ------------------------------------------------------------------
    def _event(self, sim_time, event, **details):
        self.events.append({"sim_s": round(float(sim_time), 2), "event": event, **details})

    def set_destination(self, target, x, y, yaw, sim_time):
        """A place name or a world point (x, y). Returns an error text, or None when set."""
        if isinstance(target, str):
            try:
                name = parse_destination(target, tuple(self.routes.places))
            except ValueError as error:
                return str(error)
            if isinstance(name, tuple):
                target = name
            else:
                point = self.routes.places[name]
        if not isinstance(target, str):
            point = self.routes.snap(float(target[0]), float(target[1]))
            if point is None:
                return "Not on a street"
            name = f"point {point.x:.0f}, {point.y:.0f}"
        had_route = self.route is not None
        start = self._connector_driven(x, y)
        self.destination, self.destination_name, self.arrived = point, name, False
        self._no_uturns = False
        self._event(sim_time, "destination_set", name=name, x=round(point.x, 2), y=round(point.y, 2))
        return self._plan(x, y, yaw, sim_time, "new_destination" if had_route else None, start)

    def _connector_driven(self, x, y):
        """The junction connector of the current route the car is on (within the off-route limit), or None.
        Connectors from one entry overlap for their first metres, so locating the car by its pose alone put
        it on a sibling connector to another exit when it was 0.5-1 m off the centre line (final review)."""
        if self.route is None:
            return None
        progress = self.route.progress(x, y, 0., self._index)
        part = self.route.part_at(progress["s"])
        if progress["offset_m"] > self.off_route_m or part["road_id"] not in self.routes.connectors:
            return None
        return self.routes.point_on(part["road_id"], part["lane"], x, y)

    def clear(self, sim_time):
        if self.destination is not None:
            self._event(sim_time, "cleared", name=self.destination_name)
        self.destination, self.destination_name, self.route = None, "", None
        self.arrived, self._inside, self._progress = False, None, None

    def _plan(self, x, y, yaw, sim_time, why, start=None):
        try:
            self.route = self.routes.plan(x, y, yaw, self.destination, start, u_turns=not self._no_uturns)
        except ValueError as error:
            self.route = None
            self._event(sim_time, "no_route", why=why, error=str(error))
            return str(error)
        self._index, self._inside, self._last = 0, None, (x, y)
        # Posted limits along the route, every metre (the learned speed keeps to them; docs/RL_SPEED_DESIGN_20261112.md)
        self._limits = posted_limits(self.route.parts, self.route.length)
        self._legal = legal_profile(self._limits)
        self._cruise = cruise_limit(self._limits)                 # what the learned car aims for (gear band)
        self._cruise_cap = legal_profile(self._cruise)
        legs = [(p["road_id"], p["lane"], p["node"], "u-turn" if p.get("uturn") else p.get("turn")) for p in self.route.parts]
        details = {"length_m": round(self.route.length, 1), "legs": legs}
        if why:
            self._event(sim_time, "replanned", why=why, **details)
        else:
            self._event(sim_time, "route_planned", **details)
        return None

    # ---- each control step ----------------------------------------------------------------
    def update(self, x, y, yaw, sim_time, turn_granted=None, driver_arrived=False, uturn_failed=False):
        """The driver's message for this step, or None without a destination. `uturn_failed`: the driver gave
        up turning round (no room, or no gap for two minutes): re-plan with no U-turns to this destination."""
        if self.destination is None:
            return None
        if uturn_failed and self.route is not None and self.route.uturns:
            self._no_uturns = True
            self._plan(x, y, yaw, sim_time, "uturn_given_up")
        if self.route is None:
            return {"mode": "lane", "guide": None, "junction": None, "hold_at_m": None, "arrive_in_m": None, "no_route": True}
        if self._last is not None and math.hypot(x - self._last[0], y - self._last[1]) > self.jump_m:
            self._plan(x, y, yaw, sim_time, "reset")
        self._last = (x, y)
        progress = self.route.progress(x, y, yaw, self._index)
        off = progress["off_route"] and not self.arrived
        if off and progress["offset_m"] <= 8. and any(u["start_m"] - 10. <= progress["s"] <= u["end_m"] + 15. for u in self.route.uturns):
            off = False   # turning round: across the road, forwards and backwards, for a while
        if off and progress["offset_m"] <= 9.:
            # Still on the route's street, same direction (the divided road's other lane while passing a
            # parked bus): not off the route (8 Nov smoke run re-planned 22 times in a row there).
            here, part = self.routes.locate(x, y, yaw), self.route.part_at(progress["s"])
            off = not (here.road_id == part["road_id"] and np.sign(here.lane) == np.sign(part["lane"]))
        self._off_since = (self._off_since or sim_time) if off else None
        if off and sim_time - self._off_since >= 1.:
            self._plan(x, y, yaw, sim_time, "left_route")
            self._off_since = None
            progress = self.route.progress(x, y, yaw, self._index)
        self._index, s = progress["index"], progress["s"]
        remaining = progress["remaining_m"]
        current = next((j for j in self.route.junctions if j["start_m"] - self.junction_lead_m <= s <= j["end_m"]), None)
        key = None if current is None else (current["node"], round(current["start_m"], 1))
        if key != self._inside:
            if self._inside is not None:
                self._event(sim_time, "junction_left", node=self._inside[0])
            if key is not None:
                self._event(sim_time, "junction_entered", node=current["node"], name=current.get("name"), turn=current.get("turn"))
            self._inside = key
        ahead = np.array([self.route.at(s + k)[:2] for k in np.arange(0., self.guide_m + 1e-9, 1.)])
        keep = np.r_[True, np.hypot(*np.diff(ahead, axis=0).T) > 1e-6]
        ahead = ahead[keep]
        dx, dy = ahead[:, 0] - x, ahead[:, 1] - y
        c, sn = math.cos(yaw), math.sin(yaw)
        guide = np.stack([dx * c + dy * sn, -dx * sn + dy * c], axis=1)
        hold = None
        uturn = None
        to_turn, part = self.route.next_uturn(s)
        if part is not None and -3. <= to_turn <= 60.:
            px, py, pyaw = self.route.at(part["start_m"])
            # Only while the car still faces the way it came (once round, the turn is behind it).
            if abs(math.remainder(yaw - pyaw, 2 * math.pi)) < math.radians(45.):
                ddx, ddy = px - x, py - y
                uturn = {"in_m": float(to_turn), "pose": (ddx * c + ddy * sn, -ddx * sn + ddy * c, math.remainder(pyaw - yaw, 2 * math.pi)),
                         "across_m": part["across_m"], "left_m": part["left_m"], "right_m": part["right_m"]}
        distance, nxt = self.route.next_junction(s)
        if turn_granted is False and nxt["node"] is not None and 0. <= distance <= self.hold_within_m:
            hold = nxt["start_m"] - s - self.front_m
        # The driver's own "arrived" (stopped short behind a vehicle) counts only near the pin: on the
        # step after a new destination is set it can still carry the last one's (8 Nov town run).
        if not self.arrived and (remaining <= self.arrived_m or (driver_arrived and remaining <= self.driver_arrived_within_m)):
            self.arrived = True
            self._event(sim_time, "arrived", name=self.destination_name, short_m=round(remaining, 1))
        self._progress = {"s": s, "remaining_m": remaining, "next": (distance, nxt)}
        posted, legal, next_limit, next_limit_m = limits_at(self._limits, self._legal, s + self.front_m)
        here = int(np.clip(np.floor(s + self.front_m), 0, len(self._cruise) - 1))
        return {"mode": "junction" if current is not None else "lane", "guide": guide,
                "junction": None if current is None else {"name": current.get("name"), "turn": current.get("turn"),
                                                          "node": current["node"]},
                "hold_at_m": hold, "arrive_in_m": remaining if remaining <= self.arrive_report_m else None, "uturn": uturn,
                # posted limit at the car's front, the fastest legal speed now (in time for a lower limit
                # ahead), the next different limit and how far ahead of the front, and the next junction
                "limit": posted, "legal_cap": legal, "next_limit": next_limit, "next_limit_m": next_limit_m,
                "cruise_limit": float(self._cruise[here]), "cruise_cap": float(self._cruise_cap[here]),
                "junction_m": distance if nxt["node"] is not None and distance <= 60. else None}

    def hud(self):
        """What the viewer shows: destination, distance left, the next instruction, route and pin."""
        if self.destination is None:
            return {"destination": None, "remaining_m": None, "next": None, "route_xy": np.zeros((0, 2)), "pin": None}
        pin = (self.destination.x, self.destination.y)
        if self.route is not None:
            pin = (self.route.destination.x, self.route.destination.y)   # the side of the street it arrives on
        if self.route is None:
            return {"destination": self.destination_name, "remaining_m": None, "next": "No route", "route_xy": np.zeros((0, 2)),
                    "pin": pin}
        progress = self._progress or {"s": 0., "remaining_m": self.route.length, "next": self.route.next_junction(0.)}
        distance, nxt = progress["next"]
        to_turn, _ = self.route.next_uturn(progress["s"])
        if self.arrived:
            text = "Arrived"
        elif to_turn is not None and (nxt["node"] is None or to_turn < distance):
            text = f"turn round in {max(0., to_turn):.0f} m" if to_turn > 1. else "turning round"
        elif nxt["node"] is None:
            text = f"{progress['remaining_m']:.0f} m to go"
        elif nxt.get("turn") == "bend":
            text = "follow the road"
        else:
            text = f"{TURN_WORDS.get(nxt.get('turn'), 'on')} at {nxt.get('name', nxt['node'])} in {max(0., distance):.0f} m"
        return {"destination": self.destination_name, "remaining_m": progress["remaining_m"], "next": text,
                "route_xy": self.route.points[self._index:, :2].copy(), "pin": pin}
