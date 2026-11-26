"""Routes through the connected Indian town for the camera car's sat-nav.

A lane graph built from the town's network file (streets, and the junction
connectors that list every allowed turn) and the map's lane centre lines,
which come in through `sample(road_id, lane, s) -> (x, y, yaw)` so this module
never touches CARLA itself. Positive lanes run from a street's `a` node to its
`b` node (increasing s); negative lanes the other way. Routes keep to the
kerb-side lane of each street (India drives on the left) and are
LoopPath-compatible for the town's junction reservations. A destination on a
two-way street can be reached on either side of it, and a route may turn round
(a U-turn, driven as a three-point turn) on a two-way street, well clear of its
junctions, when that is much shorter than going round the block.
See docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import heapq
import math
from typing import Callable

import numpy as np

STEP = .35            # lane-centre sample spacing (m), as the other vehicles' routes
SNAP_MARGIN = 5.      # a snapped destination stays this far from a street's ends (the junctions)
SNAP_REFUSE = 15.     # clicks further than this from any lane are refused
OFF_ROUTE = 4.
JUNCTION_CRUISE = 2.6
UTURN_COST = 80.      # a U-turn counts as this many metres more (stop, wait for a gap, three or more moves)
UTURN_MARGIN = 25.    # U-turns stay this far from a street's ends (the junctions and their queues)
UTURN_LEAD = 12.      # from the car to a U-turn on the street it is driving (room to slow and stop)
UTURN_CRUISE = 1.
NO_JUNCTION = {"node": None, "start_m": 1e5, "end_m": 1e5 + 1., "road_id": None}
# District spots: (name, street, lane, where along it as a share of its length; None = the bus stop)
DISTRICT_PLACES = (("Bus stop", 2, 3, None), ("Village road", 3, 1, .5), ("Colony street", 5, 1, .5),
                   ("Market street", 1, 1, .5))


@dataclass(frozen=True)
class LanePoint:
    road_id: int
    lane: int
    s: float
    x: float
    y: float
    yaw: float


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Route:
    """An open path from the car to its destination (lane centre line, travel order)."""

    def __init__(self, points, parts, destination):
        self.points = np.asarray(points, float)
        gaps = np.hypot(*np.diff(self.points[:, :2], axis=0).T) if len(self.points) > 1 else np.zeros(0)
        self.distances = np.r_[0., np.cumsum(gaps)]
        self.length = float(self.distances[-1])
        self.destination = destination
        self.parts = parts
        for part in self.parts:
            part["start_m"] = float(self.distances[part.pop("index_start")])
            part["end_m"] = float(self.distances[part.pop("index_end")])
        self.junctions = [p for p in self.parts if p["node"]]
        self.uturns = [p for p in self.parts if p.get("uturn")]
        self.starts = [p["start_m"] for p in self.parts]

    def at(self, s, offset=0.):
        s = float(np.clip(s, 0., self.length))
        i = min(int(np.searchsorted(self.distances, s)), len(self.points) - 1)
        if i == 0:
            p = self.points[0].copy()
        else:
            f = (s - self.distances[i - 1]) / max(self.distances[i] - self.distances[i - 1], 1e-9)
            p = self.points[i - 1].copy()
            p[:2] += f * (self.points[i, :2] - p[:2])
            p[2] += f * wrap(self.points[i, 2] - p[2])
        p[:2] += offset * np.array([-math.sin(p[2]), math.cos(p[2])])
        return p

    def project(self, x, y, last=0):
        lo, hi = max(0, last - 10), min(len(self.points), last + 130)
        window = np.arange(lo, hi)
        d2 = np.sum((self.points[window, :2] - [x, y]) ** 2, axis=1)
        i = int(window[np.argmin(d2)])
        if d2.min() > 36.:   # far from where it was last: search the whole route
            i = int(np.argmin(np.sum((self.points[:, :2] - [x, y]) ** 2, axis=1)))
        return i, float(self.distances[i])

    def part_at(self, s):
        return self.parts[max(0, bisect_right(self.starts, s) - 1)]

    def next_junction(self, s):
        for junction in self.junctions:
            if s <= junction["end_m"] + 8.:
                return junction["start_m"] - s, junction
        return 1e5, dict(NO_JUNCTION)

    def next_uturn(self, s):
        """(metres from s to the next U-turn's start, its part), or (None, None)."""
        for part in self.uturns:
            if s <= part["end_m"]:
                return part["start_m"] - s, part
        return None, None

    def progress(self, x, y, yaw, last=0):
        index, s = self.project(x, y, last)
        offset = float(math.hypot(x - self.points[index, 0], y - self.points[index, 1]))
        return {"index": index, "s": s, "offset_m": offset, "off_route": offset > OFF_ROUTE,
                "remaining_m": max(0., self.length - s)}


class TownRoutes:
    def __init__(self, network: dict, sample: Callable[[int, int, float], tuple[float, float, float]], bus_stop_s: float = 180.):
        self.network = network
        self.streets = {s["id"]: s for s in network["streets"]}
        self.connectors = {c["id"]: c for c in network["connectors"]}
        self.names = network.get("node_names", {})
        degree = {}
        for street in self.streets.values():
            for node in (street["a"], street["b"]):
                degree[node] = degree.get(node, 0) + 1
        self.lanes = {}   # (road_id, lane) -> (N, 4) samples: s, x, y, yaw (s increasing)
        for street in self.streets.values():
            kerb = 3 if street["divided"] else 1
            for lane in (kerb, -kerb):
                self.lanes[(street["id"], lane)] = self._sample(sample, street["id"], lane, street["length"])
        for c in self.connectors.values():
            self.lanes[(c["id"], 1)] = self._sample(sample, c["id"], 1, c["length"])
        self.connector_turn = {}
        for c in self.connectors.values():
            samples = self.lanes[(c["id"], 1)]
            delta = math.degrees(wrap(samples[-1, 3] - samples[0, 3]))
            self.connector_turn[c["id"]] = ("bend" if degree[c["node"]] == 2 else
                                            "right" if delta > 45. else "left" if delta < -45. else "straight")
        # Turns between kerb lanes: (street, sign) -> list of (connector id, outgoing street, outgoing sign).
        self.turns = {}
        chosen = {}
        for c in self.connectors.values():
            key = (c["incoming"], int(np.sign(c["from_lane"])), c["outgoing"], int(np.sign(c["to_lane"])))
            exact = abs(c["from_lane"]) == self._kerb(c["incoming"]) and abs(c["to_lane"]) == self._kerb(c["outgoing"])
            if key not in chosen or (exact and not chosen[key][1]):
                chosen[key] = (c["id"], exact)
        for (street, sign, out, out_sign), (cid, _) in chosen.items():
            self.turns.setdefault((street, sign), []).append((cid, out, out_sign))
        self.places = self._places(bus_stop_s)

    # ---- sampling ---------------------------------------------------------------
    @staticmethod
    def _sample(sample, road_id, lane, length):
        rows = []
        for s in np.arange(.02, length - .02 + 1e-9, STEP):
            x, y, yaw = sample(road_id, lane, float(s))
            rows.append((float(s), x, y, yaw))
        return np.array(rows, float)

    def _kerb(self, street_id):
        return 3 if self.streets[street_id]["divided"] else 1

    def _travel(self, road_id, lane, s_from=None, s_to=None):
        """Lane samples between two s values in the lane's travel order (whole lane by default)."""
        samples = self.lanes[(road_id, lane if road_id in self.streets else 1)]
        if lane < 0 and road_id in self.streets:
            samples = samples[::-1]
            keep = np.ones(len(samples), bool)
            if s_from is not None:
                keep &= samples[:, 0] <= s_from + 1e-6
            if s_to is not None:
                keep &= samples[:, 0] >= s_to - 1e-6
        else:
            keep = np.ones(len(samples), bool)
            if s_from is not None:
                keep &= samples[:, 0] >= s_from - 1e-6
            if s_to is not None:
                keep &= samples[:, 0] <= s_to + 1e-6
        return samples[keep]

    def _point(self, road_id, lane, s):
        samples = self.lanes[(road_id, lane)]
        i = int(np.argmin(np.abs(samples[:, 0] - s)))
        return LanePoint(road_id, lane, float(samples[i, 0]), *map(float, samples[i, 1:]))

    def _places(self, bus_stop_s):
        places = {}
        for node, name in self.names.items():
            street = next(s for s in self.network["streets"] if node in (s["a"], s["b"]))
            kerb = self._kerb(street["id"])
            if street["a"] == node:
                places[name] = self._point(street["id"], kerb, 15.)
            else:
                places[name] = self._point(street["id"], -kerb, street["length"] - 15.)
        for name, street, lane, share in DISTRICT_PLACES:
            length = self.streets[street]["length"]
            s = min(bus_stop_s + 20., length - SNAP_MARGIN) if share is None else length * share   # just past the bus
            places[name] = self._point(street, lane, s)
        return places

    # ---- snapping and locating --------------------------------------------------------
    def snap(self, x, y):
        """A clicked point -> the nearest street lane point on the clicked side, kept SNAP_MARGIN
        from the junctions; None when more than SNAP_REFUSE from every lane."""
        nearest = min(((float(np.min(np.hypot(v[:, 1] - x, v[:, 2] - y))), key) for key, v in self.lanes.items()),
                      key=lambda pair: pair[0])
        if nearest[0] > SNAP_REFUSE:
            return None
        if nearest[1][0] in self.connectors:
            # Inside a junction or on a bend's arc: keep to that connector's own lanes, the street it comes
            # from (just before the junction) or the one it leads to (just after), whichever is nearer.
            # The nearest street sample was often the opposite carriageway (final review, 9 Nov).
            c = self.connectors[nearest[1][0]]
            options = []
            for street, lane, into in ((c["incoming"], c["from_lane"], True), (c["outgoing"], c["to_lane"], False)):
                sign, length = int(np.sign(lane)), self.streets[street]["length"]
                at_junction_end = (sign > 0) == into     # a->b lanes enter at b (s = length) and leave at a
                lane_id = sign * self._kerb(street)
                samples = self.lanes[(street, lane_id)]
                inside = samples[(samples[:, 0] >= SNAP_MARGIN) & (samples[:, 0] <= length - SNAP_MARGIN)]
                row = inside[-1] if at_junction_end else inside[0]
                options.append(LanePoint(street, lane_id, float(row[0]), *map(float, row[1:])))
            return min(options, key=lambda q: math.hypot(q.x - x, q.y - y))
        best = None
        for (road_id, lane), v in self.lanes.items():
            if road_id not in self.streets:
                continue
            inside = (v[:, 0] >= SNAP_MARGIN) & (v[:, 0] <= self.streets[road_id]["length"] - SNAP_MARGIN)
            d = np.where(inside, np.hypot(v[:, 1] - x, v[:, 2] - y), np.inf)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), road_id, lane, v[i])
        _, road_id, lane, row = best
        return LanePoint(road_id, lane, float(row[0]), *map(float, row[1:]))

    def locate(self, x, y, yaw):
        """The car's pose -> the lane (street or connector) it is driving, heading within 60 degrees."""
        best = None
        for (road_id, lane), v in self.lanes.items():
            ok = np.abs((v[:, 3] - yaw + math.pi) % (2 * math.pi) - math.pi) <= math.radians(60.)
            d = np.where(ok, np.hypot(v[:, 1] - x, v[:, 2] - y), np.inf)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), road_id, lane, v[i])
        _, road_id, lane, row = best
        return LanePoint(road_id, lane, float(row[0]), *map(float, row[1:]))

    def point_on(self, road_id, lane, x, y):
        """The point of one given lane nearest to (x, y)."""
        samples = self.lanes[(road_id, lane)]
        row = samples[int(np.argmin(np.hypot(samples[:, 1] - x, samples[:, 2] - y)))]
        return LanePoint(road_id, lane, float(row[0]), *map(float, row[1:]))

    # ---- planning -------------------------------------------------------------------
    def _remaining(self, road_id, lane, s):
        length = self.streets[road_id]["length"] if road_id in self.streets else self.connectors[road_id]["length"]
        return length - s if lane > 0 or road_id not in self.streets else s

    def _end_node(self, street_id, sign):
        street = self.streets[street_id]
        return street["b"] if sign > 0 else street["a"]

    def across(self, point: LanePoint) -> LanePoint | None:
        """The same spot on the other side of a two-way street (None on the divided road, or near its ends)."""
        street = self.streets.get(point.road_id)
        if street is None or street["divided"]:
            return None
        samples = self.lanes[(point.road_id, -point.lane)]
        row = samples[int(np.argmin(np.abs(samples[:, 0] - point.s)))]
        if not SNAP_MARGIN <= row[0] <= street["length"] - SNAP_MARGIN:
            return None
        return LanePoint(point.road_id, -point.lane, float(row[0]), *map(float, row[1:]))

    def _uturn_at(self, street_id, sign, s_in, lead):
        """Where a U-turn on lane (street, sign) entered at s_in happens: at least `lead` metres on and
        UTURN_MARGIN from both ends; None on the divided road or when there is no such place."""
        street = self.streets[street_id]
        if street["divided"]:
            return None
        low, high = UTURN_MARGIN, street["length"] - UTURN_MARGIN
        u = max(s_in + lead, low) if sign > 0 else min(s_in - lead, high)
        return u if low <= u <= high else None

    def plan(self, x, y, yaw, destination: LanePoint, start: LanePoint | None = None,
             either_side=True, u_turns=True) -> Route:
        """The shortest route, a U-turn counted UTURN_COST metres more; `start` (the lane being driven, when
        the caller knows it) overrides locating the car by its pose. With `either_side` a destination on a
        two-way street may be reached on the other side of it; with `u_turns` False the route never turns round."""
        start = start or self.locate(x, y, yaw)
        ends = [destination]
        other = self.across(destination) if either_side else None
        if other is not None:
            ends.append(other)
        # Dijkstra over lane ends ("exit", street, sign); each step keeps the legs it drives:
        # ("street", id, sign, s_from, s_to), ("connector", id, s_from) and ("uturn", id, sign, s).
        heap, best, previous = [], {}, {}
        counter = 0

        def push(cost, state, prev, legs):
            nonlocal counter
            if cost < best.get(state, math.inf):
                best[state] = cost
                previous[state] = (prev, legs)
                counter += 1
                heapq.heappush(heap, (cost, counter, state))

        def enter(cost, street_id, sign, s_in, prev, legs, lead, may_turn=True):
            """Drive lane (street, sign) on from s_in: to a destination on it, to its end, or turn round."""
            length = self.streets[street_id]["length"]
            for k, end in enumerate(ends):
                if (end.road_id, int(np.sign(end.lane))) == (street_id, sign) and (end.s - s_in) * sign >= -1e-6:
                    push(cost + abs(end.s - s_in), ("DEST", k), prev, legs + [("street", street_id, sign, s_in, end.s)])
            push(cost + (length - s_in if sign > 0 else s_in), ("exit", street_id, sign), prev,
                 legs + [("street", street_id, sign, s_in, None)])
            u = self._uturn_at(street_id, sign, s_in, lead) if (u_turns and may_turn) else None
            if u is not None:
                enter(cost + abs(u - s_in) + UTURN_COST, street_id, -sign, u, prev,
                      legs + [("street", street_id, sign, s_in, u), ("uturn", street_id, sign, u)], 0., may_turn=False)

        if start.road_id in self.connectors:
            c = self.connectors[start.road_id]
            out, out_sign = c["outgoing"], int(np.sign(c["to_lane"]))
            entry = 0. if out_sign > 0 else self.streets[out]["length"]
            enter(self._remaining(c["id"], 1, start.s), out, out_sign, entry, "START",
                  [("connector", c["id"], start.s)], 0.)
        else:
            enter(0., start.road_id, int(np.sign(start.lane)), start.s, "START", [], UTURN_LEAD)
        found = None
        while heap:
            cost, _, state = heapq.heappop(heap)
            if cost > best.get(state, math.inf):
                continue
            if state[0] == "DEST":
                found = state
                break
            _, street_id, sign = state
            for cid, out, out_sign in self.turns.get((street_id, sign), []):
                entry = 0. if out_sign > 0 else self.streets[out]["length"]
                enter(cost + self.connectors[cid]["length"], out, out_sign, entry, state, [("connector", cid, None)], 0.)
        if found is None:
            raise ValueError("No route to the destination")
        legs, state = [], found
        while state != "START":
            prev, step = previous[state]
            legs[:0] = step
            state = prev
        return self._build(start, ends[found[1]], legs)

    def _uturn_points(self, street_id, sign, s):
        """A half circle from lane (street, sign) at s to the other side's lane, bulging forward (travel order).
        Returns (N, 4) rows s, x, y, yaw and the distance between the two lane centres."""
        kerb = self._kerb(street_id)
        a, b = self._point(street_id, sign * kerb, s), self._point(street_id, -sign * kerb, s)
        cx, cy, r = (a.x + b.x) / 2, (a.y + b.y) / 2, math.hypot(b.x - a.x, b.y - a.y) / 2
        back = math.atan2(a.y - cy, a.x - cx)            # from the centre to a
        turn = 1. if wrap(math.atan2(b.y - a.y, b.x - a.x) - a.yaw) > 0 else -1.   # b to the right (+) or left
        n = max(8, int(math.pi * r / .2))
        angles = back + turn * np.linspace(0., math.pi, n)
        xs, ys = cx + r * np.cos(angles), cy + r * np.sin(angles)
        yaws = angles + turn * math.pi / 2
        return np.column_stack([np.zeros(n), xs, ys, yaws]), 2 * r

    def _build(self, start, destination, legs):
        points, parts = [], []

        def add(samples, road_id, lane, node, district, cruise, **extra):
            if not len(samples):
                return
            begin = len(points)
            rows = samples[:, 1:4]
            if points and math.hypot(rows[0, 0] - points[-1][0], rows[0, 1] - points[-1][1]) < 1e-3:
                rows = rows[1:]
            if not len(rows):
                return
            points.extend(map(tuple, rows))
            parts.append({"road_id": road_id, "lane": lane, "district": district, "node": node, "cruise": cruise,
                          "index_start": max(0, begin - 1) if extra.get("uturn") else begin,
                          "index_end": len(points) - 1, **extra})

        for leg in legs:
            if leg[0] == "street":
                _, street_id, sign, s_from, s_to = leg
                street, lane = self.streets[street_id], sign * self._kerb(street_id)
                add(self._travel(street_id, lane, s_from, s_to), street_id, lane, None, street["district"], street["cruise"])
            elif leg[0] == "connector":
                _, cid, s_from = leg
                c = self.connectors[cid]
                add(self._travel(cid, 1, s_from, None), cid, 1, c["node"], self.streets[c["incoming"]]["district"], JUNCTION_CRUISE)
                if parts and parts[-1]["road_id"] == cid:
                    parts[-1].update(name=self.names.get(c["node"], c["node"]), turn=self.connector_turn[cid])
            else:
                _, street_id, sign, s = leg
                samples, across = self._uturn_points(street_id, sign, s)
                street = self.streets[street_id]
                # The road's edges from the lane the turn starts in: half a lane to the left, a lane and a half
                # to the right, plus any shoulder (a kerb or sidewalk is not driven on).
                lane_w, shoulder = float(street["width"]), float(street.get("shoulder", 0.))
                add(samples, street_id, sign * self._kerb(street_id), None, street["district"], UTURN_CRUISE,
                    uturn=True, across_m=across, left_m=lane_w / 2 + shoulder, right_m=across + lane_w / 2 + shoulder)
        if not points:
            points.append((start.x, start.y, start.yaw))
            parts.append({"road_id": start.road_id, "lane": start.lane, "district": None, "node": None,
                          "cruise": JUNCTION_CRUISE, "index_start": 0, "index_end": 0})
        return Route(points, parts, destination)
