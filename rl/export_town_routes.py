"""Export route profiles of the connected town for the RL stand-in (rl/speed_env.py).

Builds the town's map offline (carla.Map from india_combined_map.build_combined(), no CARLA server)
and plans routes with the camera car's own planner (india_route_planner.TownRoutes), then samples
each route every metre: the posted limit (the street's district speed; inside a junction the lower
of the two streets'), the curvature, and which junction (if any) the metre is in.

  .venv-carla/Scripts/python.exe rl/export_town_routes.py

Writes rl/data/town_routes.npz:
  * "eval" routes: every ordered pair of the 11 named places (how the car is used in the town tests)
  * "train" routes: random kerb-lane points to random kerb-lane points (seeded)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from india_rl_speed import posted_limits  # noqa: E402

TURNS = {"bend": 0, "left": 1, "right": 2, "straight": 3}
DISTRICTS = ("market_street", "divided_main_road", "rural_road", "irregular_junction", "residential_colony")


def town():
    import carla
    from india_combined_map import build_combined
    from india_route_planner import TownRoutes
    xml, network = build_combined()
    native = carla.Map("town", xml)

    def sample(road_id, lane, s):
        t = native.get_waypoint_xodr(int(road_id), int(lane), float(s)).transform
        return t.location.x, t.location.y, math.radians(t.rotation.yaw)
    return TownRoutes(network, sample), network


def profile(route, step=1.):
    """Per-metre arrays of one route: limit, curvature, junction index, district; and its junctions."""
    pts = route.points
    d = route.distances
    length = route.length
    s = np.arange(0., length + 1e-9, step)
    yaw = np.unwrap(pts[:, 2])
    yaw_s = np.interp(s, d, yaw)
    # Curvature from the heading's rate of change, smoothed over about 3 m.
    k = np.gradient(yaw_s, step)
    kernel = np.ones(3) / 3.
    curv = np.convolve(np.pad(k, 1, mode="edge"), kernel, mode="valid")
    parts = route.parts
    limit = posted_limits(parts, length, step)[:len(s)]
    junction = -np.ones(len(s), dtype=np.int16)
    district = np.zeros(len(s), dtype=np.int8)
    juncs = []
    for i, part in enumerate(parts):
        lo, hi = part["start_m"], part["end_m"]
        m = (s >= lo - 1e-6) & (s <= hi + 1e-6)
        if part["node"]:
            junction[m] = len(juncs)
            juncs.append((lo, hi, TURNS.get(part.get("turn"), 3), ord(part["node"][0])))
        if part.get("district") in DISTRICTS:
            district[m] = DISTRICTS.index(part["district"])
    return {"length": length, "limit": limit, "curv": curv, "junction": junction, "district": district,
            "junctions": np.array(juncs, dtype=np.float64).reshape(-1, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-routes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--min-length", type=float, default=150.)
    parser.add_argument("--output", type=Path, default=ROOT / "rl" / "data" / "town_routes.npz")
    args = parser.parse_args()
    from india_route_planner import LanePoint
    routes, network = town()
    rng = np.random.default_rng(args.seed)
    profiles, names, split = [], [], []
    places = list(routes.places)
    for a in places:
        for b in places:
            if a == b:
                continue
            p = routes.places[a]
            r = routes.plan(p.x, p.y, p.yaw, routes.places[b], either_side=False, u_turns=False)   # as trained on
            profiles.append(profile(r)); names.append(f"{a} -> {b}"); split.append(1)
    street_lanes = [(rid, lane) for (rid, lane) in routes.lanes if rid in routes.streets]
    tries = 0
    while sum(1 for x in split if x == 0) < args.train_routes and tries < 20 * args.train_routes:
        tries += 1
        picks = []
        for _ in range(2):
            rid, lane = street_lanes[rng.integers(len(street_lanes))]
            v = routes.lanes[(rid, lane)]
            length = routes.streets[rid]["length"]
            inside = v[(v[:, 0] > 8.) & (v[:, 0] < length - 8.)]
            row = inside[rng.integers(len(inside))]
            picks.append(LanePoint(rid, lane, float(row[0]), *map(float, row[1:])))
        a, b = picks
        try:
            r = routes.plan(a.x, a.y, a.yaw, b, start=a, either_side=False, u_turns=False)
        except ValueError:
            continue
        if r.length < args.min_length:
            continue
        profiles.append(profile(r)); names.append(f"{a.road_id}/{a.lane}@{a.s:.0f} -> {b.road_id}/{b.lane}@{b.s:.0f}"); split.append(0)
    L = int(max(len(p["limit"]) for p in profiles)) + 1
    J = int(max(len(p["junctions"]) for p in profiles))
    K = len(profiles)
    limit = np.zeros((K, L), np.float32); curv = np.zeros((K, L), np.float32)
    junction = -np.ones((K, L), np.int16); district = np.zeros((K, L), np.int8)
    jstart = np.full((K, J), 1e9, np.float32); jend = np.full((K, J), 1e9, np.float32)
    jturn = np.zeros((K, J), np.int8); jnode = np.zeros((K, J), np.int16); jcount = np.zeros(K, np.int16)
    length = np.zeros(K, np.float32)
    for i, p in enumerate(profiles):
        n = len(p["limit"])
        limit[i, :n] = p["limit"]; limit[i, n:] = p["limit"][-1]
        curv[i, :n] = p["curv"]; junction[i, :n] = p["junction"]; district[i, :n] = p["district"]
        length[i] = p["length"]
        jc = len(p["junctions"])
        jcount[i] = jc
        if jc:
            jstart[i, :jc], jend[i, :jc] = p["junctions"][:, 0], p["junctions"][:, 1]
            jturn[i, :jc], jnode[i, :jc] = p["junctions"][:, 2], p["junctions"][:, 3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, limit=limit, curv=curv, junction=junction, district=district, length=length,
                        jstart=jstart, jend=jend, jturn=jturn, jnode=jnode, jcount=jcount, split=np.array(split, np.int8),
                        names=np.array(names))
    split = np.array(split)
    summary = {"routes": K, "train": int((split == 0).sum()), "eval": int((split == 1).sum()), "max_length_m": float(length.max()),
               "median_length_m": float(np.median(length)), "max_junctions": J,
               "limits_m_s": sorted(set(np.round(np.unique(limit), 2).tolist()))}
    (args.output.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
