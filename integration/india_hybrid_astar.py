"""Hybrid A* for junction turns on the camera's top-down road grid (NumPy only).

The camera driver's 0.1 m bird's-eye grid (road model + detected obstacles) is
pooled to 0.25 m cells; cells mostly not-road by the road model, and the ground
under detected objects, are blocked and widened by the car's body discs. The
search runs forward-only arcs from the car (0, 0, 0) in its own frame (x forward,
y right) to a goal pose taken from the sat-nav's route, kept near the route's
lane by a soft cost. No simulator, map or route access.
See docs/CAMERA_DRIVING_DESTINATIONS_DESIGN_20261108.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

FORCE_BUDGET_S: float | None = None   # checks set this to force the time limit (then restore it)
ARC_M = 1.
SUB_STEPS = 4
STEERING_CHANGE_COST = .5
GUIDE_COST = .3            # per metre of distance from the guide, per metre travelled
UNKNOWN_COST = 1.
GOAL_DISTANCE = 1.
GOAL_HEADING = math.radians(15.)
CLOSED_XY = .5
CLOSED_YAW = math.radians(5.)
# The body as discs along its length. Three discs needed 1.32 m clearance each side, too much
# for the town's 2.8 m lanes (8 Nov town runs found 62 of 446 junction plans); seven need 1.16 m.
BODY_DISCS = 7


def disc_offsets(half_length):
    """Forward offsets of the body discs' centres from the car's centre."""
    return half_length * ((2 * np.arange(BODY_DISCS) + 1) / BODY_DISCS - 1)


def disc_radius(half_width, half_length):
    """Each disc covers its share of the body's length and the full width, plus 0.2 m."""
    return math.hypot(half_width, half_length / BODY_DISCS) + .2


@dataclass
class Occupancy:
    blocked: np.ndarray        # bool (nx, ny), already widened by the body-disc radius
    unknown: np.ndarray        # bool (nx, ny): no visible ground in the cell
    xs: np.ndarray             # cell centres, forward (m)
    ys: np.ndarray             # cell centres, lateral (m, right positive)
    cell: float
    half_length: float = 1.9   # for the body discs (Task 2 ruling: plan() needs it)


@dataclass
class PlanResult:
    path: np.ndarray | None    # (M, 3) x, y, yaw every 0.25 m, vehicle frame
    reason: str                # "found" | "no_path" | "budget"
    expansions: int
    ms: float
    cost: float


def occupancy(road, valid, xs, ys, obstacles, half_width, half_length, cell=.25) -> Occupancy:
    road, valid = np.asarray(road, bool), np.asarray(valid, bool)
    fine = float(xs[1] - xs[0])
    x0, y0 = float(xs[0]) - fine / 2, float(ys[0]) - fine / 2
    nx = int(math.ceil((float(xs[-1]) + fine / 2 - x0) / cell))
    ny = int(math.ceil((float(ys[-1]) + fine / 2 - y0) / cell))
    ci = np.minimum(((np.asarray(xs) - x0) / cell).astype(int), nx - 1)
    cj = np.minimum(((np.asarray(ys) - y0) / cell).astype(int), ny - 1)
    I, J = np.meshgrid(ci, cj, indexing="ij")
    visible = np.zeros((nx, ny), int)
    np.add.at(visible, (I[valid], J[valid]), 1)
    off_road = np.zeros((nx, ny), int)
    np.add.at(off_road, (I[valid & ~road], J[valid & ~road]), 1)
    # Majority pooling: a cell is not-road when most of its visible ground is. Stray not-road
    # pixels inside the road would each block a disc the car's width across.
    blocked = 2 * off_road > visible
    cx = x0 + (np.arange(nx) + .5) * cell
    cy = y0 + (np.arange(ny) + .5) * cell
    for o in obstacles:
        bx = (cx >= o["x"] - cell / 2) & (cx <= o["x"] + o["length"] + cell / 2)
        by = (cy >= o["y_left"] - cell / 2) & (cy <= o["y_right"] + cell / 2)
        blocked |= bx[:, None] & by[None, :]
    radius = disc_radius(half_width, half_length)
    reach = int(math.ceil(radius / cell))
    grown = blocked.copy()
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            if (di * di + dj * dj) * cell * cell > radius * radius or (di == 0 and dj == 0):
                continue
            src = blocked[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
            grown[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)] |= src
    return Occupancy(grown, visible == 0, cx, cy, cell, float(half_length))


def _distance_to_polyline(px, py, line):
    """Distance from each point to a polyline (vectorised)."""
    a, b = line[:-1], line[1:]
    d = b - a
    length2 = np.maximum((d ** 2).sum(axis=1), 1e-9)
    t = np.clip(((px[..., None] - a[:, 0]) * d[:, 0] + (py[..., None] - a[:, 1]) * d[:, 1]) / length2, 0., 1.)
    qx, qy = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
    return np.sqrt((px[..., None] - qx) ** 2 + (py[..., None] - qy) ** 2).min(axis=-1)


def _cost_to_go(free, cell_cost, goal_ij, cell):
    """Cheapest 8-connected grid cost from every free cell to the goal cell, where moving a metre
    through a cell costs cell_cost there (1 + the guide penalty). Vectorised relaxation."""
    nx, ny = free.shape
    cost = np.full((nx, ny), np.inf)
    cost[goal_ij] = 0.
    moves = [(di, dj, cell * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    for _ in range(4 * (nx + ny)):
        before = cost
        cost = cost.copy()
        for di, dj, length in moves:
            src = before[max(0, di):nx + min(0, di), max(0, dj):ny + min(0, dj)]
            step = length * cell_cost[max(0, -di):nx + min(0, -di), max(0, -dj):ny + min(0, -dj)]
            dst = cost[max(0, -di):nx + min(0, -di), max(0, -dj):ny + min(0, -dj)]
            np.minimum(dst, src + step, out=dst)
        cost[~free] = np.inf
        if np.array_equal(np.isfinite(cost), np.isfinite(before)) and np.allclose(cost[np.isfinite(cost)], before[np.isfinite(cost)]):
            break
    return cost


def plan(occ: Occupancy, goal, guide, wheelbase, max_steer, budget_s=.15, max_expansions=4000) -> PlanResult:
    started = time.perf_counter()
    budget = FORCE_BUDGET_S if FORCE_BUDGET_S is not None else budget_s
    nx, ny = occ.blocked.shape
    x0, y0, cell = occ.xs[0] - occ.cell / 2, occ.ys[0] - occ.cell / 2, occ.cell
    guide = np.asarray(guide, float)
    guide_cost = GUIDE_COST * _distance_to_polyline(*np.meshgrid(occ.xs, occ.ys, indexing="ij"), guide) if len(guide) > 1 \
        else np.zeros((nx, ny))
    gx, gy, gyaw = map(float, goal)

    def done(reason, path=None, cost=0., expansions=0):
        return PlanResult(path, reason, expansions, (time.perf_counter() - started) * 1000, cost)

    # Free where not blocked; behind the grid's first row (where the car stands) counts as free.
    free = ~occ.blocked
    gi, gj = int((gx - x0) / cell), int((gy - y0) / cell)
    if not (0 <= gi < nx and 0 <= gj < ny) or not free[gi, gj]:
        return done("no_path")
    # Heuristic: the cheapest grid cost to the goal over free cells, the guide penalty included
    # (Task 2 ruling: straight-line distance alone took the right turn past the time limit).
    togo = _cost_to_go(free, 1. + guide_cost, (gi, gj), cell)
    j0 = int((0. - y0) / cell)
    entry = next((i for i in range(nx) if free[i, j0]), None) if 0 <= j0 < ny else None
    if entry is None or not np.isfinite(togo[entry, j0]):
        return done("no_path")

    def heuristic(x, y):
        i, j = int(math.floor((x - x0) / cell)), int(math.floor((y - y0) / cell))
        if 0 <= j < ny and i < 0 and np.isfinite(togo[0, j]):
            return float(togo[0, j]) + (x0 - x)
        if 0 <= i < nx and 0 <= j < ny and np.isfinite(togo[i, j]):
            return float(togo[i, j])
        return math.hypot(x - gx, y - gy)

    deltas = (-1., -.5, 0., .5, 1.)
    steer_max = .7 * max_steer
    step = ARC_M / SUB_STEPS
    # Motion primitives in the car's own frame: SUB_STEPS points per steering angle.
    local = []
    for k in deltas:
        curvature = math.tan(k * steer_max) / wheelbase
        pts = []
        for n in range(1, SUB_STEPS + 1):
            s = n * step
            if abs(curvature) < 1e-9:
                pts.append((s, 0., 0.))
            else:
                yaw = curvature * s
                pts.append((math.sin(yaw) / curvature, (1 - math.cos(yaw)) / curvature, yaw))
        local.append(pts)
    local = np.array(local)                                   # (5, SUB_STEPS, 3)
    disc = disc_offsets(occ.half_length)

    def free_points(px, py):
        i = np.floor((px - x0) / cell).astype(int)
        j = np.floor((py - y0) / cell).astype(int)
        behind = i < 0
        inside = (i < nx) & (j >= 0) & (j < ny)
        ok = np.zeros(px.shape, bool)
        idx = inside & ~behind
        ok[idx] = free[i[idx], j[idx]]
        ok[behind & (j >= 0) & (j < ny)] = True
        return ok, i, j

    def key(x, y, yaw):
        return (round(x / CLOSED_XY), round(y / CLOSED_XY), round(math.remainder(yaw, 2 * math.pi) / CLOSED_YAW))

    start = (0., 0., 0.)
    heap = [(heuristic(0., 0.), 0, 0., start, None, -1)]   # f, counter, g, state, parent id, steer index
    nodes, closed = [], set()
    counter, expansions = 0, 0
    while heap:
        if expansions >= max_expansions or time.perf_counter() - started > budget:
            return done("budget", expansions=expansions)
        f, _, g, state, parent, steer = heapq.heappop(heap)
        k = key(*state)
        if k in closed:
            continue
        closed.add(k)
        nodes.append((state, parent, steer))
        me = len(nodes) - 1
        x, y, yaw = state
        if math.hypot(x - gx, y - gy) <= GOAL_DISTANCE and abs(math.remainder(yaw - gyaw, 2 * math.pi)) <= GOAL_HEADING:
            points, n = [], me
            while n is not None:
                s, p, arc_steer = nodes[n]
                if p is not None:
                    # The arc from the parent's pose, rebuilt here rather than stored for every child.
                    px, py, pyaw = nodes[p][0]
                    pc, ps = math.cos(pyaw), math.sin(pyaw)
                    arc = local[arc_steer]
                    points.append(np.stack([px + arc[:, 0] * pc - arc[:, 1] * ps, py + arc[:, 0] * ps + arc[:, 1] * pc,
                                            pyaw + arc[:, 2]], axis=1))
                n = p
            path = np.vstack([np.array([[0., 0., 0.]])] + points[::-1])
            return done("found", path, g, expansions)
        expansions += 1
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + local[:, :, 0] * c - local[:, :, 1] * s       # (5, SUB_STEPS)
        wy = y + local[:, :, 0] * s + local[:, :, 1] * c
        wyaw = yaw + local[:, :, 2]
        cx = wx[..., None] + disc * np.cos(wyaw)[..., None]   # (5, SUB_STEPS, BODY_DISCS)
        cy = wy[..., None] + disc * np.sin(wyaw)[..., None]
        ok, _, _ = free_points(cx, cy)
        ok = ok.all(axis=(1, 2))
        ei, ej = np.floor((wx[:, -1] - x0) / cell).astype(int), np.floor((wy[:, -1] - y0) / cell).astype(int)
        for n in np.flatnonzero(ok):
            end = (float(wx[n, -1]), float(wy[n, -1]), float(wyaw[n, -1]))
            if key(*end) in closed:
                continue
            cost = g + ARC_M + (STEERING_CHANGE_COST if steer >= 0 and n != steer else 0.)
            if 0 <= ei[n] < nx and 0 <= ej[n] < ny:
                cost += guide_cost[ei[n], ej[n]] * ARC_M + (UNKNOWN_COST if occ.unknown[ei[n], ej[n]] else 0.)
            counter += 1
            heapq.heappush(heap, (cost + heuristic(end[0], end[1]), counter, cost, end, me, int(n)))
    return done("no_path", expansions=expansions)
