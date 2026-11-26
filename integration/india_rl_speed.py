"""Learned speed governor for the camera car: a small policy picks the target speed, a fixed shield bounds it.

The camera driver already works out every reason to go slower (a vehicle or person on its path, the
junction it must wait at, a curve, the destination, wide or uncertain road, ...). Each is one
(speed, reason) constraint. Without the policy the driver takes the lowest of them, with a fixed 6 m/s
cruise on top. With the policy (DriverSettings.rl_speed = 1):

* The **shield** is the lowest of the hard constraints: every safety constraint the rule driver has,
  unchanged, plus the posted speed limit of the street (from the sat-nav's map), slowing in time for
  a lower limit ahead. Only two rule-driver choices are not in it, because they are about pace, not
  safety: the fixed 6 m/s cruise (replaced by the posted limit) and the 3 m/s crawl over the last 30 m
  to a destination (the stopping curve to the pin stays).
* The **policy** picks a fraction of the shield's speed, ten times a second, from what the driver
  itself knows (its speed, the shield and why, the nearest vehicle and person, the junction and
  destination ahead, the next limit). It can only slow the car below the shield, never speed it up.

So the policy cannot break a rule or a safety constraint the shield enforces; it learns when going
at the shield's speed costs more (hard braking, near misses with what perception sees late) than
it gains. It is trained in rl/speed_env.py, a fast stand-in for the town, by rl/train_speed_ppo.py.
Pure numpy and no simulator handle, so it runs inside the camera driver's process.
See docs/RL_SPEED_DESIGN_20261112.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

# The policy's choices, as fractions of the shield's speed: STOP, four SLOW levels, PROCEED.
ACTIONS = np.array([0., .3, .5, .65, .78, .88, .95, 1.], dtype=np.float32)
PROCEED_ACTION = len(ACTIONS) - 1
DECISION_DT = .1                 # the policy decides ten times a second; the driver holds it in between
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "models" / "rl_speed" / "policy.npz"

# Why the shield is as low as it is, grouped (the observation carries the group, one-hot).
CATEGORIES = ("legal", "curve", "vehicle", "person", "person_near", "hold", "arrival", "junction", "area",
              "road_end", "other")
CATEGORY = {"speed_limit": 0, "cruise": 0, "curve": 1, "vehicle_ahead": 2, "emergency_vehicle": 2,
            "person_ahead": 3, "emergency_person": 3, "person_near_path": 4, "waiting_turn": 5,
            "arriving": 6, "arrived": 6, "junction_turn": 7, "junction_fallback": 7, "junction_ahead": 7, "wide_road_area": 8,
            "uncertain_road": 8, "lost_road": 8, "lost_road_stop": 8, "road_ends": 9}
# Pace choices of the rule driver that the shield leaves to the policy (everything else is hard).
SOFT = frozenset({"cruise", "arriving_slow"})

# Normalising ranges of the observation.
VEHICLE_RANGE_M, PERSON_RANGE_M, JUNCTION_RANGE_M, ARRIVE_RANGE_M, LIMIT_RANGE_M = 40., 30., 60., 60., 80.
SPEED_SCALE = 10.
OBS_FIELDS = ("v", "accel", "cap", "limit", "next_limit", "next_limit_m", "veh_m", "veh_rate", "ped_m",
              "near_ped", "hold_m", "junction", "junction_m", "arrive_m", "curv", "stopped_s", "prev_frac",
              "area", "cap_cat", "people_n", "people_clear")
OBS_SIZE = 24 + len(CATEGORIES)   # see observe()
PEOPLE_RANGE_M = 20.             # people counted ahead (any side) for people_n / people_clear

# Shield constants shared by the driver and the stand-in town.
LIMIT_DECEL = 1.5                # slow for a lower posted limit ahead at this rate (gentle, before the sign)
JUNCTION_EASE = 2.               # ease down to the junction speed at this rate before the junction zone (20 m out)
JUNCTION_ZONE_M = 20.            # the sat-nav switches to junction steering this far before a junction


HOLD_GAIN = .8                   # share of the speed controller's shortfall made up when holding a posted limit


# Throttle that holds a speed on the camera car (vehicle.nissan.micra), fitted per speed band on three 12 Nov town
# drives (54,000 steps, 0-8 m/s): about 0.61 at every speed from 3 m/s up, less below (rl/README notes).
HOLD_THROTTLE_V = (.75, 2.25, 3.75)
HOLD_THROTTLE = (.30, .55, .61)


def hold_offset(speed):
    """How much to raise the speed controller's set-point so the car actually drives at `speed`.

    CameraDriver._pedals settles below its target: its integral can supply at most 1.05 m/s^2 of the
    "acceleration" that, through its throttle mapping (0.16 + 0.32 x), must hold the speed; the rest comes from
    the speed error. With the measured holding throttle (0.61 from 3 m/s up) that is 0.42 m/s at every speed
    (town runs: 0.49 m/s under a 6 m/s target). Only HOLD_GAIN of it is made up, so the car settles just under the
    limit. Arrays or floats."""
    v = np.asarray(speed, dtype=float)
    throttle = np.interp(v, HOLD_THROTTLE_V, HOLD_THROTTLE)
    command = (throttle - .16) / .32
    return HOLD_GAIN * np.maximum(0., (command - 1.05) / .85)


HOLD_FADE_M_S = .5               # the raised set-point fades out over this band of speed around the limit
# The camera car's automatic gearbox hunts when it is asked to hold 5.9-7 m/s: it shifts up at about 5.9 m/s, loses
# ~2 m/s at -5 m/s^2 while it shifts, shifts back down and repeats (12 Nov town drives rl01 and s42_rl: 3.9 <-> 6.4
# m/s every 2.5 s on the 6.0 m/s rural street). The rule car cruises at 5.5 m/s, just under it. On a street posted
# inside this band the learned car cruised at GEAR_BAND_CRUISE instead; above it (the 8 m/s main road) it passes the
# shift once and holds 7.8 m/s. Since the user's 12 Nov decision the camera car has one fixed ratio and never shifts
# (india_combined_scene.single_gear), so the band is off: set GEAR_BAND = (5.6, 7.3) again for the stock gearbox
# (setup/run_camera_driving.py --keep-gearbox).
GEAR_BAND = None
GEAR_BAND_CRUISE = 5.6           # with the hold this settled near 5.45 m/s, the rule car's proven-steady speed


def cruise_limit(posted):
    """The speed the learned car aims for under a posted limit: the limit, except inside GEAR_BAND when one is set."""
    p = np.asarray(posted, dtype=float)
    if GEAR_BAND is None:
        return p.copy()
    return np.where((p > GEAR_BAND[0]) & (p < GEAR_BAND[1]), GEAR_BAND_CRUISE, p)


def hold_weight(speed, posted):
    """How much of hold_offset to apply: all of it up to 0.25 m/s under the limit, none from 0.25 m/s over, linear
    between. Fading (not switching it off at the limit) matters in CARLA: a set-point that drops 0.5-0.8 m/s at
    once resets the speed controller's integral, the throttle dips to zero and the car's automatic gearbox shifts,
    dropping 2 m/s at -5 m/s^2 (12 Nov town run rl01, 18 such drops on the rural roads)."""
    return np.clip((np.asarray(posted, dtype=float) + HOLD_FADE_M_S / 2 - np.asarray(speed, dtype=float)) / HOLD_FADE_M_S, 0., 1.)


def category(why: str) -> int:
    return CATEGORY.get(why, len(CATEGORIES) - 1)


def limit_ahead_cap(next_limit, next_limit_m, front_m=0., decel=LIMIT_DECEL):
    """The highest speed now that still reaches a lower limit `next_limit_m` ahead (of the car's centre)
    by the time the car's front gets there. Arrays or floats."""
    room = np.maximum(0., np.asarray(next_limit_m, dtype=float) - front_m)
    return np.sqrt(np.asarray(next_limit, dtype=float) ** 2 + 2. * decel * room)


def posted_limits(parts, length, step=1.):
    """Posted limit every `step` metres of a route: each street's district speed; inside a junction, the lower
    of the two streets' it joins (parts: india_route_planner.Route.parts, with start_m/end_m/node/cruise)."""
    s = np.arange(0., float(length) + 1e-9, step)
    limit = np.zeros(len(s))
    for i, part in enumerate(parts):
        inside = (s >= part["start_m"] - 1e-6) & (s <= part["end_m"] + 1e-6)
        if part["node"]:
            beside = [p["cruise"] for p in (parts[i - 1] if i > 0 else None, parts[i + 1] if i + 1 < len(parts) else None)
                      if p is not None and not p["node"]]
            limit[inside] = min(beside) if beside else part["cruise"]
        else:
            limit[inside] = part["cruise"]
    for i in np.flatnonzero(limit <= 0):          # a metre between two parts takes its neighbour's
        limit[i] = limit[i - 1] if i > 0 else limit[np.flatnonzero(limit > 0)[0]]
    return limit


def legal_profile(limit, step=1., decel=LIMIT_DECEL):
    """The fastest legal speed at each metre that still meets every lower limit ahead."""
    req = np.asarray(limit, float).copy()
    for i in range(len(req) - 2, -1, -1):
        req[i] = min(req[i], math.sqrt(req[i + 1] ** 2 + 2. * decel * step))
    return req


def limits_at(limit, legal, front_s, horizon=80.):
    """Posted limit, legal speed and the next different limit (value, metres from the front) at the car's
    front `front_s` along a 1 m profile."""
    n = len(limit)
    i = int(np.clip(math.floor(front_s), 0, n - 1))
    ahead = np.flatnonzero(limit[i + 1:min(n, i + 1 + int(horizon) + 1)] != limit[i])
    if len(ahead):
        j = i + 1 + int(ahead[0])
        next_limit, next_m = float(limit[j]), float(j - front_s)
        if next_m > horizon:
            next_limit, next_m = float(limit[i]), None
    else:
        next_limit, next_m = float(limit[i]), None
    return float(limit[i]), float(legal[i]), next_limit, next_m


def _clip01(x):
    return np.clip(x, 0., 1.)


def observe(ctx: dict) -> np.ndarray:
    """Observation(s) from the driver's own knowledge. Every value of `ctx` is an array of shape (N,)
    (or a float, for one car). Missing distances are np.inf. Returns float32 (N, OBS_SIZE).

    Fields: v, accel (smoothed, m/s^2), cap (the shield's speed), limit (posted, now), next_limit and
    next_limit_m (the next different limit within LIMIT_RANGE_M, else limit and inf), veh_m (nearest
    vehicle on the path, from the front), veh_rate (its distance's rate of change, m/s), ped_m (nearest
    person on the path), near_ped (0/1), hold_m (where to wait for the junction, or inf), junction (0/1,
    steering through one), junction_m (next junction's start, or inf), arrive_m (destination, or inf),
    curv (largest |curvature| of the route in the next 30 m, 1/m), stopped_s, prev_frac (last action's
    fraction), area (0/1 wide/uncertain/lost road), cap_cat (category index of the shield's reason),
    people_n (people seen within PEOPLE_RANGE_M ahead, either side), people_clear (the smallest gap from
    the car's side to one of them, m, or inf).
    """
    c = {k: np.atleast_1d(np.asarray(ctx[k], dtype=np.float64)) for k in OBS_FIELDS}
    n = max(len(v) for v in c.values())
    c = {k: np.broadcast_to(v, (n,)) for k, v in c.items()}

    def dist(x, scale):
        present = np.isfinite(x)
        return np.where(present, _clip01(np.where(present, x, 0.) / scale), 1.), present.astype(np.float64)

    veh, veh_on = dist(c["veh_m"], VEHICLE_RANGE_M)
    ped, ped_on = dist(c["ped_m"], PERSON_RANGE_M)
    hold, hold_on = dist(c["hold_m"], JUNCTION_RANGE_M)
    junc, junc_on = dist(c["junction_m"], JUNCTION_RANGE_M)
    arrive, arrive_on = dist(c["arrive_m"], ARRIVE_RANGE_M)
    nlm, _ = dist(c["next_limit_m"], LIMIT_RANGE_M)
    cols = [
        c["v"] / SPEED_SCALE,
        np.clip(c["accel"] / 3., -2., 2.),
        c["cap"] / SPEED_SCALE,
        c["limit"] / SPEED_SCALE,
        (c["next_limit"] - c["limit"]) / SPEED_SCALE,
        nlm,
        veh, veh_on,
        np.where(veh_on > 0, np.clip(np.where(veh_on > 0, c["veh_rate"], 0.) / 5., -2., 2.), 0.),
        ped, ped_on,
        c["near_ped"],
        hold, hold_on,
        c["junction"],
        junc,
        arrive, arrive_on,
        np.clip(c["curv"] * 10., 0., 3.),
        np.clip(c["stopped_s"] / 10., 0., 1.),
        c["prev_frac"],
        c["area"],
        np.clip(c["people_n"] / 5., 0., 1.),
        np.where(np.isfinite(c["people_clear"]), np.clip(np.where(np.isfinite(c["people_clear"]), c["people_clear"], 0.) / 4., 0., 1.), 1.),
    ]
    onehot = np.zeros((n, len(CATEGORIES)))
    onehot[np.arange(n), np.clip(c["cap_cat"].astype(int), 0, len(CATEGORIES) - 1)] = 1.
    out = np.concatenate([np.stack(cols, axis=1), onehot], axis=1).astype(np.float32)
    assert out.shape[1] == OBS_SIZE, out.shape
    return out


class SpeedPolicy:
    """The trained policy (an MLP saved by rl/train_speed_ppo.py) evaluated with numpy."""

    def __init__(self, path: Path | str = DEFAULT_POLICY):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.layers = []
        i = 0
        while f"w{i}" in data:
            self.layers.append((data[f"w{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64)))
            i += 1
        self.obs_mean = data["obs_mean"].astype(np.float64)
        self.obs_std = data["obs_std"].astype(np.float64)
        self.meta = json.loads(str(data["meta"])) if "meta" in data else {}
        if self.layers[0][0].shape[0] != OBS_SIZE or self.layers[-1][0].shape[1] != len(ACTIONS):
            raise ValueError(f"{self.path} does not match this governor's observation or actions")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        x = np.clip(x, -10., 10.)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if k < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Greedy action indices, shape (N,)."""
        return np.argmax(self.logits(obs), axis=1)


class SpeedGovernor:
    """Inside the camera driver: holds the policy's choice for DECISION_DT and applies it under the shield."""

    def __init__(self, policy: SpeedPolicy | None = None, path: Path | str = DEFAULT_POLICY):
        self.policy = policy or SpeedPolicy(path)
        self.reset()

    def reset(self):
        self.action = PROCEED_ACTION
        self.next_decision = -1e9
        self.prev_frac = 1.
        self.last_obs = None
        self.veh_m = self.veh_time = None
        self.veh_rate = 0.
        self.decisions = 0

    def vehicle_rate(self, veh_m, sim_time):
        """Rate of change of the nearest vehicle's distance (smoothed); 0 when none is seen."""
        if not math.isfinite(veh_m):
            self.veh_m, self.veh_rate = None, 0.
            return 0.
        if self.veh_m is not None and sim_time > self.veh_time and abs(veh_m - self.veh_m) < 5.:
            rate = (veh_m - self.veh_m) / (sim_time - self.veh_time)
            self.veh_rate = .7 * self.veh_rate + .3 * float(np.clip(rate, -15., 15.))
        self.veh_m, self.veh_time = veh_m, sim_time
        return self.veh_rate

    def choose(self, ctx: dict, sim_time: float) -> float:
        """The fraction of the shield's speed to drive at now."""
        if sim_time >= self.next_decision - 1e-9:
            # (the lead vehicle's distance rate is smoothed at the decision rate, as in rl/speed_env.py)
            ctx = dict(ctx, prev_frac=self.prev_frac, veh_rate=self.vehicle_rate(float(ctx["veh_m"]), sim_time))
            self.last_obs = observe(ctx)
            self.action = int(self.policy.act(self.last_obs)[0])
            self.next_decision = sim_time + DECISION_DT
            self.prev_frac = float(ACTIONS[self.action])
            self.decisions += 1
        return float(ACTIONS[self.action])
