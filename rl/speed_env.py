"""A fast stand-in for the connected town in which the speed governor is trained (numpy, vectorised).

The car drives one real route of the town (rl/export_town_routes.py: posted limits, curvature and
junctions every metre) towards a destination at its end. Around it, simplified versions of what the
camera car meets in CARLA:

* vehicles ahead on its lane that keep their district speed, slow to 2.7 m/s near junctions, wait
  their turn at junctions, sometimes leave at one, and (buses) stop at the roadside;
* people on the footpaths, some close to the kerb, and jaywalkers who cross in front of the car;
* the town's junction turn-taking: within 35 m of a junction the car waits for its turn (a random
  wait), and not before any vehicle queuing ahead of it has gone;
* perception the way the camera driver has it: a 10 Hz camera, 0.1 s late, missed detections,
  distance errors, people's sideways speed from a short track, phantom vehicles and phantom road
  ends, and the driver's own rules for what is on its path (india_camera_driver.CameraDriver).

The driver's speed constraints are recomputed every 0.05 s from that perception exactly as
CameraDriver.control() does, and its speed controller (CameraDriver._pedals) drives a pedal model fitted
to the camera car in CARLA (9 Nov town run: throttle 0.6 holds speed, 0.8 gives +1.5 m/s^2, full
brake -5.2 m/s^2, 0.1 s lag). Collisions and near misses are judged on the true positions.

Modes (what sets the target speed):
  "rl"     the policy's fraction of the shield (india_rl_speed)
  "shield" always the shield's full speed (fraction 1)
  "legal"  the rule driver with posted limits instead of its 6 m/s cruise
  "rule"   the rule driver as it is (6 m/s cruise everywhere, no posted limits)

Every random draw comes from a counter-based hash of (episode seed, time step or item, stream), so
two modes driving the same episode seed meet the same town (common random numbers).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from india_rl_speed import ACTIONS, CATEGORIES, LIMIT_DECEL, cruise_limit, hold_offset, hold_weight, observe  # noqa: E402

DT = .05
DECISION_TICKS = 2
CAM_TICKS = 2           # 10 Hz camera
LATENCY_TICKS = 2       # 0.1 s from capture to result (session.json latency_ticks)
HL, HW, FRONT = 1.82, .92, 1.82     # the camera car (vehicle.nissan.micra)
# Driver settings (india_camera_driver.DriverSettings defaults)
CRUISE, BRAKE, VEH_GAP, LARGE_GAP, PERSON_GAP, ROAD_END_GAP = 6., 2.5, 4., 6., 5., 3.
EMERGENCY, NEAR_PERSON, LAT_ACCEL, WIDE, JUNC_SPEED, JUNC_FALLBACK = 2.5, 2.5, 2., 3.5, 3.5, 2.5
ARRIVE_SLOW_M, ARRIVE_SPEED, ARRIVE_STOP, HOLD_GAP, HOLD_WITHIN, JUNCTION_LEAD = 30., 3., 1., 3., 35., 20.
PERSON_MARGIN, PERSON_NEAR_M, PERSON_NEAR_RANGE, CROSSING_SPEED = .5, 1., 15., .5
HOLD_S, NEAR_VEHICLE_M, NEAR_VEHICLE_HOLD_S, PASSING = .3, 4., 2.5, 2.5
M, P = 3, 5                                  # vehicle and person slots per env
# Constraint indices (columns of the per-step constraint matrix)
C_LEGAL, C_CURVE, C_VEH, C_PER, C_NEAR, C_HOLD, C_ARR, C_JUNC, C_AREA, C_ROAD, C_OTHER, C_JAHEAD, C_CRUISE = range(13)
N_CONSTRAINTS = 13
JUNCTION_EASE = 2.                # the shield eases down to the junction speed at this rate before the junction zone
DISTRICT_PEOPLE_PER_M = np.array([1 / 45., 1 / 120., 1 / 250., 1 / 100., 1 / 80.])   # market, main, rural, irregular, colony
JAYWALKER_SHARE, HESITATE_SHARE = .12, .05
DISTRICT_VEHICLES_PER_S = np.array([1 / 25., 1 / 20., 1 / 45., 1 / 35., 1 / 35.])
KIND_LEN = np.array([4., 1.9, 10.])          # car, two-wheeler, bus/truck
# Throttle response by speed (band centres; rl/ fit of 12 Nov town drives: gain, offset in m/s^2)
PLANT_V = np.array([.75, 2.25, 3.75, 5., 6., 7., 8.25])
PLANT_GAIN = np.array([3.72, 4.90, 7.38, 10.21, 2.99, 4.34, 5.66])
PLANT_OFFSET = np.array([-1.11, -2.70, -4.49, -6.24, -1.83, -2.51, -3.46])
KIND_GAP = np.array([VEH_GAP, VEH_GAP, LARGE_GAP])

_K = np.array([0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xD6E8FEB86659FD93,
               0xA5A5A5A5A5A5A5A5], dtype=np.uint64)


def hash_uniform(seed, a, b=0, stream=0):
    """Uniform [0, 1) from integer keys (broadcast), splitmix64 finaliser."""
    with np.errstate(over="ignore"):
        z = (np.asarray(seed, np.uint64) * _K[0] ^ np.asarray(a, np.uint64) * _K[1] ^ np.asarray(b, np.uint64) * _K[3]
             ^ np.asarray(stream, np.uint64) * _K[4])
        z = z + _K[0]
        z = (z ^ (z >> np.uint64(30))) * _K[1]
        z = (z ^ (z >> np.uint64(27))) * _K[2]
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1. / 9007199254740992.)


def hash_normal(seed, a, b=0, stream=0):
    u1 = np.maximum(hash_uniform(seed, a, b, stream), 1e-12)
    u2 = hash_uniform(seed, a, b, stream + 7919)
    return np.sqrt(-2. * np.log(u1)) * np.cos(2. * np.pi * u2)


class Routes:
    def __init__(self, path=ROOT / "rl" / "data" / "town_routes.npz"):
        d = np.load(path)
        self.limit, self.curv, self.junction = d["limit"].astype(np.float64), d["curv"].astype(np.float64), d["junction"]
        self.district = d["district"]
        self.length = d["length"].astype(np.float64)
        self.jstart, self.jend, self.jcount = d["jstart"].astype(np.float64), d["jend"].astype(np.float64), d["jcount"]
        self.jturn = d["jturn"]
        self.split, self.names = d["split"], d["names"]
        K, L = self.limit.shape
        # The fastest legal speed at each metre that still meets every lower limit ahead (braking at LIMIT_DECEL).
        req = self.limit.copy()
        for i in range(L - 2, -1, -1):
            req[:, i] = np.minimum(req[:, i], np.sqrt(req[:, i + 1] ** 2 + 2. * LIMIT_DECEL))
        self.legal = req
        # what the learned car aims for (below the car's gear-change band), with the same slowing ahead
        self.cruise = cruise_limit(self.limit)
        req = self.cruise.copy()
        for i in range(L - 2, -1, -1):
            req[:, i] = np.minimum(req[:, i], np.sqrt(req[:, i + 1] ** 2 + 2. * LIMIT_DECEL))
        self.legal_cruise = req
        # The next different limit ahead of each metre: where and what.
        nxt_m = np.full((K, L), np.inf)
        nxt_v = self.limit.copy()
        for i in range(L - 2, -1, -1):
            change = self.limit[:, i + 1] != self.limit[:, i]
            nxt_m[:, i] = np.where(change, 1., nxt_m[:, i + 1] + 1.)
            nxt_v[:, i] = np.where(change, self.limit[:, i + 1], nxt_v[:, i + 1])
        self.next_m, self.next_v = nxt_m, nxt_v
        self.L = L


class TownSpeedEnv:
    def __init__(self, n, routes: Routes | None = None, split="train", seed=0, wait_median=8., episode_m=(150., 650.),
                 max_steps_factor=1.0, people_scale=1., vehicle_scale=1., phantom_scale=1., limit_hold=False,
                 jaywalker_share=JAYWALKER_SHARE, miss_scale=1., stress=0., entry_ease=False, w_hard=.03, w_emergency=.05,
                 gear_dips=False):
        self.n = n
        self.limit_hold = limit_hold      # the learned car raises its set-point to hold a posted limit (india_rl_speed.hold_offset)
        # Harder towns (domain randomisation): more jaywalkers, more missed detections. With stress > 0 each
        # episode draws its own jaywalker share in [share, share + stress * 0.4] and miss scale in [1, 1 + 2 stress].
        self.jaywalker_share, self.miss_scale, self.stress = jaywalker_share, miss_scale, stress
        # (learned car) reach the junction speed at the junction's entry instead of 20 m before it
        self.entry_ease = entry_ease
        self.w_hard, self.w_emergency = w_hard, w_emergency   # reward weights: braking harder than 2.5 m/s^2, emergency stops
        self.gear_dips = gear_dips      # the CARLA car's gearbox drop when the throttle dips to zero at speed
        self.th_prev, self.dip_until = np.zeros(n), np.full(n, -1e9)
        self.ep_jay, self.ep_miss = np.full(n, jaywalker_share), np.full(n, miss_scale)
        self.r = routes or Routes()
        self.route_ids = np.flatnonzero(self.r.split == (0 if split == "train" else 1)) if split in ("train", "eval") \
            else np.arange(len(self.r.length))
        self.base_seed = int(seed)
        self.wait_median, self.episode_m = wait_median, episode_m
        self.max_steps_factor = max_steps_factor
        self.people_scale, self.vehicle_scale, self.phantom_scale = people_scale, vehicle_scale, phantom_scale
        self.episodes_started = 0
        self.fixed = None          # evaluation: list of (route, start_s, seed)
        z = lambda *shape: np.zeros((n,) + shape)
        self.k = np.zeros(n, int); self.seed = np.zeros(n, np.uint64)
        self.s, self.v, self.a, self.t = z(), z(), z(), z()
        self.tick = np.zeros(n, np.int64)
        self.s_start, self.s_goal, self.t_limit = z(), z(), z()
        # Speed controller (CameraDriver._pedals) and the pedal lag
        self.ctl_ac, self.ctl_int, self.ctl_sa, self.ctl_pv, self.ctl_pt = z(), z(), z(), z(), z()
        self.lag_th, self.lag_br = z(LATENCY_TICKS), z(LATENCY_TICKS)
        self.stopped_since = np.full(n, np.nan)
        self.frac = np.ones(n); self.action = np.full(n, len(ACTIONS) - 1)
        self.veh_rate, self.veh_prev = z(), np.full(n, np.inf)
        # Junction turn-taking (per junction of the route)
        J = self.r.jstart.shape[1]
        self.j_drawn, self.j_grant_t, self.j_granted, self.j_violated = np.zeros((n, J), bool), z(J), np.zeros((n, J), bool), np.zeros((n, J), bool)
        self.j_blocked = np.zeros((n, J), bool)
        self.j_planned = z(J)      # how often Hybrid A* finds a path there (fraction of re-plans)
        self.j_mode_planned = np.zeros(n, bool); self.j_mode_t = z()
        # Vehicles ahead on the lane
        self.l_on, self.l_s, self.l_v, self.l_kind = np.zeros((n, M), bool), z(M), z(M), np.zeros((n, M), int)
        self.l_des, self.l_stop_s, self.l_stop_end, self.l_stop_dur = z(M), np.full((n, M), np.inf), np.full((n, M), np.inf), z(M)
        self.l_jgrant, self.l_jdrawn, self.l_jidx, self.l_passed = z(M), np.zeros((n, M), bool), np.zeros((n, M), int), np.zeros((n, M), bool)
        self.l_id = np.zeros((n, M), np.int64); self.l_count = np.zeros(n, np.int64)
        # People (route s, signed lateral offset from the car's path; + is right)
        self.p_on, self.p_s, self.p_lat, self.p_vlat = np.zeros((n, P), bool), z(P), z(P), z(P)
        self.p_mode, self.p_trigger, self.p_speed, self.p_target_lat = np.zeros((n, P), int), z(P), z(P), z(P)
        self.p_pause_at, self.p_pause_end = np.full((n, P), np.inf), np.full((n, P), np.inf)
        self.p_seen, self.p_meas_lat, self.p_meas_t = np.zeros((n, P), int), np.full((n, P, 4), np.nan), np.full((n, P, 4), np.nan)
        self.p_next_s, self.p_count = z(), np.zeros(n, np.int64)
        # Perception results (what the driver holds): one vehicle track, phantoms, people
        self.h_veh_s, self.h_veh_t, self.h_veh_kind, self.h_veh_near = np.full(n, np.nan), np.full(n, -1e9), np.zeros(n, int), np.zeros(n, bool)
        self.pend_veh_gap, self.pend_veh_kind = np.full(n, np.nan), np.zeros(n, int)
        self.ph_s, self.ph_end = np.full(n, np.nan), np.full(n, -1e9)
        self.re_s, self.re_end = np.full(n, np.nan), np.full(n, -1e9)
        self.h_per_s, self.h_per_t, self.h_per_kind = np.full((n, P), np.nan), np.full((n, P), -1e9), np.zeros((n, P), int)  # kind 1 path, 2 near
        self.pend_per = np.full((n, P, 3), np.nan)    # long, lat, vlat estimate at capture
        self.vis_long, self.vis_lat, self.vis_t = np.full((n, P), np.nan), np.full((n, P), np.nan), np.full((n, P), -1e9)
        self.area_until = np.full(n, -1e9)
        # Passing a stopped bus (abstracted)
        self.pass_slot, self.pass_until_s = np.full(n, -1), np.full(n, -np.inf)
        self.behind_stopped_since = np.full(n, np.nan)
        # Episode statistics
        self.stats = {k: z() for k in ("time", "dist", "collisions", "near_miss_s", "person_near_miss_s", "hard_brake_s",
                                         "emergency_s", "speeding_s", "violations", "min_gap", "min_person_clear", "jerk2",
                                         "arrived", "overshoot", "timeout", "wait_s", "over_limit_s", "max_over", "gear_dips")}
        self.done = np.ones(n, bool)
        self.last_cat = np.zeros(n, int)
        self.cap = np.zeros(n)

    # ------------------------------------------------------------------ episodes
    def set_fixed_episodes(self, episodes):
        """Evaluation: each env runs episodes[i] (route, start_s, seed) once, then stays done."""
        assert len(episodes) == self.n
        self.fixed = list(episodes)
        self._reset(np.ones(self.n, bool))

    def reset(self):
        self._reset(np.ones(self.n, bool))
        return self.observe()

    def _reset(self, mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return
        r = self.r
        for i in idx:
            if self.fixed is not None:
                k, s0, seed = self.fixed[i]
            else:
                self.episodes_started += 1
                seed = (self.base_seed * 1_000_003 + self.episodes_started) & 0xFFFFFFFFFFFF
                k = int(self.route_ids[int(hash_uniform(seed, 1, 0, 1) * len(self.route_ids))])
                span = self.episode_m[0] + hash_uniform(seed, 2, 0, 1) * (self.episode_m[1] - self.episode_m[0])
                s0 = max(0., r.length[k] - span)
            self.k[i], self.seed[i] = k, np.uint64(seed)
            self.s[i], self.s_start[i], self.s_goal[i] = s0, s0, r.length[k]
        seed = self.seed[idx]
        u = lambda a, stream=0: hash_uniform(seed, a, 0, 100 + stream)
        self.ep_jay[idx] = self.jaywalker_share + self.stress * .4 * u(90)
        self.ep_miss[idx] = self.miss_scale * (1. + 2. * self.stress * u(91))
        self.v[idx] = np.where(u(1) < .5, 0., u(2) * self._limit(idx, self.s[idx]) * .9)
        self.a[idx] = 0.; self.t[idx] = 0.; self.tick[idx] = (u(3) * 2).astype(np.int64)   # camera phase
        L = self.s_goal[idx] - self.s_start[idx]
        self.t_limit[idx] = (L / 2.5 + 150.) * self.max_steps_factor
        for arr in (self.ctl_ac, self.ctl_int, self.ctl_sa):
            arr[idx] = 0.
        self.ctl_pv[idx] = self.v[idx]; self.ctl_pt[idx] = self.v[idx]
        self.lag_th[idx] = 0.; self.lag_br[idx] = 0.
        self.th_prev[idx] = .6; self.dip_until[idx] = -1e9
        self.stopped_since[idx] = np.nan
        self.frac[idx] = 1.; self.action[idx] = len(ACTIONS) - 1
        self.veh_rate[idx] = 0.; self.veh_prev[idx] = np.inf
        self.j_drawn[idx] = False; self.j_granted[idx] = False; self.j_violated[idx] = False; self.j_blocked[idx] = False
        self.j_grant_t[idx] = np.inf
        js = r.jstart[self.k[idx]]
        self.j_granted[idx] = js < self.s[idx, None] + FRONT + 3.        # already in or past it
        self.j_planned[idx] = hash_uniform(seed[:, None], np.arange(js.shape[1])[None, :], 0, 140) * .8
        self.j_mode_planned[idx] = False; self.j_mode_t[idx] = -1e9
        # vehicles: up to two ahead at the start
        self.l_on[idx] = False; self.l_passed[idx] = False; self.l_count[idx] = 0
        self.l_stop_s[idx] = np.inf; self.l_stop_end[idx] = np.inf
        for m in range(2):
            gap = 12. + u(10 + m) * 45. + m * 25.
            want = u(20 + m) < .55 * self.vehicle_scale
            self._spawn_vehicle(idx[want], self.s[idx[want]] + FRONT + gap[want], m)
        # people
        self.p_on[idx] = False; self.p_count[idx] = 0; self.p_seen[idx] = 0
        self.p_meas_lat[idx] = np.nan; self.p_meas_t[idx] = np.nan
        self.p_next_s[idx] = self.s[idx] + 8. + u(30) * 20.
        # perception
        self.h_veh_s[idx] = np.nan; self.h_veh_t[idx] = -1e9; self.pend_veh_gap[idx] = np.nan
        self.ph_s[idx] = np.nan; self.ph_end[idx] = -1e9; self.re_s[idx] = np.nan; self.re_end[idx] = -1e9
        self.h_per_s[idx] = np.nan; self.h_per_t[idx] = -1e9; self.pend_per[idx] = np.nan
        self.vis_long[idx] = np.nan; self.vis_lat[idx] = np.nan; self.vis_t[idx] = -1e9
        self.area_until[idx] = -1e9
        self.pass_slot[idx] = -1; self.pass_until_s[idx] = -np.inf; self.behind_stopped_since[idx] = np.nan
        for key, arr in self.stats.items():
            arr[idx] = np.inf if key in ("min_gap", "min_person_clear") else -np.inf if key == "max_over" else 0.
        self.done[idx] = False

    # ------------------------------------------------------------------ helpers
    def _metre(self, idx, s):
        return np.clip(s.astype(np.int64), 0, self.r.L - 1)

    def _limit(self, idx, s):
        return self.r.limit[self.k[idx], self._metre(idx, s)]

    def _spawn_vehicle(self, idx, s_rear, slot=None):
        """Place a vehicle (its rear at s_rear) in a free slot of each env in idx."""
        if not len(idx):
            return
        free = ~self.l_on[idx]
        slots = np.argmax(free, axis=1) if slot is None else np.full(len(idx), slot)
        ok = free[np.arange(len(idx)), slots]
        idx, slots, s_rear = idx[ok], slots[ok], np.asarray(s_rear)[ok] if np.ndim(s_rear) else s_rear
        if not len(idx):
            return
        self.l_count[idx] += 1
        seed, key = self.seed[idx], self.l_count[idx]
        u = lambda st: hash_uniform(seed, key, 1, 200 + st)
        kind = np.where(u(1) < .6, 0, np.where(u(1) < .85, 1, 2))
        self.l_on[idx, slots] = True
        self.l_s[idx, slots] = s_rear
        self.l_kind[idx, slots] = kind
        lim = self._limit(idx, s_rear)
        self.l_des[idx, slots] = lim * (.8 + .2 * u(2))
        self.l_v[idx, slots] = self.l_des[idx, slots] * (.5 + .5 * u(3))
        stop = (kind == 2) & (u(4) < .5)
        self.l_stop_s[idx, slots] = np.where(stop, s_rear + 10. + u(5) * 60., np.inf)
        self.l_stop_dur[idx, slots] = 8. + u(6) * 30.
        self.l_stop_end[idx, slots] = np.inf
        self.l_jdrawn[idx, slots] = False
        self.l_jgrant[idx, slots] = np.inf
        self.l_passed[idx, slots] = False
        self.l_id[idx, slots] = key

    def _wait(self, seed, a, b, stream, median):
        """A junction wait: often none, otherwise log-normal (the town's waits run 0-90 s)."""
        u0 = hash_uniform(seed, a, b, stream)
        z = hash_normal(seed, a, b, stream + 1)
        w = np.clip(median * np.exp(.8 * z), .5, 90.)
        return np.where(u0 < .3, 0., w)

    # ------------------------------------------------------------------ world update
    def _update_vehicles(self):
        n, r = self.n, self.r
        on = self.l_on & ~self.done[:, None]
        if not on.any():
            return
        k = self.k[:, None].repeat(M, 1)
        length = KIND_LEN[self.l_kind]
        centre = self.l_s + length / 2.
        lim = r.limit[k, np.clip(centre.astype(np.int64), 0, r.L - 1)]
        target = np.minimum(self.l_des, lim)
        # its next junction
        js, je = r.jstart[self.k], r.jend[self.k]                         # (n, J)
        nxt = np.argmax(je[:, None, :] > centre[:, :, None] - 2., axis=2)  # first junction not yet left
        jstart = np.take_along_axis(js, nxt, 1)
        jend = np.take_along_axis(je, nxt, 1)
        new_j = nxt != self.l_jidx
        self.l_jidx = np.where(on, nxt, self.l_jidx)
        self.l_jdrawn &= ~new_j
        self.l_jgrant = np.where(new_j, np.inf, self.l_jgrant)
        dist = jstart - centre
        near_j = (dist < 18.) & (dist > -(jend - jstart) - 6.)
        target = np.where(near_j, np.minimum(target, np.where(self.l_kind == 2, 1.8, 2.7)), target)
        window = (dist > -2.) & (dist < 40.) & (jstart < 1e8)
        draw = on & window & ~self.l_jdrawn
        if draw.any():
            seed = self.seed[:, None].repeat(M, 1)
            w = self._wait(seed[draw], self.l_id[draw], nxt[draw] + 50, 300, self.wait_median * .6)
            self.l_jgrant[draw] = self.t[:, None].repeat(M, 1)[draw] + w
            self.l_jdrawn[draw] = True
        # a vehicle queued ahead in the same approach goes first
        ahead_q = np.zeros_like(on)
        for m in range(M):
            other = (self.l_on[:, m:m + 1] & (self.l_s[:, m:m + 1] > self.l_s) & (self.l_s[:, m:m + 1] < jstart)
                     & (np.arange(M)[None, :] != m))
            ahead_q |= other
        waiting = window & ~((self.t[:, None] >= self.l_jgrant) & ~ahead_q)
        stop_d = dist - length / 2. - 5.
        target = np.where(waiting, np.minimum(target, np.sqrt(np.maximum(0., 2 * 1.5 * stop_d))), target)
        # roadside stop (buses)
        to_stop = self.l_stop_s - self.l_s
        stopping = np.isfinite(self.l_stop_s)
        target = np.where(stopping, np.minimum(target, np.sqrt(np.maximum(0., 2 * 1.5 * np.maximum(0., to_stop - .5)))), target)
        arrived_stop = stopping & (to_stop < 1.) & (self.l_v < .3) & ~np.isfinite(self.l_stop_end)
        self.l_stop_end = np.where(arrived_stop, self.t[:, None] + self.l_stop_dur, self.l_stop_end)
        release = np.isfinite(self.l_stop_end) & (self.t[:, None] >= self.l_stop_end)
        self.l_stop_s = np.where(release, np.inf, self.l_stop_s)
        self.l_stop_end = np.where(release, np.inf, self.l_stop_end)
        # headway behind the vehicle ahead (another slot)
        for m in range(M):
            front_other = self.l_s[:, m:m + 1]
            gap = front_other - (self.l_s + length)
            behind = self.l_on[:, m:m + 1] & on & (gap > -1.) & (np.arange(M)[None, :] != m) & (front_other > self.l_s)
            target = np.where(behind, np.minimum(target, np.maximum(0., (gap - 4.) / 1.9)), target)
        acc = np.clip(1.2 * (target - self.l_v), -3.5, 1.5)
        self.l_v = np.where(on, np.maximum(0., self.l_v + acc * DT), self.l_v)
        self.l_s = np.where(on, self.l_s + self.l_v * DT, self.l_s)
        # leaving: turns off at a junction exit, or far ahead / behind
        left_j = on & (centre > jend) & (jend < 1e8) & (hash_uniform(self.seed[:, None], self.l_id, nxt, 310) < .5) & \
            (centre - DT * self.l_v <= jend)
        gone = on & ((self.l_s > self.s[:, None] + 90.) | (self.l_s + length < self.s[:, None] - HL - 2.) |
                     (self.l_s > self.s_goal[:, None] + 20.))
        self.l_on &= ~(left_j | gone)
        # new vehicles join ahead, out of view or from a side street
        d = self.r.district[self.k, self._metre(None, self.s)]
        rate = DISTRICT_VEHICLES_PER_S[d] * self.vehicle_scale
        spawn = ~self.done & (hash_uniform(self.seed, self.tick, 0, 320) < rate * DT) & (~self.l_on).any(1)
        if spawn.any():
            gap = 35. + hash_uniform(self.seed, self.tick, 0, 321) * 30.
            s_new = self.s + FRONT + gap
            nearest = np.where(self.l_on, self.l_s, np.inf).min(1)
            far_enough = (np.abs(nearest - s_new) > 12.) & (s_new < self.s_goal - 5.)
            idx = np.flatnonzero(spawn & far_enough)
            self._spawn_vehicle(idx, s_new[idx])

    def _update_people(self):
        live = ~self.done
        # new people ahead along the route, by district density
        due = live & (self.p_next_s < self.s + FRONT + 70.) & (self.p_next_s < self.s_goal + 10.)
        if due.any():
            idx = np.flatnonzero(due)
            free = ~self.p_on[idx]
            has = free.any(1)
            slot = np.argmax(free, 1)
            self.p_count[idx] += 1
            seed, key = self.seed[idx], self.p_count[idx]
            u = lambda st: hash_uniform(seed, key, 2, 400 + st)
            ok = has & (u(0) < self.people_scale)
            i2, sl = idx[ok], slot[ok]
            uu = lambda st: u(st)[ok]
            side = np.where(uu(1) < .75, -1., 1.)       # most on the near (left) footpath
            near_kerb = uu(2) < .06                       # standing right at the kerb (1% "person near" in town runs)
            lat = np.where(side < 0, np.where(near_kerb, -(1.85 + .3 * uu(3)), -(2.25 + 1.4 * uu(3))), 4.6 + 2. * uu(3))
            jay = uu(4) < self.ep_jay[i2]
            self.p_on[i2, sl] = True
            self.p_s[i2, sl] = self.p_next_s[i2] + 0.
            self.p_lat[i2, sl] = lat
            self.p_vlat[i2, sl] = 0.
            self.p_mode[i2, sl] = np.where(jay, 1, 0)
            # Town jaywalkers (india_combined_traffic.JayWalker) wait for a gap: nothing within 12 m and nothing
            # approaching within 12 m + 2.8 s; they then cross at 1.15-1.35 m/s. This one tries once the car is
            # this much further than that gap, and not at all while the car is closer.
            self.p_trigger[i2, sl] = 25. * uu(5)
            self.p_speed[i2, sl] = 1.15 + .2 * uu(6)
            self.p_target_lat[i2, sl] = np.where(lat < 0, 4.8 + 1.5 * uu(7), -(2.4 + 1.2 * uu(7)))
            pause = jay & (uu(8) < HESITATE_SHARE)
            self.p_pause_at[i2, sl] = np.where(pause, -.3 + .9 * uu(9), np.inf)   # lateral position where it hesitates
            self.p_pause_end[i2, sl] = np.inf
            self.p_seen[i2, sl] = 0
            self.p_meas_lat[i2, sl] = np.nan; self.p_meas_t[i2, sl] = np.nan
            dens = DISTRICT_PEOPLE_PER_M[self.r.district[self.k[idx], self._metre(idx, self.p_next_s[idx])]]
            self.p_next_s[idx] += -np.log(np.maximum(u(10), 1e-9)) / dens
        on = self.p_on & live[:, None]
        # jaywalkers start when the car is a few seconds away
        long = self.p_s - (self.s[:, None] + FRONT)
        gap_needed = 12. + 2.8 * self.v[:, None]
        start = on & (self.p_mode == 1) & (long >= gap_needed) & (long <= gap_needed + self.p_trigger)
        self.p_mode = np.where(start, 2, self.p_mode)
        crossing = on & (self.p_mode == 2)
        direction = np.sign(self.p_target_lat - self.p_lat)
        at_pause = crossing & np.isfinite(self.p_pause_at) & (np.abs(self.p_lat - self.p_pause_at) < .1) & ~np.isfinite(self.p_pause_end)
        self.p_pause_end = np.where(at_pause, self.t[:, None] + .8 + 1.5 * hash_uniform(self.seed[:, None], self.p_count[:, None], 3, 410), self.p_pause_end)
        paused = crossing & np.isfinite(self.p_pause_end) & (self.t[:, None] < self.p_pause_end)
        done_pause = crossing & np.isfinite(self.p_pause_end) & (self.t[:, None] >= self.p_pause_end)
        self.p_pause_at = np.where(done_pause, np.inf, self.p_pause_at)
        self.p_vlat = np.where(crossing & ~paused, direction * self.p_speed, 0.)
        self.p_lat = self.p_lat + self.p_vlat * DT
        finished = crossing & (np.abs(self.p_target_lat - self.p_lat) < .1)
        self.p_mode = np.where(finished, 0, self.p_mode)
        self.p_vlat = np.where(finished, 0., self.p_vlat)
        gone = on & (self.p_s < self.s[:, None] - HL - 5.)
        self.p_on &= ~gone

    # ------------------------------------------------------------------ perception (10 Hz, 0.1 s late)
    def _perceive(self):
        live = ~self.done
        cam = live & (self.tick % CAM_TICKS == 0)
        if not cam.any():
            return
        c = np.flatnonzero(cam)
        now = self.t[c]
        # 1. deliver the frame captured LATENCY_TICKS ago
        gap = self.pend_veh_gap[c]
        got = np.isfinite(gap)
        cg = c[got]
        self.h_veh_s[cg] = self.s[cg] - self.v[cg] * DT * LATENCY_TICKS + FRONT + gap[got]   # where it was at capture (odometry)
        self.h_veh_t[cg] = now[got]
        self.h_veh_kind[cg] = self.pend_veh_kind[c][got]
        self.h_veh_near[cg] = gap[got] - self.v[cg] * DT * LATENCY_TICKS <= NEAR_VEHICLE_M
        pp = self.pend_per[c]                                # (nc, P, 3): long, lat, vlat (nan = not seen)
        seen = np.isfinite(pp[:, :, 0])
        s_cap = (self.s[c] - self.v[c] * DT * LATENCY_TICKS)[:, None]
        long_now = pp[:, :, 0] - (self.s[c, None] - s_cap)   # from the front, now
        lat = pp[:, :, 1]
        vlat = pp[:, :, 2]
        v = np.maximum(self.v[c, None], 1.)
        on_path = seen & (np.abs(lat) <= HW + PERSON_MARGIN + .3)
        band = HW + PERSON_MARGIN + .3
        moving_in = seen & np.isfinite(vlat) & (np.abs(vlat) >= CROSSING_SPEED) & (lat * vlat < 0) & (long_now <= PERSON_NEAR_RANGE)
        enter = np.maximum(0., (np.abs(lat) - band) / np.maximum(np.abs(vlat), 1e-6))
        leave = (np.abs(lat) + band) / np.maximum(np.abs(vlat), 1e-6)
        arrive = np.maximum(0., long_now) / v
        gone = arrive + (2 * HL + 1.) / v
        crossing = moving_in & (enter <= gone) & (leave >= arrive)
        near = seen & ~on_path & ~crossing & (long_now < PERSON_NEAR_RANGE) & (np.abs(lat) - .3 - HW <= PERSON_NEAR_M)
        kind = np.where(on_path | crossing, 1, np.where(near, 2, 0))
        hit = kind > 0
        self.vis_long[c] = np.where(seen, long_now, np.nan)
        self.vis_lat[c] = np.where(seen, lat, np.nan)
        self.vis_t[c] = np.where(seen, now[:, None], self.vis_t[c])
        hs = np.where(hit, s_cap + FRONT + pp[:, :, 0], np.nan)
        self.h_per_s[c] = np.where(hit, hs, self.h_per_s[c])
        self.h_per_t[c] = np.where(hit, now[:, None], self.h_per_t[c])
        self.h_per_kind[c] = np.where(hit, kind, self.h_per_kind[c])
        # 2. capture this frame
        seed, tick = self.seed[c], self.tick[c]
        # nearest real vehicle ahead (the others are behind it)
        lead_gap = np.where(self.l_on[c] & ~self.l_passed[c], self.l_s[c] - (self.s[c, None] + FRONT), np.inf)
        lead_gap = np.where(lead_gap > -.5, lead_gap, np.inf)
        j = np.argmin(lead_gap, 1)
        g = lead_gap[np.arange(len(c)), j]
        kind_v = self.l_kind[c, j]
        rng = np.where(kind_v == 2, 50., 40.)
        p_det = 1. - np.clip(.03 + .3 * g / rng, .03, .5) * self.ep_miss[c]
        det = np.isfinite(g) & (g <= rng) & (hash_uniform(seed, tick, 0, 500) < p_det)
        sigma = .04 + .08 * np.clip(g / 40., 0., 1.)
        meas = np.maximum(0., g * (.96 + sigma * hash_normal(seed, tick, 0, 502)))
        self.pend_veh_gap[c] = np.where(det, meas, np.nan)
        self.pend_veh_kind[c] = kind_v
        # people
        plong = self.p_s[c] - (self.s[c, None] + FRONT)
        plat = self.p_lat[c]
        vis = self.p_on[c] & (plong > -.5) & (plong <= 25.) & (np.abs(plat) < np.maximum(plong, 0.) * 1.09 + 1.2)
        p_det = 1. - np.clip(.07 + .3 * plong / 25., .07, .45)[:, :] * self.ep_miss[c][:, None]
        ar = np.arange(P)[None, :]
        detp = vis & (hash_uniform(seed[:, None], tick[:, None], ar, 510) < p_det)
        ml = np.maximum(0., plong * (1. + .06 * hash_normal(seed[:, None], tick[:, None], ar, 512)))
        mlat = plat + (.08 + .008 * np.maximum(plong, 0.)) * hash_normal(seed[:, None], tick[:, None], ar, 514)
        # a short track (last 4 sightings) gives the sideways speed, as CameraDriver._track_people
        self.p_meas_lat[c] = np.where(detp[:, :, None], np.concatenate([self.p_meas_lat[c][:, :, 1:], mlat[:, :, None]], 2), self.p_meas_lat[c])
        self.p_meas_t[c] = np.where(detp[:, :, None], np.concatenate([self.p_meas_t[c][:, :, 1:], now[:, None, None].repeat(P, 1)], 2), self.p_meas_t[c])
        self.p_seen[c] = np.where(detp, self.p_seen[c] + 1, self.p_seen[c])
        tt, ll = self.p_meas_t[c], self.p_meas_lat[c]
        okt = np.isfinite(tt) & (now[:, None, None] - tt <= 1.)
        cnt = okt.sum(2)
        tm = np.where(okt, tt, 0.).sum(2) / np.maximum(cnt, 1)
        lm = np.where(okt, ll, 0.).sum(2) / np.maximum(cnt, 1)
        cov = np.where(okt, (tt - tm[:, :, None]) * (ll - lm[:, :, None]), 0.).sum(2)
        var = np.where(okt, (tt - tm[:, :, None]) ** 2, 0.).sum(2)
        span = np.where(okt, tt, np.inf).min(2)
        vl = np.where((cnt >= 3) & (now[:, None] - span >= .25) & (var > 1e-6), cov / np.maximum(var, 1e-9), np.nan)
        self.pend_per[c] = np.where(detp[:, :, None], np.stack([ml, mlat, vl], 2), np.nan)
        # phantoms: a vehicle box or a road end that is not there
        start_ph = hash_uniform(seed, tick, 0, 520) < (CAM_TICKS * DT / 150.) * self.phantom_scale
        idx = c[start_ph & (self.ph_end[c] < now)]
        self.ph_s[idx] = self.s[idx] + FRONT + 3. + 12. * hash_uniform(self.seed[idx], self.tick[idx], 0, 521)
        self.ph_end[idx] = self.t[idx] + .2 + 1.0 * hash_uniform(self.seed[idx], self.tick[idx], 0, 522)
        start_re = hash_uniform(seed, tick, 0, 530) < (CAM_TICKS * DT / 50.) * self.phantom_scale
        idx = c[start_re & (self.re_end[c] < now)]
        self.re_s[idx] = self.s[idx] + FRONT + 4. + 10. * hash_uniform(self.seed[idx], self.tick[idx], 0, 531)
        self.re_end[idx] = self.t[idx] + .3 + 1.7 * hash_uniform(self.seed[idx], self.tick[idx], 0, 532)
        # wide or uncertain road (mostly inside junctions)
        jmode = self._junction_mode()[c]
        p_area = np.where(jmode, .15, .004)
        hit_area = hash_uniform(seed, tick, 0, 540) < p_area
        self.area_until[c] = np.where(hit_area, now + .3, self.area_until[c])

    def _junction_mode(self):
        js, je = self.r.jstart[self.k], self.r.jend[self.k]
        return ((js - JUNCTION_LEAD <= self.s[:, None]) & (self.s[:, None] <= je)).any(1)

    # ------------------------------------------------------------------ the driver's constraints
    def _constraints(self):
        n, r = self.n, self.r
        C = np.full((n, N_CONSTRAINTS), np.inf)
        emergency = np.zeros(n, bool)
        s, v, t = self.s, self.v, self.t
        k = self.k
        front = s + FRONT
        mf = np.clip(front.astype(np.int64), 0, r.L - 1)
        C[:, C_LEGAL] = r.legal[k, mf]
        C[:, C_CRUISE] = r.legal_cruise[k, mf]          # the learned car's version (gear band)
        # curve (pure pursuit look-ahead), from the route's curvature over the look-ahead
        look = np.clip(4. + .5 * v, 5., 11.)
        ii = np.clip((s[:, None] + np.linspace(0., 1., 6)[None, :] * look[:, None]).astype(np.int64), 0, r.L - 1)
        kap = np.abs(r.curv[k[:, None], ii]).mean(1)
        C[:, C_CURVE] = np.where(kap > 1e-3, np.sqrt(LAT_ACCEL / np.maximum(kap, 1e-6)), np.inf)
        # vehicles: the held track and a phantom
        veh_m = np.full(n, np.inf)
        age = t - self.h_veh_t
        ahead = self.h_veh_s - front
        keep = np.isfinite(self.h_veh_s) & ((age <= HOLD_S + 1e-9) | (self.h_veh_near & (age <= NEAR_VEHICLE_HOLD_S) & (ahead <= NEAR_VEHICLE_M)))
        keep &= ahead >= -1.
        a_ = np.maximum(0., ahead)
        gap = KIND_GAP[self.h_veh_kind]
        veh_c = np.where(keep, np.sqrt(2 * BRAKE * np.maximum(0., a_ - gap)), np.inf)
        emergency |= keep & (a_ < EMERGENCY)
        veh_m = np.where(keep, a_, veh_m)
        ph = t <= self.ph_end
        pa = np.maximum(0., self.ph_s - front)
        ph &= (self.ph_s - front) >= -1.
        veh_c = np.minimum(veh_c, np.where(ph, np.sqrt(2 * BRAKE * np.maximum(0., pa - VEH_GAP)), np.inf))
        emergency |= ph & (pa < EMERGENCY)
        veh_m = np.minimum(veh_m, np.where(ph, pa, np.inf))
        C[:, C_VEH] = np.where(emergency, 0., veh_c)
        # people
        pkeep = (t[:, None] - self.h_per_t <= HOLD_S + 1e-9) & np.isfinite(self.h_per_s)
        pahead = self.h_per_s - front[:, None]
        pkeep &= pahead >= -1.
        pa = np.maximum(0., pahead)
        on_path = pkeep & (self.h_per_kind == 1)
        near = pkeep & (self.h_per_kind == 2)
        pe = (on_path & (pa < EMERGENCY)).any(1)
        per_c = np.where(on_path, np.sqrt(2 * BRAKE * np.maximum(0., pa - PERSON_GAP)), np.inf).min(1)
        C[:, C_PER] = np.where(pe, 0., per_c)
        emergency |= pe
        ped_m = np.where(on_path, pa, np.inf).min(1)
        near_any = near.any(1)
        C[:, C_NEAR] = np.where(near_any, NEAR_PERSON, np.inf)
        # junctions: wait for the turn; steer through at the junction speeds
        js, je = r.jstart[k], r.jend[k]
        nxt = np.argmax(je + 8. >= s[:, None], 1)
        jstart = js[np.arange(n), nxt]
        dist = jstart - s
        granted = self.j_granted[np.arange(n), nxt]
        hold_m = np.where(~granted & (dist >= 0.) & (dist <= HOLD_WITHIN) & (jstart < 1e8), dist - FRONT, np.inf)
        C[:, C_HOLD] = np.where(np.isfinite(hold_m), np.sqrt(2 * BRAKE * np.maximum(0., hold_m - HOLD_GAP)), np.inf)
        jmode = self._junction_mode()
        replan = jmode & (t - self.j_mode_t >= .5)
        if replan.any():
            cur = np.argmax((js - JUNCTION_LEAD <= s[:, None]) & (s[:, None] <= je), 1)
            p_plan = self.j_planned[np.arange(n), cur]
            self.j_mode_planned = np.where(replan, hash_uniform(self.seed, self.tick, 0, 600) < p_plan, self.j_mode_planned)
            self.j_mode_t = np.where(replan, t, self.j_mode_t)
        flat = np.where(self.j_mode_planned, JUNC_SPEED, JUNC_FALLBACK)
        C[:, C_JUNC] = np.where(jmode, flat, np.inf)                      # the rule driver's junction speeds
        # The learned car's junction constraint (C_JAHEAD replaces C_JUNC in its shield): ease down at
        # JUNCTION_EASE to the junction speed by the start of the 20 m junction zone, then that speed; or, with
        # entry_ease, by the junction's entry (the car's front at its start), then that speed inside it.
        approach = (jstart < 1e8) & (dist <= 60.)
        to_zone = dist - JUNCTION_LEAD
        zone = np.where(~jmode & approach & (to_zone > 0.), np.sqrt(JUNC_SPEED ** 2 + 2 * JUNCTION_EASE * np.maximum(0., to_zone)), np.inf)
        C[:, C_JAHEAD] = np.minimum(zone, C[:, C_JUNC])
        if self.entry_ease:
            # only once the turn is granted and the road is not wide or unclear (town drive s42)
            before = approach & (dist > FRONT) & granted & ~(t <= self.area_until)
            v_in = np.where(jmode, flat, JUNC_SPEED)
            ease = np.sqrt(v_in ** 2 + 2 * JUNCTION_EASE * np.maximum(0., dist - FRONT))
            C[:, C_JAHEAD] = np.where(before, ease, C[:, C_JAHEAD])
        # arrival (the stopping curve to the pin is hard; the 3 m/s crawl is the rule driver's own)
        to_go = self.s_goal - s
        arr = np.where(to_go <= ARRIVE_STOP, 0., np.sqrt(2 * BRAKE * np.maximum(0., to_go - .5)))
        C[:, C_ARR] = np.where(to_go <= 60., arr, np.inf)
        # wide/uncertain road; phantom road end
        C[:, C_AREA] = np.where(t <= self.area_until, WIDE, np.inf)
        re = (t <= self.re_end) & ~jmode
        rea = np.maximum(0., self.re_s - front)
        C[:, C_ROAD] = np.where(re, np.sqrt(2 * BRAKE * np.maximum(0., rea - ROAD_END_GAP)), np.inf)
        # passing a stopped bus
        C[:, C_OTHER] = np.where(self.pass_slot >= 0, PASSING, np.inf)
        soft_arrive = np.where(to_go <= ARRIVE_SLOW_M, ARRIVE_SPEED, np.inf)
        ctx = {"veh_m": veh_m, "ped_m": ped_m, "near_ped": near_any.astype(float), "hold_m": hold_m,
               "junction": jmode.astype(float), "to_go": to_go, "dist_junction": np.where(jstart < 1e8, dist, np.inf),
               "soft_arrive": soft_arrive}
        return C, emergency, ctx

    @staticmethod
    def shield(C):
        """The learned car's shield: its speed and the category of what sets it."""
        learned = C.copy()
        learned[:, C_JUNC] = np.inf                       # its own junction constraint is C_JAHEAD
        learned[:, C_LEGAL] = np.inf                      # and its own legal speed C_CRUISE
        cap = learned.min(1)
        cat = np.argmin(learned, 1)
        cat = np.where(cat == C_CRUISE, C_LEGAL, cat)
        return cap, np.where(cat == C_JAHEAD, C_JUNC, cat)

    def _targets(self, C, emergency, ctx, mode):
        if mode in ("rule", "legal"):
            hard_nolegal = C[:, 1:C_JAHEAD].min(1)     # the rule driver's own constraints
            cap = np.minimum(C[:, C_LEGAL], hard_nolegal)
            cat = np.argmin(C[:, :C_JAHEAD], 1)
        else:
            cap, cat = self.shield(C)
        if mode in ("rl", "shield"):
            target = (self.frac if mode == "rl" else 1.) * cap
            if self.limit_hold:
                # holding the posted limit itself (not a lower limit ahead): raise the set-point by the
                # controller's shortfall, never while at or over the limit
                lim_now = self.r.limit[self.k, np.clip((self.s + FRONT).astype(np.int64), 0, self.r.L - 1)]
                aim = cruise_limit(lim_now)
                hold = (cat == C_LEGAL) & (np.abs(cap - aim) < 1e-9) &                     ((self.frac >= 1.) if mode == "rl" else True)
                target = np.where(hold, target + hold_offset(cap) * hold_weight(self.v, aim), target)
        elif mode == "legal":
            target = np.minimum(cap, ctx["soft_arrive"])
        elif mode == "rule":
            target = np.minimum(np.minimum(CRUISE, hard_nolegal), ctx["soft_arrive"])
        else:
            raise ValueError(mode)
        self.cap, self.last_cat = cap, cat
        return np.maximum(0., target)

    # ------------------------------------------------------------------ the car
    def _pedals(self, target, emergency):
        v = self.v
        error = target - v
        self.ctl_sa = .8 * self.ctl_sa + .2 * (v - self.ctl_pv) / DT
        drop = (target < self.ctl_pt - .3) | (error < -.5)
        self.ctl_int = np.where(drop, np.minimum(0., self.ctl_int), self.ctl_int)
        self.ctl_int = np.clip(self.ctl_int + error * DT, -3, 3)
        requested = np.clip(.85 * error + .35 * self.ctl_int - .4 * self.ctl_sa, -3.5, 2.1)
        self.ctl_pv, self.ctl_pt = v.copy(), target.copy()
        self.ctl_ac = self.ctl_ac + np.clip(requested - self.ctl_ac, np.where(emergency, -8., -.15), .12)
        th, br = np.zeros(self.n), np.zeros(self.n)
        hold = (target < .08) & (v < .2)
        slow = ~hold & ~emergency & (error < -.3)
        self.ctl_ac = np.where(slow, np.minimum(self.ctl_ac, 0.), self.ctl_ac)
        pos = ~hold & ~emergency & ~slow & (self.ctl_ac >= 0)
        neg = ~hold & ~emergency & ~slow & (self.ctl_ac < 0)
        br = np.where(hold, .7, br)
        br = np.where(emergency & ~hold, 1., br)
        br = np.where(slow, np.clip(.1 + .16 * -error, 0, .75), br)
        th = np.where(pos, np.clip(.16 + .32 * self.ctl_ac, 0, .85), th)
        br = np.where(neg, np.clip(-self.ctl_ac * .06, 0, .25), br)
        return th, br

    def _plant(self, th, br):
        # 0.1 s pedal lag
        th_d, br_d = self.lag_th[:, 0].copy(), self.lag_br[:, 0].copy()
        self.lag_th = np.concatenate([self.lag_th[:, 1:], th[:, None]], 1)
        self.lag_br = np.concatenate([self.lag_br[:, 1:], br[:, None]], 1)
        v = self.v
        # Pedal model of the camera car, fitted per speed band on three 12 Nov town drives (rule x2, learned):
        # a = gain(v) * throttle + offset(v); light braking already gives about -2.6 m/s^2, full brake -4.9.
        gain = np.interp(v, PLANT_V, PLANT_GAIN)
        a_th = gain * th_d + np.interp(v, PLANT_V, PLANT_OFFSET)
        a_br = -(2.6 + 2.5 * np.maximum(0., br_d - .15))
        a_coast = -(1.5 + .1 * v)
        a = np.where(br_d > 0, a_br, np.where(th_d > 0, a_th, a_coast))
        if self.gear_dips:
            dip_start = (th_d < .05) & (br_d == 0) & (self.th_prev >= .15) & (v > 2.5) & (self.dip_until < self.t)
            self.dip_until = np.where(dip_start, self.t + .5, self.dip_until)
            self.stats["gear_dips"] += dip_start
            a = np.where(self.t < self.dip_until, np.minimum(a, -5.), a)
        self.th_prev = th_d
        a = np.clip(a + .08 * hash_normal(self.seed, self.tick, 0, 700), -6., 2.6)
        v_new = np.maximum(0., v + a * DT)
        a = (v_new - v) / DT
        self.s = self.s + .5 * (v + v_new) * DT
        jerk = (a - self.a) / DT
        self.v, self.a = v_new, a
        return jerk

    # ------------------------------------------------------------------ ego junction turn
    def _ego_junction(self):
        n, r = self.n, self.r
        js, je = r.jstart[self.k], r.jend[self.k]
        ar = np.arange(n)
        nxt = np.argmax(je + 8. >= self.s[:, None], 1)
        jstart = js[ar, nxt]
        dist = jstart - self.s
        live = ~self.done
        window = live & (dist <= HOLD_WITHIN) & (dist >= -5.) & (jstart < 1e8) & ~self.j_granted[ar, nxt]
        draw = window & ~self.j_drawn[ar, nxt]
        if draw.any():
            w = self._wait(self.seed[draw], nxt[draw], 1, 800, self.wait_median)
            self.j_grant_t[ar[draw], nxt[draw]] = self.t[draw] + w
            self.j_drawn[ar[draw], nxt[draw]] = True
        # vehicles queued between the car and the junction go first
        queued = (self.l_on & (self.l_s > self.s[:, None]) & (self.l_s < jstart[:, None]) & ~self.l_passed).any(1)
        was_blocked = self.j_blocked[ar, nxt]
        cleared = window & was_blocked & ~queued
        if cleared.any():
            # it follows them through (same movement) or waits a little longer
            w2 = np.where(hash_uniform(self.seed[cleared], nxt[cleared], 2, 810) < .5, 0.,
                          1. + 6. * hash_uniform(self.seed[cleared], nxt[cleared], 2, 811))
            self.j_grant_t[ar[cleared], nxt[cleared]] = np.maximum(self.j_grant_t[ar[cleared], nxt[cleared]], self.t[cleared] + w2)
        self.j_blocked[ar, nxt] = np.where(window, queued, self.j_blocked[ar, nxt])
        grant = window & ~queued & (self.t >= self.j_grant_t[ar, nxt])
        self.j_granted[ar[grant], nxt[grant]] = True
        self.stats["wait_s"] += np.where(window & (self.v < .3), DT, 0.)
        # entering the junction without the turn breaks the town's right-of-way rule
        inside = live & (self.s + FRONT > jstart + .5) & (jstart < 1e8) & ~self.j_granted[ar, nxt] & ~self.j_violated[ar, nxt]
        if inside.any():
            self.j_violated[ar[inside], nxt[inside]] = True
            self.stats["violations"][inside] += 1
        return inside

    def _passing(self):
        """Passing a stopped bus: after the car has waited 3 s behind it (the camera driver's
        pass_after_stopped_s), it goes round at 2.5 m/s and the bus no longer counts."""
        live = ~self.done
        gaps = np.where(self.l_on & ~self.l_passed, self.l_s - (self.s[:, None] + FRONT), np.inf)
        j = np.argmin(gaps, 1)
        ar = np.arange(self.n)
        g = gaps[ar, j]
        blocker = live & (g < 14.) & (self.l_kind[ar, j] == 2) & (self.l_v[ar, j] < .3) & np.isfinite(self.l_stop_end[ar, j]) \
            & (self.pass_slot < 0)
        stopped = blocker & (self.v < .5)
        self.behind_stopped_since = np.where(stopped, np.where(np.isnan(self.behind_stopped_since), self.t, self.behind_stopped_since), np.nan)
        start = stopped & (self.t - self.behind_stopped_since >= 3.)
        self.pass_slot = np.where(start, j, self.pass_slot)
        self.l_passed[ar[start], j[start]] = True
        self.pass_until_s = np.where(start, self.l_s[ar, j] + KIND_LEN[2] + 4., self.pass_until_s)
        finished = (self.pass_slot >= 0) & (self.s - HL > self.pass_until_s)
        if finished.any():
            self.l_on[ar[finished], self.pass_slot[finished]] = False
            self.pass_slot = np.where(finished, -1, self.pass_slot)

    # ------------------------------------------------------------------ truth
    def _truth(self, jerk):
        n = self.n
        live = ~self.done
        front = self.s + FRONT
        length = KIND_LEN[self.l_kind]
        gap = np.where(self.l_on & ~self.l_passed, self.l_s - front[:, None], np.inf)
        overlap_rear = self.l_s + length > self.s[:, None] - HL
        gap = np.where(overlap_rear, gap, np.inf)
        g = gap.min(1)
        lead_v = np.where(gap == g[:, None], self.l_v, np.inf).min(1)
        hit_v = live & (g <= 0.)
        closing = self.v > lead_v + .2
        near_v = live & ~hit_v & (g < 1.) & closing & (self.v > .5)
        # people (body radius 0.3 m)
        plong = self.p_s - self.s[:, None]
        inside_long = (plong >= -HL - .3) & (plong <= FRONT + .3)
        clear = np.abs(self.p_lat) - HW - .3
        hit_p = live & (self.p_on & inside_long & (clear <= 0.)).any(1) & (self.v > .05)
        ahead_long = (plong >= -HL) & (plong <= FRONT + 2.5)
        near_p = live & ~hit_p & (self.p_on & ahead_long & (clear <= .5)).any(1) & (self.v > 1.)
        pc = np.where(self.p_on & ahead_long, clear, np.inf).min(1)
        self.stats["min_gap"] = np.where(live, np.minimum(self.stats["min_gap"], g), self.stats["min_gap"])
        self.stats["min_person_clear"] = np.where(live & (self.v > .5), np.minimum(self.stats["min_person_clear"], pc), self.stats["min_person_clear"])
        hard = live & (self.a < -3.)
        lim_now = self.r.limit[self.k, np.clip(front.astype(np.int64), 0, self.r.L - 1)]
        speeding = live & (self.v > lim_now + .5)
        self.stats["over_limit_s"] += np.where(live & (self.v > lim_now + .05), DT, 0.)
        self.stats["max_over"] = np.where(live, np.maximum(self.stats["max_over"], self.v - lim_now), self.stats["max_over"])
        st = self.stats
        st["collisions"] += hit_v | hit_p
        st["near_miss_s"] += np.where(near_v, DT, 0.)
        st["person_near_miss_s"] += np.where(near_p, DT, 0.)
        st["hard_brake_s"] += np.where(hard, DT, 0.)
        st["speeding_s"] += np.where(speeding, DT, 0.)
        st["jerk2"] += np.where(live, jerk ** 2 * DT, 0.)
        return hit_v | hit_p, near_v, near_p, hard, speeding, np.where(speeding, self.v - lim_now - .5, 0.)

    # ------------------------------------------------------------------ step
    def step(self, actions=None, mode="rl"):
        """One decision (0.1 s): two control steps at the held action. Returns reward, ended (this step).
        Call observe() (or reset_done() while training) for the next observation."""
        if actions is not None:
            a = np.asarray(actions, int)
            dfrac = np.where(self.done, 0., np.abs(ACTIONS[a] - ACTIONS[self.action]))
            self.frac = np.where(self.done, self.frac, ACTIONS[a])
            self.action = np.where(self.done, self.action, a)
        else:
            dfrac = np.zeros(self.n)
        reward = np.zeros(self.n)
        live0 = ~self.done
        s0 = self.s.copy()
        ended = np.zeros(self.n, bool)
        for _ in range(DECISION_TICKS):
            live = ~self.done
            self.tick += live
            self.t += np.where(live, DT, 0.)
            self._update_vehicles()
            self._update_people()
            self._passing()
            self._perceive()
            C, emergency, ctx = self._constraints()
            target = self._targets(C, emergency, ctx, mode)
            target = np.where(live, target, 0.)
            th, br = self._pedals(target, emergency)
            v_before, s_before = self.v.copy(), self.s.copy()
            jerk = self._plant(np.where(live, th, 0.), np.where(live, br, 1.))
            self.v = np.where(live, self.v, v_before)
            self.s = np.where(live, self.s, s_before)
            violated = self._ego_junction()
            hit, near_v, near_p, hard, speeding, over = self._truth(jerk)
            self.stats["emergency_s"] += np.where(live & emergency, DT, 0.)
            self.stats["time"] += np.where(live, DT, 0.)
            self.stopped_since = np.where(self.v < .5, np.where(np.isnan(self.stopped_since), self.t, self.stopped_since), np.nan)
            reward += np.where(live, -.25 * near_v - .5 * near_p - self.w_hard * np.maximum(0., -self.a - 2.5) - self.w_emergency * emergency
                               - .3 * over - .002 * np.minimum(jerk ** 2, 400.) * DT - 10. * violated, 0.)
            to_go = self.s_goal - self.s
            arrived = live & (to_go <= 2.) & (self.v < .5)
            overshoot = live & (to_go < -2.5)
            timeout = live & (self.t >= self.t_limit)
            newly = live & (hit | arrived | overshoot | timeout)
            reward += np.where(live & hit, -40., 0.) + np.where(arrived, 4., 0.) + np.where(overshoot, -5., 0.)
            self.stats["arrived"] += arrived
            self.stats["overshoot"] += overshoot
            self.stats["timeout"] += timeout & ~arrived & ~hit
            ended |= newly
            self.done |= newly
        progress = np.where(live0, self.s - s0, 0.)
        self.stats["dist"] += progress
        reward += np.where(live0, .1 * progress - .02 - .01 * dfrac, 0.)
        return reward, ended

    def observe(self):
        """The policy's observation now (and the driver-side smoothing of the lead vehicle's distance rate)."""
        r = self.r
        C, emergency, ctx = self._constraints()
        cap, cat = self.shield(C)
        front = self.s + FRONT
        mf = np.clip(front.astype(np.int64), 0, r.L - 1)
        k = self.k
        nlm = r.next_m[k, mf] - (front - np.floor(front))
        nlm = np.where(nlm <= 80., nlm, np.inf)
        veh_m = ctx["veh_m"]
        rate = np.where(np.isfinite(veh_m) & np.isfinite(self.veh_prev) & (np.abs(veh_m - self.veh_prev) < 5.),
                        (veh_m - self.veh_prev) / (DT * DECISION_TICKS), np.nan)
        self.veh_rate = np.where(np.isfinite(veh_m), np.where(np.isfinite(rate), .7 * self.veh_rate + .3 * np.clip(rate, -15, 15),
                                                             self.veh_rate), 0.)
        self.veh_prev = veh_m
        ii = np.clip((self.s[:, None] + np.arange(0., 31., 2.)[None, :]).astype(np.int64), 0, r.L - 1)
        curv = np.abs(r.curv[k[:, None], ii]).max(1)
        vis = (self.t[:, None] - self.vis_t <= HOLD_S + 1e-9) & np.isfinite(self.vis_long) & (self.vis_long >= -1.) & \
            (self.vis_long <= 20.)
        people_n = vis.sum(1).astype(float)
        people_clear = np.where(vis, np.abs(np.nan_to_num(self.vis_lat)) - HW - .3, np.inf).min(1)
        return observe({"v": self.v, "accel": self.ctl_sa, "cap": cap, "limit": r.limit[k, mf], "next_limit": np.where(np.isfinite(nlm), r.next_v[k, mf], r.limit[k, mf]),
                        "next_limit_m": nlm, "veh_m": veh_m, "veh_rate": self.veh_rate, "ped_m": ctx["ped_m"],
                        "near_ped": ctx["near_ped"], "hold_m": ctx["hold_m"], "junction": ctx["junction"],
                        "junction_m": np.where(ctx["dist_junction"] <= 60., ctx["dist_junction"], np.inf),
                        "arrive_m": np.where(ctx["to_go"] <= 60., ctx["to_go"], np.inf), "curv": curv,
                        "stopped_s": np.where(np.isnan(self.stopped_since), 0., self.t - np.nan_to_num(self.stopped_since)),
                        "prev_frac": self.frac, "area": (C[:, C_AREA] < np.inf).astype(float), "cap_cat": cat,
                        "people_n": people_n, "people_clear": people_clear})

    def reset_done(self):
        """Training: start new episodes where the last ones ended. Returns the observations."""
        if self.fixed is None:
            done = self.done.copy()
            self._reset(done)
            self.veh_rate[done] = 0.
            self.veh_prev[done] = np.inf
        return self.observe()


def run_episodes(env: TownSpeedEnv, mode: str, policy=None, max_decisions=200000):
    """Drive every fixed episode to its end in one mode; returns the stats (arrays, one value per episode)."""
    obs = env.observe()
    total = np.zeros(env.n)
    for _ in range(max_decisions):
        if env.done.all():
            break
        actions = policy(obs) if mode == "rl" else None
        reward, _ = env.step(actions, mode=mode)
        total += reward
        obs = env.observe()
    out = {k: v.copy() for k, v in env.stats.items()}
    out["return"] = total
    return out
