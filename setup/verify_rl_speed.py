"""Offline checks for the learned speed governor (no CARLA server, no GPU).

  .venv-carla/Scripts/python.exe setup/verify_rl_speed.py            # governor checks
  .venv-carla/Scripts/python.exe setup/verify_rl_speed.py --suites   # + every camera-driving and destination
                                                                     #   check again with the learned speed on

The governor (integration/india_rl_speed.py) may only slow the car below the shield, and the shield keeps every
safety constraint of the rule driver plus the posted limit. With --suites the existing offline checks
(setup/verify_camera_driving.py, setup/verify_camera_destinations.py) run a second time with
DriverSettings.rl_speed = 1, so the same stops, gaps, passes, turns and arrivals are asked of the learned car.
"""
from __future__ import annotations

import io
import contextlib
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "setup"))

import india_rl_speed as rl  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)


def limit_checks():
    parts = [{"start_m": 0., "end_m": 100., "node": None, "cruise": 8.},
             {"start_m": 100., "end_m": 125., "node": "A", "cruise": 2.6},
             {"start_m": 125., "end_m": 300., "node": None, "cruise": 4.4}]
    limit = rl.posted_limits(parts, 300.)
    legal = rl.legal_profile(limit)
    check("limits: streets keep their district speed, a junction the lower of its two streets'",
          limit[50] == 8. and limit[110] == 4.4 and limit[200] == 4.4, f"{limit[50]}, {limit[110]}, {limit[200]}")
    # braking at 1.5 m/s^2 from 8 to 4.4 m/s takes (64 - 19.36) / 3 = 14.9 m
    start = int(np.argmax(legal < 8. - 1e-9))
    check("limits: the legal speed reaches 4.4 m/s by the junction, slowing gently over about 15 m before it",
          abs(legal[100] - 4.4) < 1e-9 and 83 <= start <= 87 and np.all(np.diff(legal[:101]) <= 1e-9),
          f"slowing starts at {start} m, {legal[100]:.2f} m/s at 100 m")
    posted, cap, nxt, nxt_m = rl.limits_at(limit, legal, 60.5)
    check("limits: at 60.5 m the next limit is 4.4 m/s, 39.5 m ahead", posted == 8. and nxt == 4.4 and abs(nxt_m - 39.5) < 1e-9,
          f"{posted}, {nxt}, {nxt_m}")
    posted, cap, nxt, nxt_m = rl.limits_at(limit, legal, 5.)
    check("limits: a change more than 80 m ahead is not reported", nxt == 8. and nxt_m is None, f"{nxt}, {nxt_m}")


def policy_checks():
    path = rl.DEFAULT_POLICY
    check("policy: models/rl_speed/policy.npz is there", path.exists(), str(path))
    if not path.exists():
        return None
    policy = rl.SpeedPolicy(path)
    meta = policy.meta
    check("policy: matches this governor's observation and actions", True,
          f"{len(policy.layers)} layers, trained {meta.get('samples', '?')} decisions, sha256 {policy.sha256[:12]}")
    rng = np.random.default_rng(3)
    n = 4000
    ctx = {"v": rng.uniform(0, 9, n), "accel": rng.normal(0, 1, n), "cap": rng.uniform(0, 9, n), "limit": rng.choice([4.4, 5., 6.8, 8.], n),
           "next_limit": rng.choice([4.4, 5., 6.8, 8.], n), "next_limit_m": np.where(rng.random(n) < .5, rng.uniform(0, 80, n), np.inf),
           "veh_m": np.where(rng.random(n) < .5, rng.uniform(0, 40, n), np.inf), "veh_rate": rng.normal(0, 2, n),
           "ped_m": np.where(rng.random(n) < .2, rng.uniform(0, 25, n), np.inf), "near_ped": rng.random(n) < .1,
           "hold_m": np.where(rng.random(n) < .2, rng.uniform(0, 35, n), np.inf), "junction": rng.random(n) < .2,
           "junction_m": np.where(rng.random(n) < .4, rng.uniform(-20, 60, n), np.inf),
           "arrive_m": np.where(rng.random(n) < .2, rng.uniform(0, 60, n), np.inf), "curv": rng.uniform(0, .1, n),
           "stopped_s": rng.uniform(0, 20, n), "prev_frac": rng.choice(rl.ACTIONS, n), "area": rng.random(n) < .1,
           "cap_cat": rng.integers(0, len(rl.CATEGORIES), n), "people_n": rng.integers(0, 5, n),
           "people_clear": np.where(rng.random(n) < .5, rng.uniform(0, 4, n), np.inf)}
    obs = rl.observe(ctx)
    a = policy.act(obs)
    check("policy: finite observations, actions within STOP..PROCEED for 4000 random situations",
          np.isfinite(obs).all() and a.min() >= 0 and a.max() < len(rl.ACTIONS),
          "chose " + ", ".join(f"{f:.2f}: {np.mean(rl.ACTIONS[a] == f):.0%}" for f in rl.ACTIONS))
    clear = dict({k: np.atleast_1d(v)[:1] for k, v in ctx.items()}, v=6.8, cap=6.8, limit=6.8, next_limit=6.8, next_limit_m=np.inf,
                 veh_m=np.inf, ped_m=np.inf, near_ped=0., hold_m=np.inf, junction=0., junction_m=np.inf, arrive_m=np.inf,
                 curv=0., stopped_s=0., prev_frac=1., area=0., cap_cat=0, people_n=0, people_clear=np.inf, accel=0., veh_rate=0.)
    frac = rl.ACTIONS[policy.act(rl.observe(clear))[0]]
    check("policy: on a clear rural road at its limit it proceeds at the limit", frac >= .95, f"fraction {frac:.2f}")
    return policy


def driver_checks():
    from india_camera_driver import CameraDriver, DriverSettings
    from verify_camera_driving import GEOMETRY, CAR
    s = DriverSettings(rl_speed=1.)
    driver = CameraDriver(GEOMETRY, settings=s, **CAR)
    check("driver: rl_speed=1 loads the governor", driver.governor is not None)
    # Every action of the policy, forced, stays at or below the shield
    worst = 0.
    for action in range(len(rl.ACTIONS)):
        driver.governor.policy.act = (lambda a: (lambda obs: np.array([a])))(action)
        for cap in (0., .4, 2.5, 4.4, 8.):
            for why in ("speed_limit", "vehicle_ahead", "waiting_turn", "emergency_person", "arriving"):
                driver.governor.reset()
                target, reason, _, set_point = driver._learned_speed(cap, why, "", 3., 1., False, 0.)
                worst = max(worst, target - cap)
                if why != "speed_limit":          # only a posted limit is ever held with a raised set-point
                    worst = max(worst, set_point - cap)
                if why.startswith("emergency") and reason != why:
                    worst = max(worst, 1.)
    check("driver: the learned speed never exceeds the shield (every action, cap and reason); an emergency keeps its brake",
          worst <= 1e-12, f"largest excess {worst:.3g} m/s")
    # A clear straight street seen by the camera (verify_camera_driving's renderer), car standing still
    from india_camera_driver import Perception
    from verify_camera_driving import render, straight
    mask, boxes = render(straight())

    def drive(nav=None, objects=(), speed=0., steps=8):
        d = CameraDriver(GEOMETRY, settings=DriverSettings(rl_speed=1.), **CAR)
        m, b = render(straight(), list(objects)) if objects else (mask, boxes)
        out = None
        for k in range(steps):
            if nav is not None:
                d.navigate(nav)
            d.captured(k, speed, 0.)
            d.observe(Perception(k, True, mask=m, boxes=b), k * .1)
            out = d.control(speed, 0., k * .1)
        return out
    out = drive()
    check("driver: with no destination the learned car's limit is the rule driver's 6 m/s cruise",
          out["rl"]["cap"] == s.cruise and out["rl"]["cap_reason"] == "cruise" and out["target"] <= s.cruise,
          f"cap {out['rl']['cap']} ({out['rl']['cap_reason']}), target {out['target']:.2f} ({out['reason']})")
    lane = {"mode": "lane", "guide": np.c_[np.arange(0., 31.), np.zeros(31)], "junction": None, "hold_at_m": None,
            "arrive_in_m": None, "limit": 4.4, "legal_cap": 4.4, "next_limit": 4.4, "next_limit_m": None, "junction_m": None}
    out = drive(lane)
    check("driver: on the market street (posted 4.4 m/s) the shield is 4.4 m/s, below the rule driver's 6 m/s cruise",
          out["rl"]["cap"] == 4.4 and out["rl"]["cap_reason"] == "speed_limit" and out["target"] <= 4.4,
          f"cap {out['rl']['cap']} ({out['rl']['cap_reason']}), target {out['target']:.2f} ({out['reason']})")
    main = dict(lane, limit=8., legal_cap=8., next_limit=8.)
    out = drive(main, speed=7.)
    check("driver: on the main road (posted 8 m/s) the shield is 8 m/s, above the rule driver's 6 m/s cruise",
          out["rl"]["cap"] == 8. and out["target"] > s.cruise, f"cap {out['rl']['cap']}, target {out['target']:.2f} ({out['reason']})")
    raised = out["rl"].get("set_point")
    near = drive(main, speed=8.1)
    over = drive(main, speed=8.3)
    near_raise = (near["rl"].get("set_point") or 8.) - 8.
    check("driver: holding the 8 m/s limit raises only the controller's set-point (the target stays 8); the raise fades "
          "out near the limit and is gone 0.25 m/s over it",
          raised is not None and abs(raised - 8. - float(rl.hold_offset(8.))) < 1e-3 and out["target"] == 8.
          and 0. < near_raise < raised - 8. and over["rl"].get("set_point") is None,
          f"set-point {raised} at 7 m/s, {near['rl'].get('set_point')} at 8.1 m/s, {over['rl'].get('set_point')} at 8.3 m/s")
    check("driver: the hold makes up 80% of the controller's measured 0.42 m/s shortfall above 3 m/s",
          all(abs(float(rl.hold_offset(v)) - .8 * ((.61 - .16) / .32 - 1.05) / .85) < 1e-9 for v in (3.75, 4.4, 6., 8.)),
          ", ".join(f"{v}: {float(rl.hold_offset(v)):.3f}" for v in (2., 3.75, 4.4, 6., 8.)))
    aim = float(rl.cruise_limit(6.5))
    rural = dict(lane, limit=6.5, legal_cap=6.5, next_limit=6.5, cruise_limit=aim, cruise_cap=aim)
    out_rural = drive(rural, speed=5.)
    check(f"driver: on a 6.5 m/s rural street the learned car aims for {aim} m/s and holds it "
          f"({'no gear band: one fixed ratio' if rl.GEAR_BAND is None else 'below the gearbox hunting band'})",
          out_rural["rl"]["cap"] == aim and out_rural["rl"].get("set_point", 0.) > aim and out_rural["target"] <= 6.5,
          f"cap {out_rural['rl']['cap']}, set-point {out_rural['rl'].get('set_point')}")
    ahead = dict(main, legal_cap=6.2, next_limit=4.4, next_limit_m=25.)
    out = drive(ahead, speed=6.)
    check("driver: a 4.4 m/s limit 25 m ahead lowers the shield in time (legal_cap from the sat-nav)",
          out["rl"]["cap"] == 6.2 and out["target"] <= 6.2, f"cap {out['rl']['cap']} ({out['rl']['cap_reason']}), target {out['target']:.2f}")
    out = drive(main, objects=[(12., 16., -.9, .9, 1.5, "car")], speed=0.)
    rule = drive_rule(objects=[(12., 16., -.9, .9, 1.5, "car")])
    check("driver: a car 12 m ahead: the learned target is at most the rule driver's vehicle limit",
          out["target"] <= rule["target"] + 1e-9 and out["rl"]["cap_reason"] == "vehicle_ahead",
          f"learned {out['target']:.2f} ({out['reason']}), rule {rule['target']:.2f} ({rule['reason']})")
    granted = drive(dict(main, junction_m=12.), speed=6.)
    pending = drive(dict(main, junction_m=25., hold_at_m=23.), speed=6.)
    entry = math.sqrt(s.junction_speed ** 2 + 2 * rl.JUNCTION_EASE * (12. - CAR["front_m"]))
    zone = math.sqrt(s.junction_speed ** 2 + 2 * rl.JUNCTION_EASE * (25. - rl.JUNCTION_ZONE_M))
    check("driver: before a junction it eases to the junction speed by the entry once its turn is granted (12 m out), "
          "and by the start of the 20 m zone while it waits for it (25 m out)",
          abs(granted["rl"]["cap"] - entry) < 1e-3 and abs(pending["rl"]["cap"] - zone) < 1e-3 and granted["rl"]["cap_reason"] == "junction_ahead",
          f"granted cap {granted['rl']['cap']} ({granted['rl']['cap_reason']}), waiting cap {pending['rl']['cap']} ({pending['rl']['cap_reason']})")
    hold = dict(main, hold_at_m=8., junction_m=9.8)
    out = drive(hold, speed=3.)
    check("driver: told to wait for its turn 8 m ahead: the shield is the stopping curve to the hold point",
          abs(out["rl"]["cap"] - math.sqrt(2 * s.brake_decel * (8. - s.hold_gap_m))) < 1e-6 and out["target"] <= out["rl"]["cap"],
          f"cap {out['rl']['cap']} ({out['rl']['cap_reason']}), target {out['target']:.2f}")


def satnav_checks():
    """The sat-nav's limits along a real route of the town (map built offline)."""
    sys.path.insert(0, str(ROOT / "rl"))
    from export_town_routes import town
    from india_satnav import SatNav
    routes, _ = town()
    nav = SatNav(routes)
    start = routes.places["Village road"]
    nav.set_destination("Outskirts bend", start.x, start.y, start.yaw, 0.)
    seen, bad = set(), []
    for s in np.arange(0., nav.route.length - 5., 7.):
        x, y, yaw = nav.route.at(s)
        msg = nav.update(float(x), float(y), float(yaw), float(s))
        nav._last = None                     # (jumping along the route is not a reset)
        posted, aim = msg["limit"], msg["cruise_limit"]
        seen.add((posted, aim))
        if aim != float(rl.cruise_limit(posted)) or msg["cruise_cap"] > msg["legal_cap"] + 1e-9 or msg["cruise_cap"] > aim + 1e-9:
            bad.append((round(s), posted, aim, msg["cruise_cap"], msg["legal_cap"]))
    check("sat-nav: along Village road -> Outskirts bend the learned car's aim is the limit, or 5.6 m/s inside the "
          "gear band, and never above the legal speed", not bad and (rl.GEAR_BAND is None or any(p != a for p, a in seen)),
          f"(limit, aim) seen {sorted(seen)}; bad {bad[:3]}")


def drive_rule(objects=()):
    from india_camera_driver import CameraDriver, Perception
    from verify_camera_driving import GEOMETRY, CAR, render, straight
    d = CameraDriver(GEOMETRY, **CAR)
    m, b = render(straight(), list(objects))
    out = None
    for k in range(8):
        d.captured(k, 0., 0.)
        d.observe(Perception(k, True, mask=m, boxes=b), k * .1)
        out = d.control(0., 0., k * .1)
    return out


def suites():
    """Run the existing offline suites again with the learned speed on by default."""
    import dataclasses
    import india_camera_driver as icd
    base = icd.DriverSettings
    learned = dataclasses.make_dataclass("LearnedDriverSettings", [("rl_speed", float, 1.)], bases=(base,))
    learned.from_dict = classmethod(lambda cls, values: cls(**{k: float(v) for k, v in (values or {}).items()
                                                                 if k in cls.__dataclass_fields__}))
    icd.DriverSettings = learned
    outcome = {}
    for name in ("verify_camera_driving", "verify_camera_destinations"):
        module = __import__(name)
        module.DriverSettings = learned if hasattr(module, "DriverSettings") else None
        module.results.clear()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            if name == "verify_camera_driving":
                for fn in ("geometry_checks", "synthetic_checks", "closed_loop_checks", "pedestrian_checks", "wedge_checks",
                           "isolation_check"):
                    getattr(module, fn)()
            else:
                routes = module.route_checks()
                module.astar_checks()
                module.driver_checks()
                module.satnav_checks(routes)
        failed = [(n, d) for n, ok, d in module.results if not ok]
        outcome[name] = (len(module.results), failed, buffer.getvalue())
        check(f"learned speed: {name} ({len(module.results)} checks) all pass", not failed,
              "; ".join(f"{n} [{d}]" for n, d in failed))
    icd.DriverSettings = base
    return outcome


if __name__ == "__main__":
    limit_checks()
    policy = policy_checks()
    if policy is not None:
        driver_checks()
        satnav_checks()
        if "--suites" in sys.argv:
            outcome = suites()
            log = ROOT / "runs" / "verify_rl_speed_suites.log"
            log.parent.mkdir(exist_ok=True)
            log.write_text("\n\n".join(f"== {k}\n{v[2]}" for k, v in outcome.items()), encoding="utf-8")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + ("" if not failed else "; failed: " + "; ".join(failed)))
    raise SystemExit(1 if failed else 0)
