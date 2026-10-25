"""Independent scalar audit of saved CARLA traces (does not import the controller)."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def corners(row):
    x, y, yaw, ex, ey = [float(row[k]) for k in ("x", "y", "yaw_rad", "ex", "ey")]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + a * c - b * s, y + a * s + b * c) for a, b in ((-ex, -ey), (ex, -ey), (ex, ey), (-ex, ey))]


def overlap(a, b):
    for poly in (a, b):
        for p, q in zip(poly, poly[1:] + poly[:1]):
            nx, ny = q[1] - p[1], p[0] - q[0]
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


def gap(a, b):
    if overlap(a, b):
        return 0.
    minimum = float("inf")
    for points, poly in ((a, b), (b, a)):
        for x, y in points:
            for (u, v), (w, z) in zip(poly, poly[1:] + poly[:1]):
                dx, dy = w - u, z - v
                t = min(1., max(0., ((x - u) * dx + (y - v) * dy) / max(dx * dx + dy * dy, 1e-12)))
                minimum = min(minimum, math.hypot(x - u - t * dx, y - v - t * dy))
    return minimum


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * percent))]


def audit(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    incidents = json.loads((directory / "incidents.json").read_text(encoding="utf-8"))["events"]
    rows = list(csv.DictReader((directory / "actor_trace.csv").open(encoding="utf-8")))
    actors, frames, pairs, outside, wrongway = {}, {}, {}, {}, {}
    for row in rows:
        row["polygon"] = corners(row)
        frames.setdefault(row["frame"], []).append(row)
        actors.setdefault(row["name"], []).append(row)
        if row["type"] == "vehicle":
            x, y, yaw = float(row["x"]), float(row["y"]), float(row["yaw_rad"])
            if ((abs(x) > 18 and abs(math.cos(yaw)) > .8 and y * math.cos(yaw) > .1) or
                    (abs(y) > 18 and abs(math.sin(yaw)) > .8 and x * math.sin(yaw) < -.1)):
                wrongway.setdefault(row["name"], {"frame": row["frame"], "time": row["sim_s"]})
            def legal(x, y):
                return ((abs(x) <= 12.2 and abs(y) <= 12.2) or
                        (abs(x) <= 152 and abs(y) <= 4.62) or (abs(y) <= 152 and abs(x) <= 4.62) or
                        (row["name"] == "parked_car" and 12 < x < 152 and -6.9 <= y <= -4.5))
            if any(not legal(x, y) for x, y in row["polygon"]):
                outside.setdefault(row["name"], {"frame": row["frame"], "time": row["sim_s"]})
    min_gap, min_pair = float("inf"), None
    for group in frames.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                distance = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
                bound = math.hypot(float(a["ex"]), float(a["ey"])) + math.hypot(float(b["ex"]), float(b["ey"]))
                if distance > bound + 4:
                    continue
                separation = gap(a["polygon"], b["polygon"])
                if separation < min_gap:
                    min_gap, min_pair = separation, [a["name"], b["name"], a["sim_s"]]
                if separation == 0:
                    pairs.setdefault(tuple(sorted((a["name"], b["name"]))), {"frame": a["frame"], "time": a["sim_s"]})
    stats = {}
    for name, data in actors.items():
        speeds, times = [], []
        distance = 0.
        for previous, current in zip(data, data[1:]):
            dt = float(current["sim_s"]) - float(previous["sim_s"])
            travelled = math.hypot(float(current["x"]) - float(previous["x"]), float(current["y"]) - float(previous["y"]))
            distance += travelled
            speeds.append(travelled / dt)
            times.append(float(current["sim_s"]))
        # Half-second differences reduce the numerical amplification of tiny position noise.
        acceleration = [(speeds[i] - speeds[i - 10]) / (times[i] - times[i - 10]) for i in range(10, len(speeds))]
        jerk = [(acceleration[i] - acceleration[i - 10]) / .5 for i in range(10, len(acceleration))]
        stats[name] = {"distance_m": distance, "max_speed_mps": max(speeds, default=0.),
                       "stopped_s": sum(.05 for v in speeds if v < .15),
                       "braking_500ms_max_mps2": max([-v for v in acceleration] + [0]),
                       "abs_jerk_500ms_p95_mps3": percentile([abs(v) for v in jerk], .95),
                       "abs_jerk_500ms_max_mps3": max([abs(v) for v in jerk] + [0])}
    case = result["case"]
    ego_events = [e for e in incidents if e["actor"] == "ego"]
    checks = {"runtime": result["runtime_passed"], "camera_alignment": result["camera_frames_aligned"],
              "cleanup": result["cleanup_complete"] and not result["cleanup_warnings"],
              "no_native_collisions": not result["collision_events"], "no_body_overlap": not pairs,
              "no_road_departure": not outside, "no_wrongway": not wrongway,
              "progress": stats.get("ego", {}).get("distance_m", 0) > 80}
    if case in ("clear_overtake", "return_gap"):
        checks["completed_pass"] = result["completed_overtakes"] > 0
    if case == "oncoming":
        checks["oncoming_blocks_pass"] = any(e["kind"] == "overtake_blocked" and e.get("reason") == "oncoming_gap" for e in ego_events)
    if case == "pedestrian":
        checks["yielded_to_pedestrian"] = any(e["kind"] == "pedestrian_yield" for e in ego_events)
        checks["person_crossed"] = any(e["kind"] == "pedestrian_finished_crossing" for e in incidents)
    if case == "return_gap":
        checks["return_gap_respected"] = any(e["kind"] == "return_blocked" and "return_guard" in e.get("actors", []) for e in ego_events)
    if case == "junction":
        checks["junction_yield_exercised"] = any(e["kind"] == "junction_yield" for e in incidents)
        checks["ego_cleared_junction"] = any(e["kind"] == "junction_released" for e in ego_events)
    if case == "mixed" and result["sim_s"] >= 100:
        checks["full_route_goal"] = result["ego_goal_reached"]
    # Only claim a complete route when actually observed at a standstill at the goal.
    report = {"passed": all(checks.values()), "case": case, "seed": result["seed"], "checks": checks,
              "completed": result["exit_reason"] == "bounded_test_complete", "exit_reason": result["exit_reason"],
              "health_passed": all(checks[k] for k in ("runtime", "camera_alignment", "cleanup", "no_native_collisions", "no_body_overlap", "no_road_departure", "no_wrongway")),
              "measured_frames": len(frames), "actors": stats, "overlap_pairs": [{"actors": pair, **info} for pair, info in pairs.items()],
              "offroad": outside, "wrongway": wrongway, "min_gap_m": min_gap if math.isfinite(min_gap) else None, "min_gap_pair": min_pair,
              "goal_reached": result["ego_goal_reached"], "events": len(incidents),
              "method": "Independent saved-trace scalar SAT + all rectangle vertices + position-derived speed; 20 Hz sampling"}
    (directory / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    directories = sorted(args.run.glob("episode_*")) if not (args.run / "actor_trace.csv").exists() else [args.run]
    reports = [audit(path) for path in directories if path.is_dir()]
    completed = [r for r in reports if r["completed"]]
    interrupted = [r for r in reports if not r["completed"]]
    config_file = args.run / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    expected = {(case, config.get("seed", 7)) for case in ("clear_overtake", "oncoming", "pedestrian", "return_gap", "junction", "mixed")}
    if config.get("seed", 7) != 42:
        expected.add(("mixed", 42))
    coverage = not config.get("suite") or expected <= {(r["case"], r["seed"]) for r in completed}
    summary = {"passed": bool(completed) and all(r["passed"] for r in completed) and coverage
                           and all(r["exit_reason"] == "replay_requested" and r["health_passed"] for r in interrupted),
               "suite_coverage_complete": coverage, "completed_episodes": len(completed),
               "interrupted_episodes": len(interrupted), "episodes": reports}
    if not (args.run / "actor_trace.csv").exists():
        (args.run / "independent_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "episodes": [{"case": r["case"], "seed": r["seed"], "checks": r["checks"],
                     "ego_distance_m": r["actors"].get("ego", {}).get("distance_m"), "min_gap_m": r["min_gap_m"]} for r in reports]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
