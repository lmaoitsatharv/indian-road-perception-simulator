"""Audit actual speed key handlers, pause/replay and saved native CARLA traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setup/carla_india"))
from verify_run import audit


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config, launcher = load(run / "config.json"), load(run / "launcher_result.json")
    directories = sorted(run.glob("episode_*"))
    results = [load(p / "result.json") for p in directories]
    audits = [audit(p) for p in directories]
    if not (launcher["status"] == "completed" and launcher["viewer_exit_code"] == 0
            and launcher["owned_job_handles_closed"] and len(results) == 2
            and results[0]["exit_reason"] == "replay_requested"
            and results[1]["exit_reason"] == "bounded_test_complete"
            and audits[0]["health_passed"] and audits[1]["passed"]):
        raise ValueError("Native controls, scenario health, bounded completion or owned cleanup did not pass")
    changes = [c for result in results for c in result["playback_speed_changes"]]
    seen = {r["initial_playback_speed"] for r in results} | {c["speed"] for c in changes}
    if seen != {1, 1.5, 2, 2.5, 3} or not {"1", "2", "3"} <= {c["key"] for c in changes}:
        raise ValueError("Not every speed or direct shortcut was exercised by the actual event loop")
    if results[0]["final_playback_speed"] != results[1]["initial_playback_speed"]:
        raise ValueError("Replay did not preserve the selected speed")
    if not all(r["control_tests"][key] for r in results for key in ("pause", "resume", "tab", "screenshot_saved")):
        raise ValueError("Pause, resume, camera or screenshot control was not exercised")
    if not all(r["pause_checks"] and all(p["passed"] for p in r["pause_checks"]) for r in results):
        raise ValueError("The world advanced during a tested pause")
    max_dt_error = 0.
    for directory in directories:
        with (directory / "actor_trace.csv").open(newline="") as stream:
            ego = [row for row in csv.DictReader(stream) if row["name"] == "ego"]
        times = [float(row["sim_s"]) for row in ego]
        max_dt_error = max(max_dt_error, max(abs(b - a - .05) for a, b in zip(times, times[1:])))
    if config["dt"] != .05 or max_dt_error > 1e-5:
        raise ValueError("The fixed simulation step changed")
    measured = [s["measured_speed"] for r in results for s in r["playback_speed_samples"]
                if s["requested_speed"] == 3]
    if not measured or not list(run.glob("episode_*/speed_controls_3x.png")):
        raise ValueError("No measured 3x playback or HUD evidence was saved")
    final = results[-1]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run / "source").glob("*.py")}
    summary = {"passed": True, "run": str(run), "speeds_exercised": sorted(seen),
               "replay_preserved_speed": True, "pause_world_frame_unchanged": True,
               "camera_controls_and_screenshot_passed": True, "fixed_dt_s": .05,
               "max_trace_dt_error_s": max_dt_error, "three_x_samples": len(measured),
               "three_x_median_measured": statistics.median(measured),
               "three_x_min_measured": min(measured), "three_x_max_measured": max(measured),
               "goal_reached": final["ego_goal_reached"], "goal_time_sim_s": final["ego_goal_time_s"],
               "completed_overtakes": final["completed_overtakes"],
               "native_collisions": len(final["collision_events"]),
               "sampled_overlaps": len(final["all_actor_overlaps"]),
               "offroad_events": len(final["offroad_events"]), "wrongway_events": len(final["wrongway_events"]),
               "owned_jobs_closed": True, "source_sha256": source_hashes,
               "limits": "Finite seeded native test; selected speed is a request and actual throughput is hardware dependent"}
    (run / "speed_controls_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
