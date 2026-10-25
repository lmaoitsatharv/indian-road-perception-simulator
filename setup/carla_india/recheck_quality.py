"""Fresh owned CARLA sessions and independent audits for the V4 scene changes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from verify_run import audit

ROOT = Path(__file__).resolve().parents[2]
PLAN = [("mixed", 7, 110, "clear"), ("mixed", 42, 110, "clear"), ("mixed", 19, 110, "clear"),
        ("clear_overtake", 7, 60, "clear"), ("pedestrian", 7, 60, "overcast"),
        ("oncoming", 7, 60, "wet"), ("return_gap", 7, 60, "dusk")]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New local runs/ manifest directory")
    parser.add_argument("--observe-road", action="store_true", help="Also record paired model cameras for mixed seeds 7/42")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Keep suite evidence in a new runs/ directory")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    status = {"created_utc": datetime.now(timezone.utc).isoformat(), "scenario_version": "lht-neighbourhood-v4.2",
              "model_control": False, "rl_trained": False, "episodes": records, "completed": False}
    for case, seed, duration, weather in PLAN:
        argv = [sys.executable, "-u", "setup/carla/run_session.py", "--india", "--smoke", "--quality", "Low",
                "--surface-style", "asphalt", "--weather", weather, "--case", case, "--seed", str(seed),
                "--speed", "3", "--sim-seconds", str(duration), "--seconds", "420"]
        if args.observe_road and case == "mixed" and seed in (7,42):
            argv.append("--observe-road")
        previous = load(ROOT / "setup/carla/server_owner.json").get("output") if (ROOT / "setup/carla/server_owner.json").exists() else None
        launched = subprocess.run(argv, cwd=ROOT)
        owner = load(ROOT / "setup/carla/server_owner.json")
        current_output = owner["output"] if owner["output"] != previous else None
        record = {"case": case, "seed": seed, "weather": weather, "output": current_output,
                  "launcher_exit_code": launched.returncode, "owned_jobs_closed": owner.get("owned_job_handles_closed"),
                  "argv": argv}
        records.append(record)
        try:
            if not current_output or launched.returncode or owner.get("status") != "completed" or not owner.get("owned_job_handles_closed"):
                raise RuntimeError("Owned native session did not complete cleanly")
            episode = next(Path(owner["output"]).glob("episode_*"))
            measured = audit(episode)
            result, events = load(episode / "result.json"), load(episode / "incidents.json")["events"]
            crossing = {p["name"] for p in load(episode / "actors.json")["pedestrians"] if p["crossing"]}
            finished = {e["actor"] for e in events if e["kind"] == "pedestrian_finished_crossing"}
            views = {f["view"] for f in load(episode / "camera_frames.json")["frames"]}
            controls_ok = (bool(result["pause_checks"]) and all(p["passed"] for p in result["pause_checks"])
                           and {"driver", "junction"} <= views and result["control_tests"]["screenshot_saved"])
            (episode / "independent_audit.json").write_text(json.dumps(measured, indent=2)+"\n", encoding="utf-8")
            record.update(audit=measured, native_light_flags_verified=result["native_light_flags_verified"],
                          close_clearance_alerts=result["close_clearance_alerts"],
                          pedestrian_spacing_events=[e for e in events if e["kind"] in
                              ("pedestrian_waiting_for_person", "pedestrian_person_gap_accepted")],
                          completed_overtakes=result["completed_overtakes"], control_tests=result["control_tests"],
                          completed_crossings=sorted(finished), all_crossings_completed=crossing <= finished,
                          viewer_controls_verified=controls_ok,
                          playback_actual=result["simulated_time_per_wall_second"])
            record["passed"] = measured["passed"] and result["native_light_flags_verified"] and crossing <= finished and controls_ok
        except Exception as error:
            record.update(passed=False, error=str(error))
        status["passed"] = all(r["passed"] for r in records)
        (output / "report.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: record.get(k) for k in ("case", "seed", "weather", "passed", "error")}), flush=True)
        if "error" in record:
            raise RuntimeError("Inspect the preserved failed run before continuing")
    status["completed"] = True
    (output / "report.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
