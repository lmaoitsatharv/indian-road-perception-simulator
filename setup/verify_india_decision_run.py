"""Independently join decision labels to actor traces, camera frames and HUD captures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setup/carla_india"))
from verify_run import audit


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def actor_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    launcher = load(run / "launcher_result.json")
    directories = sorted(run.glob("episode_*"))
    if not (launcher["status"] == "completed" and launcher["viewer_exit_code"] == 0
            and launcher["owned_job_handles_closed"] and len(directories) == 1):
        raise ValueError("A completed and cleaned-up single episode is required")
    episode = directories[0]
    result, native_audit = load(episode / "result.json"), audit(episode)
    if not (native_audit["passed"] and native_audit["completed"]):
        raise ValueError("The native rollout audit did not pass")
    rows = actor_rows(episode / "actor_trace.csv")
    ego = {int(r["frame"]): r for r in rows if r["name"] == "ego"}
    with (episode / "ego_decisions.jsonl").open(encoding="utf-8") as stream:
        decisions = [json.loads(line) for line in stream if line.strip()]
    if len(decisions) != len(ego) or len({d["source_frame"] for d in decisions}) != len(ego):
        raise ValueError("Decision records do not cover every ego physics step exactly once")
    by_world = {}
    junction_checks = 0
    for decision in decisions:
        row = ego[decision["source_frame"]]
        if not (decision["world_frame"] == decision["source_frame"] + 1
                and abs(decision["sim_s"] - float(row["sim_s"])) < 1e-7
                and decision["controller_reason"] == row["reason"]
                and decision["pass_state"] == row["pass_state"]
                and abs(decision["target_mps"] - float(row["target_mps"])) < 1e-9):
            raise ValueError("An explanation differs from the actual controller trace")
        by_world[decision["world_frame"]] = decision
        passing = decision["passing"]
        if passing["reason"] == "junction_too_close":
            constraint = next(b for b in reversed(passing["blockers"]) if b["reason"] == "junction_too_close")
            if not (abs(constraint["available_m"] - (140 - float(row["route_s"]))) < 1e-7
                    and constraint["available_m"] < constraint["required_m"]
                    and f"{constraint['available_m']:.1f} m" in decision["hud_lines"][1]
                    and f"{constraint['required_m']:.1f} m" in decision["hud_lines"][1]):
                raise ValueError("Junction text is not backed by the actual distance and threshold")
            junction_checks += 1
    cameras = load(episode / "camera_frames.json")["frames"]
    for camera in cameras:
        decision = by_world[camera["frame"]]
        if not (camera["decision_world_frame"] == camera["frame"]
                and camera["decision_source_frame"] == decision["source_frame"]
                and camera["decision_hud_lines"] == decision["hud_lines"]
                and abs(camera["time"] - decision["world_timestamp"]) < 1e-7):
            raise ValueError("Camera labels refer to another world frame")
    hud_files = sorted(episode.glob("decision_*.png"))
    for image in hud_files:
        sidecar = load(image.with_suffix(".json"))
        if sidecar["decision"] != by_world[sidecar["image_frame"]]:
            raise ValueError("HUD screenshot evidence does not match its recorded camera decision")
    reasons = sorted({d["passing"]["reason"] for d in decisions})
    if result["case"] == "mixed" and not {"junction_too_close", "junction_zone", "passing", "returning", "overtake_completed"} <= set(reasons):
        raise ValueError("The mixed scene did not exercise the expected pass explanations")
    if not hud_files or not result["control_tests"].get("decision_screenshot_saved"):
        raise ValueError("The native decision HUD screenshot was not exercised")
    example = min(decisions, key=lambda d: abs(d["sim_s"] - 28.1))
    comparison = None
    if args.baseline:
        baseline_rows = actor_rows(args.baseline / "actor_trace.csv")
        def keyed(records):
            return {(r["name"], round(float(r["sim_s"]) / .05)): r for r in records}
        earlier, current = keyed(baseline_rows), keyed(rows)
        common = set(earlier) & set(current)
        max_position = max(math.hypot(float(current[k]["x"]) - float(earlier[k]["x"]),
                                      float(current[k]["y"]) - float(earlier[k]["y"])) for k in common)
        command_errors = {field: max(abs(float(current[k][field]) - float(earlier[k][field]))
                                    for k in common if current[k]["type"] == "vehicle")
                          for field in ("target_mps", "throttle", "brake", "steer")}
        reason_changes = sum(current[k]["reason"] != earlier[k]["reason"]
                             or current[k]["pass_state"] != earlier[k]["pass_state"] for k in common)
        earlier_result = load(args.baseline / "result.json")
        comparison = {"matched_actor_steps": len(common), "max_position_difference_m": max_position,
                      "max_command_differences": command_errors, "reason_or_pass_state_changes": reason_changes,
                      "baseline_goal_time_sim_s": earlier_result["ego_goal_time_s"],
                      "goal_time_difference_s": result["ego_goal_time_s"] - earlier_result["ego_goal_time_s"],
                      "baseline_overtakes": earlier_result["completed_overtakes"]}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run / "source").glob("*.py")}
    summary = {"passed": True, "run": str(run), "case": result["case"], "seed": result["seed"],
        "ego_decision_steps": len(decisions), "camera_decision_links": len(cameras),
        "verified_junction_distance_records": junction_checks, "saved_hud_captures": len(hud_files),
        "passing_reasons_observed": reasons, "example_28_1s": {k: example[k] for k in ("sim_s", "controller_reason", "passing", "hud_lines")},
        "goal_reached": result["ego_goal_reached"], "goal_time_sim_s": result["ego_goal_time_s"],
        "completed_overtakes": result["completed_overtakes"], "native_collisions": len(result["collision_events"]),
        "sampled_overlaps": len(result["all_actor_overlaps"]), "owned_jobs_closed": True,
        "baseline_comparison": comparison, "source_sha256": source_hashes,
        "limits": "Read-only explanations of simulator-ground-truth rules; no perception control or learned policy"}
    (run / "decision_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
