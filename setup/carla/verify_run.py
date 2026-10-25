"""Independently recount recorded camera, motion and actor evidence."""
import argparse
import csv
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run
result = json.loads((run / "result.json").read_text(encoding="utf-8"))
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
launcher = json.loads((run / "launcher_result.json").read_text(encoding="utf-8"))
camera = json.loads((run / "camera_frames.json").read_text(encoding="utf-8"))["frames"]
with (run / "ego_trace.csv").open(newline="", encoding="utf-8") as stream:
    ego = list(csv.DictReader(stream))
with (run / "actor_trace.csv").open(newline="", encoding="utf-8") as stream:
    actors = list(csv.DictReader(stream))
assert launcher["viewer_exit_code"] == 0 and launcher["status"] == "completed"
assert result["cleanup_complete"] and not result["cleanup_warnings"] and not result["errors"]
assert len(ego) == len(camera) == result["frames"] == result["camera_images"]
frames = [int(row["frame"]) for row in ego]
assert len(set(frames)) == len(frames) and frames == sorted(frames)
assert frames == [row["image_frame"] for row in camera]
assert all(row["image_frame"] == row["world_frame"] <= row["latest_world_frame"] for row in camera)
assert all(abs(row["image_timestamp"] - row["snapshot_timestamp"]) < 1e-5 for row in camera)
assert all(abs(float(a["sim_seconds"]) - b["image_timestamp"]) < 1e-5 for a, b in zip(ego, camera))
distance = sum(math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])) for a, b in zip(ego, ego[1:]))
assert abs(distance - result["ego_distance_m"]) < 1e-6
groups = {}
for row in actors:
    groups.setdefault(int(row["actor_id"]), []).append(row)
recounted = []
for stats in result["actor_stats"]:
    rows = groups[stats["id"]]
    steps = [(math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])),
              float(b["sim_seconds"]) - float(a["sim_seconds"])) for a, b in zip(rows, rows[1:])]
    assert all(dt > 0 for _, dt in steps)
    observed_distance = sum(step for step, _ in steps)
    max_speed = max((step / dt for step, dt in steps), default=0)
    assert abs(observed_distance - stats["distance_m"]) < 1e-6
    if result.get("actor_speed_source"):
        assert abs(max_speed - stats["max_speed_mps"]) < 1e-6
    recounted.append({"id": stats["id"], "type": stats["type"], "distance_m": observed_distance,
                      "max_speed_from_positions_mps": max_speed})
assert len(groups) == 1 + config["spawned_npcs"] + config["spawned_walkers"]
assert all(item["world_stayed_still"] and item["paused_at_frame"] == item["resumed_at_frame"] for item in result["pause_checks"])
assert all(result["smoke_key_events"].values()) and result["key_screenshot_saved"]
assert set(result["saved_views"]) == {"driver", "follow"}
assert list(run.glob("driver_*.png")) and list(run.glob("follow_*.png"))
assert not result["collisions"] and result["smoke_passed"]
verification = {"passed": True, "seed": config["seed"], "frames": len(ego), "ego_distance_m": distance,
                "actor_count": len(groups), "actors": recounted,
                "image_snapshot_timestamps_match": True,
                "max_delivery_delay_frames": max(row["latest_world_frame"] - row["image_frame"] for row in camera),
                "zero_vehicle_collision_events": True,
                "legacy_actor_speed_field": not bool(result.get("actor_speed_source")),
                "limits": "Recorded samples and vehicle collision events only; no continuous-time all-actor safety proof"}
(run / "independent_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in verification.items() if key != "actors"}, indent=2))
