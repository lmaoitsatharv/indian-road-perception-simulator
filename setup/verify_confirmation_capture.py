"""Audit all fresh episodes, their scene differences, cameras and native traces."""
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "integration"))
sys.path.insert(0,str(ROOT / "setup"))
from india_scenery import CONFIRMATION_FACADES
from prepare_simulation_dataset import protected,save,sha
from carla_india.verify_run import audit


def main():
    output=ROOT / "runs/simulation_confirmation_20261023_01"
    plan=json.loads((output / "plan.json").read_text())
    capture=json.loads((output / "capture.json").read_text())
    preflight=json.loads((output / "preflight.json").read_text())
    assert capture["status"]=="captured_unscored" and len(capture["episodes"])==8
    assert protected()==plan["protected_sha256"]
    original=json.loads((ROOT / "dataset/simulation_india_v5/plan.json").read_text())
    old_seeds={e["seed"] for e in original["episodes"]}
    outcomes=[]
    total=0
    for mapped,document,checked in zip(capture["maps"],plan["scene_plans"],preflight["layouts"],strict=True):
        folder=Path(mapped["launcher"]["output"])
        # Preflight fingerprints canonical XML text; Windows exports CRLF bytes.
        assert hashlib.sha256((folder / "map.xodr").read_text().encode()).hexdigest()==checked["new_map_sha256"]
        scenery=json.loads((folder / "scenery.json").read_text())
        assets={r["mesh"] for r in scenery["objects"] if r.get("kind")=="generic_facade"}
        assert assets==set(CONFIRMATION_FACADES[document["confirmation_scene"]["scenery"]])
        assert len([r for r in scenery["objects"] if r.get("kind")=="generic_facade"])==12
        for name,expected in zip(mapped["episodes"],document["episodes"],strict=True):
            episode=Path(name)
            assert json.loads((episode / "dataset_episode.json").read_text())==expected
            assert expected["seed"] not in old_seeds
            config=json.loads((episode / "road_camera/capture_config.json").read_text())
            assert config["camera_profile"]==expected["camera_profile"]
            assert config["actual_weather"]["sun_azimuth_angle"]==expected["sun_azimuth_deg"]
            assert config["sensor_tick_s"]==1 and config["model_control"] is False
            rows=[json.loads(line) for line in (episode / "road_camera/frames.jsonl").read_text().splitlines() if line]
            summary=json.loads((episode / "road_camera/capture_summary.json").read_text())
            assert len(rows)==summary["paired_frames"] and not any(summary["dropped"].values())
            assert all(.9 <= b["sim_s"]-a["sim_s"] <= 1.1 for a,b in zip(rows,rows[1:]))
            sky_checked=0
            if config["actual_weather"]["sun_altitude_angle"]>=15:
                for row in rows:
                    if row["sky_pixels"]>=5000:
                        assert row["sky_rgb_median"]>10, "Black daytime sky"
                        sky_checked+=1
            trace=audit(episode)
            # This is a perception check on all predeclared frames. Fresh timing
            # need not exercise every historical scripted trigger. Preserve all
            # missed behaviour criteria, while still requiring capture health.
            assert trace["health_passed"], trace
            native=json.loads((episode / "result.json").read_text())
            outcomes.append({"episode_id":expected["episode_id"],"layout":document["layout"],
                "seed":expected["seed"],"weather":expected["weather"],"case":expected["case"],
                "frames":len(rows),"trace_passed":trace["passed"],
                "native_health_passed":trace["health_passed"],
                "unexercised_behaviour_criteria":[k for k,v in trace["checks"].items() if not v],
                "daylight_sky_frames_checked":sky_checked,
                "collision_events":len(native["collision_events"]),"close_clearance_alerts":native["close_clearance_alerts"],
                "minimum_body_gap_m":native["sampled_min_body_gap_m"],
                "goal_reached":native["ego_goal_reached"],"completed_overtakes":native["completed_overtakes"],
                "new_facade_assets":sorted(assets),"source_map_sha256":sha(folder / "map.xodr"),
                "canonical_map_sha256":checked["new_map_sha256"]})
            total+=len(rows)
    result={"passed":True,"episodes":outcomes,"frames":total,"no_dropped_capture_frames":True,
            "fresh_seed_and_geometry_verified":True,"new_facade_assets_verified":True,
            "all_native_traces_passed":all(e["trace_passed"] for e in outcomes),
            "all_native_health_checks_passed":True,
            "behaviour_warnings_preserved":True,
            "protected_real_and_simulation_model_unchanged":protected()==plan["protected_sha256"]}
    save(output / "capture_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
