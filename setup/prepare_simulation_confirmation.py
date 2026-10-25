"""Predeclare, audit and capture fresh confirmation scenes; never trains."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setup"))
sys.path.insert(0, str(ROOT / "integration"))
from prepare_simulation_dataset import save, sha, protected

DEFAULT = ROOT / "runs/simulation_confirmation_20261023_01"
TRAIN = ROOT / "runs/simulation_frozen_20261022_01"


def utc():
    return datetime.now(timezone.utc).isoformat()


def plan(output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((TRAIN / "run_config.json").read_text())
    checkpoint = TRAIN / "best.keras"
    if config["status"] != "completed" or sha(checkpoint) != config["best_checkpoint_sha256"]:
        raise RuntimeError("Use the completed validation-selected model")
    history = json.loads((TRAIN / "history.json").read_text())
    selected = max(range(len(history["val_road_iou"])), key=lambda i: history["val_road_iou"][i]) + 1
    assert selected == config["best_epoch"] == 10
    scenes = [
        ("crossroad",10.,"lowrise","worn_warm",[("mixed","clear",4101),("oncoming","wet",4109)]),
        ("t_north",11.,"apartments","cool_aggregate",[("pedestrian","overcast",4201),("clear_overtake","dusk",4209)]),
        ("t_south",10.,"lowrise","cool_aggregate",[("junction","clear",4301),("return_gap","overcast",4309)]),
        ("straight",11.,"apartments","worn_warm",[("mixed","wet",4401),("pedestrian","dusk",4409)]),
    ]
    training_plan = json.loads((ROOT / "dataset/simulation_india_v5/plan.json").read_text())
    previous_seeds = {e["seed"] for e in training_plan["episodes"]} | {7,42,137}
    documents = []
    for layout,radius,scenery,palette,episodes in scenes:
        entries = []
        for i,(case,weather,seed) in enumerate(episodes):
            assert seed not in previous_seeds
            entries.append({"episode_id": f"fresh_{layout}_{i+1:02}", "layout": layout,
                "split": "test", "case": case, "weather": weather, "seed": seed, "sim_seconds": 60,
                "sun_azimuth_deg": 85. if i == 0 else 265., "capture_hz": 1,
                "camera_profile": {"height_m": 1.30 if i == 0 else 1.45,
                    "pitch_deg": -1. if i == 0 else .5, "yaw_deg": -1. if i == 0 else 1.,
                    "fov_deg": 92. if i == 0 else 98.}})
        document = {"role": "fresh_confirmation_only", "layout": layout,
                    "confirmation_scene": {"junction_radius_m": radius, "scenery": scenery,
                                            "asphalt_palette": palette}, "episodes": entries}
        save(output / f"plan_{layout}.json", document)
        documents.append(document)
    selection = {"frozen_utc": utc(), "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha(checkpoint), "selected_epoch": selected,
        "criterion": "Maximum simulation validation road IoU; no fresh confirmation pixels seen",
        "validation_road_iou": config["best_validation_road_iou"], "threshold": .5,
        "scope": "recorded_carla_only", "training_allowed": False,
        "shared_real_deployment_approved": False,
        "existing_simulation_test_already_inspected": True,
        "real_baseline_comparison": "Same-capture comparison will follow this fixed selection; not a shared-deployment approval"}
    save(output / "selection.json", selection)
    lock = json.loads((ROOT / "models/segmentation_final_lock.json").read_text())
    manifest = {"schema": "simulator_road_model_v1", "scope": "recorded_carla_only",
        "training_allowed": False, "input_shape": [512,512,3],
        "normalization": "RGB /127.5 -1 exactly once", "threshold": .5,
        "shared_real_deployment_approved": False, "base_checkpoint_sha256": lock["checkpoint_sha256"],
        "checkpoint": selection["checkpoint"], "checkpoint_sha256": selection["checkpoint_sha256"],
        "selection_artifact": (output / "selection.json").relative_to(ROOT).as_posix(),
        "selection_artifact_sha256": sha(output / "selection.json")}
    save(output / "simulation_model.json", manifest)
    protocol = {"created_utc": utc(), "status": "planned", "training_allowed": False,
        "model_control": False, "selection_sha256": sha(output / "selection.json"),
        "model_manifest_sha256": sha(output / "simulation_model.json"),
        "reference": "Same native semantic/instance/depth/map definition and declared edge exclusion as dataset v5",
        "episodes": 8, "target_frames": 480, "capture_hz": 1, "scene_plans": documents,
        "sampling": "All frames of all predeclared episodes; no accuracy-based frame or episode selection",
        "protected_sha256": protected(), "simulation_checkpoint_sha256": sha(checkpoint),
        "simulation_manifest_sha256": sha(ROOT / "dataset/simulation_india_v5/splits/pair_splits.csv"),
        "limits": ["Same illustrative LHT road families; changed junction spacing, installed facades and asphalt palette",
                   "Related procedural maps, not actual Indian city maps; generic assets remain uncalibrated",
                   "Recorded replay comparison; not end-to-end live camera latency or vehicle control",
                   "Fresh seeds are disjoint from the 32 training/validation/test episodes; frozen checkpoint and threshold",
                   "No training, model reselection or tuning on these new scores"]}
    save(output / "plan.json", protocol)
    print(json.dumps({"status": "planned", "episodes":8,"target_frames":480,"selected_epoch":selected}))


def preflight(output):
    import carla
    from india_road import junction_xml, LAYOUT_ARMS
    from india_dataset_maps import validate_layout
    from india_streetscape import paving_layout, footprint_samples
    from india_scenery import CONFIRMATION_FACADES
    protocol = json.loads((output / "plan.json").read_text())
    original = json.loads((ROOT / "dataset/simulation_india_v5/map_preflight.json").read_text())
    historical = {r["layout"]:r["map_sha256"] for r in original["layouts"]}
    import hashlib
    results = []
    for document in protocol["scene_plans"]:
        layout, scene = document["layout"], document["confirmation_scene"]
        assert hashlib.sha256(junction_xml(layout).encode()).hexdigest() == historical[layout]
        xml = junction_xml(layout, junction_radius=scene["junction_radius_m"])
        checked = validate_layout(xml,layout)
        road_map = carla.Map("fresh_"+layout,xml)
        count = 0
        for tile in paving_layout(LAYOUT_ARMS[layout]):
            for x,y in footprint_samples(tile):
                assert road_map.get_waypoint(carla.Location(x=x,y=y,z=0),False,carla.LaneType.Driving) is None, (layout,tile,x,y)
                count += 1
        digest = hashlib.sha256(xml.encode()).hexdigest()
        assert digest not in set(historical.values())
        for asset in CONFIRMATION_FACADES[scene["scenery"]]:
            path = ROOT / "tools/carla/0.9.16/CarlaUE4/Content" / (asset.removeprefix("/Game/").split(".")[0] + ".uasset")
            assert path.is_file(), path
        results.append({**checked,"footpath_samples_checked":count,"driving_lane_intrusions":0,
                        "new_map_sha256":digest,"training_map_sha256":historical[layout],"scene":scene})
    save(output / "preflight.json", {"passed":True,"layouts":results,"default_map_hashes_unchanged":True,
                                     "all_confirmation_maps_differ_from_training":True})
    print(json.dumps({"passed":True,"layouts":results},indent=2))


def capture(output):
    protocol = json.loads((output / "plan.json").read_text())
    if not json.loads((output / "preflight.json").read_text())["passed"]:
        raise RuntimeError("Preflight must pass before native capture")
    if (output / "capture.json").exists():
        raise FileExistsError("Preserve prior capture attempts")
    assert protected() == protocol["protected_sha256"]
    state = {"status":"capturing","started_utc":utc(),"maps":[],"episodes":[]}
    save(output / "capture.json",state)
    try:
        for document in protocol["scene_plans"]:
            layout = document["layout"]
            plan_file = output / f"plan_{layout}.json"
            argv = [sys.executable,"-u","setup/carla/run_session.py","--india","--smoke","--observe-road",
                    "--road-camera-hz","1","--quality","Low","--surface-style","asphalt",
                    "--speed","3","--sim-seconds","60","--seconds","600",
                    "--seed",str(document["episodes"][0]["seed"]),"--dataset-plan",str(plan_file)]
            with (output / f"capture_{layout}.log").open("w",encoding="utf-8") as log:
                process = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=720)
            owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text())
            folder = Path(owner["output"])
            episodes = sorted(folder.glob("episode_*"))
            record = {"layout":layout,"exit_code":process.returncode,"launcher":owner,
                      "episodes":[str(p) for p in episodes],"argv":argv}
            state["maps"].append(record)
            state["episodes"].extend(str(p) for p in episodes)
            save(output / "capture.json",state)
            if (process.returncode or str(plan_file) not in owner.get("viewer_argv",[])
                    or not owner.get("owned_job_handles_closed") or len(episodes) != 2):
                raise RuntimeError(f"Fresh capture did not finish cleanly: {layout}; inspect preserved evidence")
            for episode in episodes:
                native = json.loads((episode / "result.json").read_text())
                summary = json.loads((episode / "road_camera/capture_summary.json").read_text())
                assert native["runtime_passed"] and native["cleanup_complete"] and not native["cleanup_warnings"]
                assert not any(summary["dropped"].values()) and 59 <= summary["paired_frames"] <= 61
            print(json.dumps({"layout":layout,"status":"captured","episodes":len(episodes)}),flush=True)
        state.update(status="captured_unscored",finished_utc=utc(),protected_unchanged=protected()==protocol["protected_sha256"])
        assert state["protected_unchanged"]
    except BaseException as error:
        state.update(status="failed_preserved",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "capture.json",state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","preflight","capture"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.parent != ROOT / "runs" or not output.name.startswith("simulation_confirmation_"):
        raise ValueError("Use a new runs/simulation_confirmation_* directory")
    {"plan":plan,"preflight":preflight,"capture":capture}[args.stage](output)
