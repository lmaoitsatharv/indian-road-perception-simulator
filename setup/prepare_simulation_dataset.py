"""Predeclare and capture a separate simulation dataset. Never trains a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "dataset/simulation_india_v5"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def protected():
    names = ["idd_inference.py", "predict_road.py", "predict_road_video.py", "idd_data.py", "idd_model.py",
             "train_idd.py", "IDD_Preprocessing.ipynb", "IDD_UNet_Model.ipynb", "models/model_config.json",
             "models/segmentation_final_lock.json", "models/retained_models.json", "dataset/splits/pair_splits.csv",
             "runs/frozen_01/best.keras", "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras",
             "runs/boundary_20261003/boundary_weight4/best.keras"]
    return {name: sha(ROOT / name) for name in names}


def plan(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = ["mixed", "clear_overtake", "oncoming", "pedestrian", "return_gap", "junction"]
    weather = ["clear", "overcast", "wet", "dusk"]
    episodes = []
    # Four map groups; six train + one val + one test episode per map.
    # Fewer views per episode avoid filling quotas with stationary near-copies.
    layouts = ["crossroad", "t_north", "t_south", "straight"]
    for wi, layout in enumerate(layouts):
        for j in range(8):
            condition = weather[(j + wi) % 4]
            split = "train" if j < 6 else ("val" if j == 6 else "test")
            case = cases[(j + wi * 2) % len(cases)]
            if layout == "straight" and case == "junction":
                case = "mixed"
            episodes.append({"episode_id": f"{layout}_{j + 1:02}", "layout": layout, "split": split,
                "seed": 1001 + wi * 100 + j * 7, "weather": condition,
                "case": case, "sim_seconds": 80,
                "sun_azimuth_deg": [60,120,180,240,300,150][j%6],
                "camera_profile": {"height_m": [1.25,1.35,1.5][j%3], "pitch_deg": [-2.,0.,1.][j%3],
                    "yaw_deg": [-2.,0.,2.][(wi+j)%3], "fov_deg": [90.,95.,100.][(wi+j)%3]},
                "capture_hz": 5, "pairs_to_select": ([167,167,167,167,166,166][j] if split == "train" else 125)})
    protocol = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned", "training_started": False, "seed": 42, "target_pairs": 5000,
        "split_counts": {"train": 4000, "val": 500, "test": 500}, "episodes": episodes,
        "split_method": "Predeclared whole episodes, disjoint seeds; 80/10/10 by final pair count",
        "camera": "Native 960x540 synchronized RGB/semantic/instance/depth, fixed windscreen mount, FOV 95",
        "label_source": "Native visible semantic classes + exact asphalt instance IDs + depth/OpenDRIVE geometry; no model predictions",
        "labels": {"0": "visible non-road", "1": "visible driving-road surface", "255": "uncertain: exclude from training/evaluation"},
        "layouts": layouts, "limits": ["Four related illustrative left-hand road topologies with shared local assets; not multiple real cities or calibrated traffic",
                   "No authentic autorickshaw asset; installed cars, buses, vans, scooters, bikes and walkers",
                   "Correlated frames within an episode remain; no episode crosses a split",
                   "Geometry-assisted labels have a conservative ignored boundary/range region"],
        "protected_sha256": protected()}
    protocol["atmosphere"] = {"scattering_intensity":1.,"mie_scattering_scale":.03,"rayleigh_scattering_scale":.0331,
        "cloudiness": {"clear":15,"overcast":60,"wet":60,"dusk":30},
        "fog_density": "2 clear, 3 otherwise", "fog_distance_m":.75,"fog_falloff":.1,
        "source":"https://carla.readthedocs.io/en/0.9.16/python_api/#carlaweatherparameters"}
    save(output / "plan.json", protocol)
    print(json.dumps({"status": "planned", "episodes": len(episodes), "target_pairs": 5000}), flush=True)


def capture(output):
    protocol = json.loads((output / "plan.json").read_text())
    if (output / "capture.json").exists():
        raise FileExistsError("Capture already attempted; preserve it and use a new version or explicit recovery plan")
    if protected() != protocol["protected_sha256"]:
        raise RuntimeError("Protected real-model files changed before capture")
    python = ROOT / ".venv-carla/Scripts/python.exe"
    base_argv = [str(python), "-u", "setup/carla/run_session.py", "--india", "--smoke", "--observe-road",
        "--road-camera-hz", "5", "--quality", "Low", "--surface-style", "asphalt", "--speed", "3",
        "--sim-seconds", "80", "--seconds", "1200"]
    state = {"status": "capturing", "started_utc": datetime.now(timezone.utc).isoformat(),
             "training_started": False, "maps": [], "episodes": []}
    save(output / "capture.json", state)
    for layout in protocol["layouts"]:
        entries = [e for e in protocol["episodes"] if e["layout"] == layout]
        subplan = output / f"plan_{layout}.json"
        save(subplan, {"layout": layout, "episodes": entries})
        argv = base_argv + ["--seed", str(entries[0]["seed"]), "--dataset-plan", str(subplan)]
        with (output / f"capture_{layout}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    live_owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text())
                    if str(subplan) in live_owner.get("viewer_argv",[]):
                        for episode in sorted(Path(live_owner["output"]).glob("episode_*")):
                            if not (episode/"dataset_episode.json").exists() or str(episode) in state["episodes"]:
                                continue
                            native=json.loads((episode/"result.json").read_text())
                            if native["runtime_passed"] and native["cleanup_complete"] and not native["cleanup_warnings"]:
                                state["episodes"].append(str(episode))
                                save(output/"capture.json",state)
                except (OSError,json.JSONDecodeError):
                    pass
                time.sleep(2)
            exit_code=process.returncode
        owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text())
        folder = Path(owner["output"])
        captures = sorted(folder.glob("episode_*"))
        state["episodes"] += [str(p) for p in captures if str(p) not in state["episodes"]]
        record = {"layout": layout, "exit_code": exit_code, "launcher": owner, "argv": argv,
                  "episodes": len(captures)}
        state["maps"].append(record)
        save(output / "capture.json", state)
        print(json.dumps({"layout": layout, "exit_code": exit_code, "episodes": len(captures)}), flush=True)
        if exit_code or not owner.get("owned_job_handles_closed") or len(captures) != len(entries):
            state["status"] = "failed_preserved"
            save(output / "capture.json", state)
            return 1
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["protected_unchanged"] = protected() == protocol["protected_sha256"]
    state["status"] = "captured" if state["protected_unchanged"] and len(state["episodes"]) == len(protocol["episodes"]) else "failed_preserved"
    save(output / "capture.json", state)
    print(json.dumps({"status": state["status"], "episodes": len(state["episodes"])}), flush=True)
    return 0 if state["status"] == "captured" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "capture"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "dataset"):
        raise ValueError("Use a dedicated folder under dataset/")
    return plan(output) if args.stage == "plan" else capture(output)


if __name__ == "__main__":
    raise SystemExit(main())
