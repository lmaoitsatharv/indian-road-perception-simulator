"""Own one bounded camera-only driving session in the connected Indian town.

Writes the two model manifests and a per-run session token, starts the road
worker and the detector worker in WSL (GPU), waits until both are ready, runs
the connected town with --camera-drive, then stops the workers and confirms the
protected model files and the rule driver are unchanged. Everything is kept in a
fresh runs/camera_driving_* folder; the town's own evidence goes to its usual
runs/carla_indian_worlds_* folder.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.prepare_simulation_dataset import save, sha, protected
from setup.carla.session_locks import clear_if_stale
from project_runtime import wsl_distribution, wsl_python

ROAD_CHECKPOINT = "runs/simulation_frozen_20261022_01/best.keras"
ROAD_SHA = "8d8c9fd7a3880bb69643c4973db8f14c46242c178fae6927d06b5f9d6aaaf8af"
CONFIRMATION = "runs/simulation_confirmation_20261023_01/results.json"
DETECTOR = "runs/detection_train_20261025_02/weights/best.pt"
DETECTOR_SHA = "f7bb7e4c5959520d797dcde5547fdcded9ece1e9651a3e7bd6ffa2273749182c"
DISTRO = wsl_distribution()
# The two model environments under the WSL user's home (setup/rebuild_wsl_envs.sh builds them), or set
# PROJECT_WSL_ROAD_PYTHON / PROJECT_WSL_DETECTOR_PYTHON to absolute Linux interpreters.
VENVS = {"road": ("idd-tf220", "PROJECT_WSL_ROAD_PYTHON"), "detector": ("idd-detect", "PROJECT_WSL_DETECTOR_PYTHON")}
PYTHON = {}


def wsl_interpreter(role):
    if role not in PYTHON:
        venv, variable = VENVS[role]
        PYTHON[role] = wsl_python(DISTRO, os.environ.get(variable), venv=venv)
    return PYTHON[role]
SCRIPT = {"road": "setup/camera_driving_road_worker.py", "detector": "setup/camera_driving_detector_worker.py"}
PORTS = {"road": 29161, "detector": 29162}


def linux(path):
    path = Path(path).resolve().as_posix()
    return "/mnt/" + path[0].lower() + "/" + path[3:]


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def check_workers(session, output, count=40):
    """Send saved town frames through the real client and workers; compare with the 6 Nov probe."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / "integration"))
    from india_perception_client import PerceptionClient
    probe = ROOT / "runs/camera_driving_probe_20261106_01"
    meta = json.loads((probe / "truth.json").read_text())
    run_folder = ROOT / meta["run"]
    client = PerceptionClient(session, output / "worker_check_errors.jsonl")
    rows, agreement = [], []
    try:
        for row in meta["frames"][:count]:
            rgb = np.array(Image.open(run_folder / row["rgb"]).convert("RGB"))
            client.submit(row["frame"], rgb, row["sim_s"])
            answer = client.result(row["frame"], timeout=5.)
            rows.append({"frame": row["frame"], "ok": answer.ok, "error": answer.error, **answer.timings,
                         "boxes": None if answer.boxes is None else len(answer.boxes)})
            if answer.ok:
                reference = np.array(Image.open(probe / "road_masks" / f"mask_{row['frame']:08d}.png"))
                agreement.append(float((reference == answer.mask).mean()))
    finally:
        client.close()
    ok = [r for r in rows if r["ok"]]
    summary = {"frames": len(rows), "answered": len(ok), "mask_agreement_with_probe_min": min(agreement) if agreement else None,
               "roundtrip_ms_median": float(np.median([max(r["road_roundtrip_ms"], r["detector_roundtrip_ms"]) for r in ok])) if ok else None,
               "road_ms_median": float(np.median([r["road_ms"] for r in ok])) if ok else None,
               "detector_ms_median": float(np.median([r["detector_ms"] for r in ok])) if ok else None,
               "transport": client.summary()}
    summary["passed"] = bool(len(ok) == len(rows) and agreement and min(agreement) > .995)
    save(output / "worker_check.json", {"summary": summary, "frames": rows})
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def run(output, *, sim_seconds, seed, weather, quality, auto_reset, check, worker_check=False, destination=None, destinations=None,
        rl_speed=False, keep_gearbox=False):
    if sys.platform != "win32":
        raise RuntimeError("Run this owner on Windows; it starts the WSL workers and the CARLA session")
    output = output.resolve()
    if output.parent != ROOT / "runs" or not output.name.startswith("camera_driving_"):
        raise ValueError("Use a fresh runs/camera_driving_* folder")
    # Locks left by a run that was closed or crashed are cleared; a live session still refuses.
    for lock in ("setup/carla/session.lock", "setup/carla/viewer.lock"):
        clear_if_stale(ROOT / lock)
        if (ROOT / lock).exists():
            raise RuntimeError(f"{lock} belongs to a session that is still running; close it first.")
    if (ROOT / "setup/carla/STOP_DEMO.json").exists():
        raise RuntimeError("setup/carla/STOP_DEMO.json exists; remove it intentionally before starting.")
    output.mkdir(exist_ok=False)
    before = protected()
    rule_driver = {name: sha(ROOT / "integration" / name) for name in ("india_behavior.py",)}
    if sha(ROOT / ROAD_CHECKPOINT) != ROAD_SHA or sha(ROOT / DETECTOR) != DETECTOR_SHA:
        raise RuntimeError("A model checkpoint differs from the approved camera-driving models")
    lock = json.loads((ROOT / "models/segmentation_final_lock.json").read_text())
    catalog = json.loads((ROOT / "models/perception_models.json").read_text())
    detector_entry = next(m for m in catalog["models"] if m["name"] == "idd_detector_latest")
    if detector_entry["sha256"] != DETECTOR_SHA:
        raise RuntimeError("The catalogued detector is not the approved checkpoint")
    road_manifest = {"schema": "simulator_camera_driving_v1", "scope": "native_carla_camera_driving",
        "training_allowed": False, "control_allowed": True, "shared_real_deployment_approved": False,
        "input_shape": [512, 512, 3], "normalization": "RGB /127.5 -1 exactly once", "threshold": .5,
        "checkpoint": ROAD_CHECKPOINT, "checkpoint_sha256": ROAD_SHA,
        "confirmation": CONFIRMATION, "confirmation_sha256": sha(ROOT / CONFIRMATION),
        "base_checkpoint_sha256": lock["checkpoint_sha256"]}
    detector_manifest = {"schema": "carla_camera_driving_detector_v1", "scope": "native_carla_camera_driving",
        "training_allowed": False, "control_allowed": True, "checkpoint": DETECTOR, "checkpoint_sha256": DETECTOR_SHA,
        "classes": detector_entry["classes"],
        "inference": {"imgsz": 1280, "conf": .25, "iou": .6, "half": True, "max_det": 100},
        "setting_note": "1280 / NMS 0.6 chosen for town frames (probe 6 Nov); the recorded IDD setting 1536 / 0.6 is unchanged"}
    save(output / "road_manifest.json", road_manifest)
    save(output / "detector_manifest.json", detector_manifest)
    session = {"schema": "camera_driving_session_v1", "session": secrets.token_hex(16), "token": secrets.token_hex(32),
        "model_control": True, "ground_truth_to_driver": False, "safety_backstop": False,
        "road": {"port": PORTS["road"], "manifest": relative(output / "road_manifest.json"), "folder": relative(output) + "/road_worker"},
        "detector": {"port": PORTS["detector"], "manifest": relative(output / "detector_manifest.json"), "folder": relative(output) + "/detector_worker"},
        "stop_file": relative(output) + "/stop_workers.json", "service_max_seconds": 2400,
        "camera": {"width": 960, "height": 540, "fov": 95., "mount": [1.35, .25, 1.45], "pitch": 0., "hz": 10,
                   # Rear camera, around passes and turning round: 0.1 m behind the rear bumper, 1.3 m up, looking
                   # back and 20 degrees down, so it sees the ground from 1 m behind (level, 2.1 m: a stall nearer
                   # than that was cut off at the picture's bottom and could not be placed, 18 Nov run).
                   "rear": {"z": 1.3, "fov": 95., "pitch": -20.},
                   # Side cameras at the door pillars (the car's middle, just outside its sides), 1.3 m up, looking
                   # straight out and 15 degrees down: with the front and rear ones they see all round. The right one
                   # runs around passes (the passing lane beside the car), both while turning round (the street then
                   # runs past the car's sides). Each frame sends one of the rear and side pictures, in turn.
                   "left": {"z": 1.3, "fov": 95., "pitch": -15.}, "right": {"z": 1.3, "fov": 95., "pitch": -15.}},
        "latency_ticks": 2, "result_timeout_s": 1., "unavailable_limit_s": 10.,
        "auto_reset_s": 30. if auto_reset else None,
        "driver": {"cruise": 6., "wide_area": 3.5, "lost_road": 2.5, "keep_left_m": 1.5}}
    # The camera car's drivetrain: one fixed ratio, no gear shifts (12 Nov; --keep-gearbox for the stock gearbox)
    session["vehicle"] = {"single_gear": not keep_gearbox, "top_speed_m_s": 10.}
    if rl_speed:
        # The learned speed governor (docs/RL_SPEED_DESIGN_20261112.md): record exactly which policy drove.
        policy = ROOT / "models/rl_speed/policy.npz"
        if not policy.exists():
            raise RuntimeError("models/rl_speed/policy.npz is missing; train one with rl/train_speed_ppo.py")
        session["driver"]["rl_speed"] = 1.
        session["rl_speed_policy"] = {"path": relative(policy), "sha256": sha(policy)}
    save(output / "session.json", session)
    state = {"status": "starting_workers", "sim_seconds": sim_seconds, "seed": seed, "weather": weather, "check": check,
             "rl_speed": session.get("rl_speed_policy"), "vehicle": session["vehicle"],
             "protected_sha256": before, "rule_driver_sha256": rule_driver}
    save(output / "status.json", state)
    workers, logs = {}, {}
    try:
        for role, port in PORTS.items():
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))   # refuse an occupied port rather than talk to another process
        for role in PORTS:
            argv = ["wsl.exe", "-d", DISTRO, "--cd", linux(ROOT), "--exec", "/usr/bin/env", "TF_CPP_MIN_LOG_LEVEL=2",
                    "PYTHONUNBUFFERED=1", "CUBLAS_WORKSPACE_CONFIG=:4096:8", wsl_interpreter(role), "-u", SCRIPT[role],
                    "--session", linux(output / "session.json")]
            logs[role] = (output / f"{role}_worker.log").open("w")
            workers[role] = subprocess.Popen(argv, cwd=ROOT, stdout=logs[role], stderr=subprocess.STDOUT,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
        print("Loading the road model and the detector on the GPU...", flush=True)
        deadline = time.monotonic() + 240
        while not all((output / f"{role}_worker" / "worker_ready.json").exists() for role in PORTS):
            for role, process in workers.items():
                if process.poll() is not None:
                    raise RuntimeError(f"{role} worker exited during startup; see {role}_worker.log")
            if time.monotonic() > deadline:
                raise RuntimeError("GPU workers did not become ready in 4 minutes")
            time.sleep(.5)
        if worker_check:
            state.update(status="worker_check", worker_check=check_workers(session, output))
            state["status"] = "completed" if state["worker_check"]["passed"] else "worker_check_failed"
            return
        state["status"] = "driving"
        save(output / "status.json", state)
        argv = [str(ROOT / ".venv-carla/Scripts/python.exe"), "-u", "setup/carla/run_session.py", "--india-world", "combined",
                "--seed", str(seed), "--sim-seconds", str(sim_seconds), "--seconds", "1800", "--weather", weather,
                "--quality", quality, "--camera-drive", str(output / "session.json")]
        if destination:
            argv.append(f"--destination={destination}")   # "=": a negative x is not an option
        if destinations:
            argv.append(f"--destinations={destinations}")
        if check:
            argv.append("--smoke")
        print("Building the town. Your car is driven by the camera models; ESC closes.", flush=True)
        with (output / "town_session.log").open("w") as log:
            town = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        owner = json.loads((ROOT / "setup/carla/server_owner.json").read_text())
        state.update(town_exit_code=town.returncode, town_output=owner.get("output"), town_argv=argv)
        result_file = Path(owner.get("output", "")) / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            drive = result.get("camera_driving") or {}
            state["summary"] = {k: drive.get(k) for k in ("distance_m", "districts_visited", "collision_count", "off_road_share",
                                                            "wrong_side_share", "reset_count", "perception_results", "perception_unavailable")}
            state["summary"].update(passed=result.get("passed"), sim_seconds=result.get("sim_seconds"), exit_reason=result.get("exit_reason"),
                                    errors=len(result.get("errors", [])))
        state["status"] = "completed" if town.returncode == 0 else "town_session_needs_inspection"
    except BaseException as error:
        state.update(status="failed_preserved", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        save(ROOT / session["stop_file"], {"session": session["session"], "stop": True})
        for role, process in workers.items():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()   # this owner started this exact Windows host process
                process.wait(timeout=5)
                state.setdefault("worker_problems", []).append(f"{role} worker did not stop on its marker")
            state[f"{role}_worker_exit_code"] = process.returncode
            status_file = output / f"{role}_worker" / "worker_status.json"
            if status_file.exists():
                worker = json.loads(status_file.read_text())
                state[f"{role}_worker"] = {k: worker.get(k) for k in ("status", "requests", "rejected", "connections", "model_weights_unchanged")}
        for handle in logs.values():
            handle.close()
        unchanged = protected() == before and all(sha(ROOT / "integration" / n) == h for n, h in rule_driver.items()) \
            and sha(ROOT / ROAD_CHECKPOINT) == ROAD_SHA and sha(ROOT / DETECTOR) == DETECTOR_SHA
        state["protected_files_unchanged"] = unchanged
        if not unchanged:
            state.update(status="failed_preserved", error="A protected model or the rule driver changed")
        save(output / "status.json", state)
        print(json.dumps({k: state.get(k) for k in ("status", "town_output", "summary", "protected_files_unchanged")}, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-seconds", type=float, default=600)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--weather", choices=("clear", "overcast", "wet", "dusk"), default="clear")
    parser.add_argument("--quality", choices=("Low", "Epic"), default="Low")
    parser.add_argument("--no-reset", action="store_true", help="Never lift a stuck car back onto its lane")
    parser.add_argument("--check", action="store_true", help="Run unthrottled (as fast as the PC allows) for testing")
    parser.add_argument("--worker-check", action="store_true", help="Start only the GPU workers and push saved town frames through them")
    parser.add_argument("--destination", help="Drive to this place (a name, or x,y); pick others on the map with M")
    parser.add_argument("--destinations", help="Test runs: places separated by ';', the next set 5 s after arriving at the last")
    parser.add_argument("--rl-speed", action="store_true", help="Let the learned speed governor pick the speed (models/rl_speed/policy.npz)")
    parser.add_argument("--keep-gearbox", action="store_true", help="Keep the car's stock automatic gearbox (default: one fixed ratio)")
    args = parser.parse_args()
    if not 10 <= args.sim_seconds <= 1500:
        parser.error("Choose 10..1500 simulated seconds")
    stamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%dT%H%M%S")
    run(args.output or ROOT / "runs" / ("camera_driving_" + stamp), sim_seconds=args.sim_seconds, seed=args.seed,
        weather=args.weather, quality=args.quality, auto_reset=not args.no_reset, check=args.check, worker_check=args.worker_check,
        destination=args.destination, destinations=args.destinations, rl_speed=args.rl_speed, keep_gearbox=args.keep_gearbox)
