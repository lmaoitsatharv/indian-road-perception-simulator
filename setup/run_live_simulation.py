"""Own one bounded live preview plus optional matched rule-control/fault checks."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from setup.prepare_simulation_dataset import save,sha,protected
from idd_live_simulation_inference import load_live_spec
from project_runtime import linux_path, wsl_distribution, wsl_python


def linux(path):
    return linux_path(Path(path).resolve())


def native_run(output,plan,seed,seconds,config=None):
    argv=[str(ROOT / ".venv-carla/Scripts/python.exe"),"-u","setup/carla/run_session.py","--india","--smoke","--observe-road",
          "--road-camera-hz","5","--quality","Low","--surface-style","asphalt","--speed","1",
          "--seconds",str(max(240,int(seconds)+180)),"--sim-seconds",str(seconds),"--seed",str(seed),
          "--dataset-plan",str(plan)]
    if config:
        argv += ["--live-road-config",str(config)]
    label="live" if config else "baseline"
    with (output / (label+"_capture.log")).open("w") as log:
        process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(360,seconds+240))
    owner=json.loads((ROOT / "setup/carla/server_owner.json").read_text())
    record={"exit_code":process.returncode,"launcher":owner,"argv":argv}
    save(output / (label+"_capture.json"),record)
    if process.returncode or not owner.get("owned_job_handles_closed") or str(plan) not in owner.get("viewer_argv",[]):
        raise RuntimeError(label+" native run failed; inspect preserved capture/viewer logs")
    return record


def run(output,*,check,seconds,seed,case,weather,reuse_baseline=None):
    if sys.platform!="win32":
        raise RuntimeError("Run this owner on Windows; it selects the dedicated CARLA and GPU runtimes")
    output=output.resolve()
    if output.parent!=ROOT / "runs" or not output.name.startswith("simulation_live_"):
        raise ValueError("Use a fresh runs/simulation_live_* directory")
    output.mkdir(parents=True,exist_ok=False)
    before=protected()
    checkpoint=ROOT / "runs/simulation_frozen_20261022_01/best.keras"
    confirmation=ROOT / "runs/simulation_confirmation_20261023_01/results.json"
    confirmed=json.loads(confirmation.read_text())
    assert sha(checkpoint)==confirmed["checkpoint_sha256"]
    lock=json.loads((ROOT / "models/segmentation_final_lock.json").read_text())
    manifest={"schema":"simulator_live_observer_v1","scope":"native_carla_observation_only",
        "training_allowed":False,"control_allowed":False,"shared_real_deployment_approved":False,
        "input_shape":[512,512,3],"normalization":"RGB /127.5 -1 exactly once","threshold":.5,
        "checkpoint":checkpoint.relative_to(ROOT).as_posix(),"checkpoint_sha256":sha(checkpoint),
        "confirmation":confirmation.relative_to(ROOT).as_posix(),"confirmation_sha256":sha(confirmation),
        "base_checkpoint_sha256":lock["checkpoint_sha256"]}
    save(output / "model_manifest.json",manifest)
    load_live_spec(output / "model_manifest.json")
    config={"session":secrets.token_hex(16),"token":secrets.token_hex(32),"port":29157,
        "model_control":False,"checkpoint":manifest["checkpoint"],"max_age_ms":500,"max_sim_age_ms":500,
        "presentation_reserve_ms":100,
        "service_max_seconds":max(480,int(seconds)+300),"fault_probe":check,
        "age_definition":"Conservative source physics-step-start to result/presentation; Windows monotonic timestamps only",
        "actual_exposure_wall_time_available":False,"pending_capacity":1,"inflight_capacity":1}
    save(output / "session.json",config)
    document={"role":"fresh_confirmation_only","layout":"crossroad",
        "confirmation_scene":{"junction_radius_m":10.,"scenery":"lowrise","asphalt_palette":"worn_warm"},
        "episodes":[{"episode_id":"live_preview","layout":"crossroad","split":"test","case":case,
            "seed":seed,"weather":weather,"sim_seconds":seconds,"sun_azimuth_deg":110.,"capture_hz":5,
            "camera_profile":{"height_m":1.35,"pitch_deg":0.,"yaw_deg":0.,"fov_deg":95.}}]}
    save(output / "scene_plan.json",document)
    state={"status":"preparing","check":check,"protected_sha256":before,
        "checkpoint_sha256":sha(checkpoint),"rule_controller_sha256":sha(ROOT / "integration/india_behavior.py"),
        "recorded_loader_sha256":sha(ROOT / "idd_simulation_inference.py"),"model_control":False}
    save(output / "status.json",state)
    worker=None
    worker_log=None
    try:
        if check:
            state["status"]="capturing_matched_baseline";save(output / "status.json",state)
            if reuse_baseline:
                saved=json.loads(reuse_baseline.read_text())
                previous_plan=Path(saved["argv"][saved["argv"].index("--dataset-plan")+1])
                assert json.loads(previous_plan.read_text())==document
                assert saved["exit_code"]==0 and saved["launcher"]["owned_job_handles_closed"]
                assert sha(Path(saved["launcher"]["output"]) / "source/india_behavior.py")==state["rule_controller_sha256"]
                save(output / "baseline_capture.json",saved)
                state["baseline_reused_from"]=str(reuse_baseline.resolve())
            else:
                native_run(output,output / "scene_plan.json",seed,seconds)
        # Refuse an occupied endpoint rather than attach to another process.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",config["port"]))
        state["status"]="loading_gpu_worker";save(output / "status.json",state)
        distribution=wsl_distribution()
        interpreter=wsl_python(distribution)
        argv=["wsl.exe","-d",distribution,"--cd",linux(ROOT),"--exec","/usr/bin/env",
              "TF_CPP_MIN_LOG_LEVEL=2","PYTHONUNBUFFERED=1","CUBLAS_WORKSPACE_CONFIG=:4096:8",
              interpreter,"-u","setup/live_simulation_worker.py",
              "--config",linux(output / "session.json")]
        worker_log=(output / "worker.log").open("w")
        worker=subprocess.Popen(argv,cwd=ROOT,stdout=worker_log,stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        state["worker_host_pid"]=worker.pid
        save(output / "status.json",state)
        deadline=time.monotonic()+120
        while not (output / "worker_ready.json").exists():
            if worker.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("GPU worker did not become ready; inspect worker.log")
            time.sleep(.2)
        state["status"]="live_preview";save(output / "status.json",state)
        native_run(output,output / "scene_plan.json",seed,seconds,output / "session.json")
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed_preserved",error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        save(output / "stop_worker.json",{"session":config["session"],"stop":True})
        if worker is not None:
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.terminate()  # This owner created this exact Windows host process.
                worker.wait(timeout=5)
                state.update(status="failed_preserved",error="GPU worker did not release on its stop marker")
            state["worker_host_exit_code"]=worker.returncode
        if worker_log:
            worker_log.close()
        unchanged=(protected()==before and sha(checkpoint)==state["checkpoint_sha256"] and
                   sha(ROOT / "integration/india_behavior.py")==state["rule_controller_sha256"] and
                   sha(ROOT / "idd_simulation_inference.py")==state["recorded_loader_sha256"])
        state["protected_files_unchanged"]=unchanged
        if not unchanged:
            state.update(status="failed_preserved",error="Protected model/controller changed")
        save(output / "status.json",state)
        print(json.dumps({k:state[k] for k in ("status","model_control","protected_files_unchanged")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--reuse-baseline",type=Path,help="Use an identical completed baseline for repeated engineering checks")
    parser.add_argument("--sim-seconds",type=float,default=110)
    parser.add_argument("--seed",type=int,default=4101)
    parser.add_argument("--case",choices=("mixed","clear_overtake","oncoming","pedestrian","return_gap","junction"),default="mixed")
    parser.add_argument("--weather",choices=("clear","overcast","wet","dusk"),default="clear")
    args=parser.parse_args()
    if not 20<=args.sim_seconds<=150:
        parser.error("Choose a bounded 20..150 simulation seconds")
    stamp=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5,minutes=30))).strftime("%Y%m%dT%H%M%S")
    run(args.output or ROOT / "runs" / ("simulation_live_"+stamp),check=args.check,
        seconds=args.sim_seconds,seed=args.seed,case=args.case,weather=args.weather,reuse_baseline=args.reuse_baseline)
