"""Launch one explicitly requested detector run and preserve its exit record."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json


def directories(name):
    if sys.platform != "linux" or Path(name).name != name or not name.startswith("detection_train_"):
        raise ValueError("Use the isolated Linux runtime and a detection_train_* name")
    return ROOT/"runs"/name, ROOT/"runs"/(name+"_launcher")


def worker(name, recipe_path, *, resume=False, reset_optimizer=False):
    run,owner = directories(name)
    command=[sys.executable,"-u",str(ROOT/"train_detection.py"),"--start-training","--output",str(run),
             "--recipe",str(recipe_path)]
    if resume:
        command.append("--resume")
    if reset_optimizer:
        command.append("--reset-optimizer")
    try:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL)
        save_json(owner/"trainer_process.json",{"pid":process.pid,"command":command,
            "started_utc":datetime.now(timezone.utc).isoformat()})
        code=process.wait()
    except BaseException as error:
        save_json(owner/"exit.json",{"status":"failed_preserved","error_type":type(error).__name__,
            "finished_utc":datetime.now(timezone.utc).isoformat()})
        raise
    worker_code=code
    record={"status":"completed" if code==0 else "failed_preserved", "returncode":code,
            "finished_utc":datetime.now(timezone.utc).isoformat()}
    recipe=json.loads(recipe_path.read_text())
    if recipe.get("initialization") == "verified_idd_detector":
        save_json(owner/"trainer_exit.json",record)
        post_report=subprocess.run([sys.executable,str(ROOT/"setup/report_detection_refinement.py"),"--run-name",name],cwd=ROOT)
        record["post_report_returncode"]=post_report.returncode
        if post_report.returncode:
            record["status"]="failed_preserved"
            if not worker_code:
                worker_code=post_report.returncode
        record["supervisor_finished_utc"]=datetime.now(timezone.utc).isoformat()
    save_json(owner/"exit.json",record)
    raise SystemExit(worker_code)


def launch(name, recipe_path, *, resume=False, reset_optimizer=False):
    run,owner=directories(name)
    if not resume and (run.exists() or owner.exists()):
        raise FileExistsError("Preserve existing attempts; use a fresh run name")
    if resume and (not run.is_dir() or not owner.is_dir() or not (run/"weights/last.pt").is_file()):
        raise ValueError("Resume requires an existing interrupted run and last checkpoint")
    if resume and (owner/"exit.json").exists():
        raise ValueError("Inspect the recorded process exit before attempting this recovery")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments=(entry/"cmdline").read_bytes().split(b"\0")
        except (OSError,PermissionError):
            continue
        if any(arg.endswith(b"/train_detection.py") for arg in arguments) and b"--start-training" in arguments:
            raise RuntimeError("A detector trainer is already running")
    owner.mkdir(exist_ok=resume)
    if resume:
        recovery=owner/"recovery"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        recovery.mkdir(parents=True)
        for filename in ("owner.json","trainer_process.json","preflight.json","protected_before.json","keep_awake.json"):
            if (owner/filename).is_file():
                shutil.copy2(owner/filename,recovery/filename)
        save_json(recovery/"interruption.json",{"status":"environment_disappeared_without_exit_record",
            "observed_utc":datetime.now(timezone.utc).isoformat(),"cause":"not established",
            "checkpoint_preserved":True,"completed_epochs":json.loads((run/"status.json").read_text()).get("epochs_completed")})
    settings_dir=owner/"ultralytics_settings";settings_dir.mkdir(exist_ok=resume)
    mpl=owner/"matplotlib";mpl.mkdir(exist_ok=resume)
    os.environ.update(YOLO_CONFIG_DIR=str(settings_dir),MPLCONFIGDIR=str(mpl),MPLBACKEND="Agg",
        YOLO_AUTOINSTALL="False",WANDB_DISABLED="true",PYTHONUNBUFFERED="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    import ultralytics
    from train_detection import readiness
    from setup.verify_detection_runtime import protection
    recipe_path=recipe_path.resolve()
    if not recipe_path.is_relative_to(ROOT):
        raise ValueError("The executed recipe must be a saved project file")
    recipe=json.loads(recipe_path.read_text())
    runtime=json.loads((ROOT/recipe["runtime_verification"]).read_text())
    assert torch.__version__==runtime["torch"] and ultralytics.__version__==runtime["ultralytics"]
    assert torch.cuda.is_available(),"CUDA unavailable"
    assert shutil.disk_usage(ROOT).free>10*1024**3,"Insufficient checkpoint space"
    # The trainer repeats the exhaustive per-file audit before creating its run.
    prepared=readiness(recipe_path=recipe_path,inspect_files=False)
    assert prepared["ready_for_explicit_training_start"],prepared["waiting_for"]
    protected=protection()
    if resume and protected!=json.loads((owner/"protected_before.json").read_text()):
        raise ValueError("Protected model/inference files changed since initial launch")
    save_json(owner/"protected_before.json",protected)
    save_json(owner/"preflight.json",{"passed":True,"training_started":False,
        "torch":torch.__version__,"ultralytics":ultralytics.__version__,"cuda":True,
        "gpu":torch.cuda.get_device_name(0),"gpu_memory_free_bytes":torch.cuda.mem_get_info()[0],
        "metadata_readiness":prepared,"full_file_audit":"trainer before starting",
        "recipe_sha256":file_sha(recipe_path),
        "user_authorisation":"User requested detector training/improvement, 24 October 2026 Asia/Calcutta",
        "checked_utc":datetime.now(timezone.utc).isoformat()})
    source=(recovery if resume else owner)/"source";source.mkdir()
    source_files=["setup/launch_detection_training.py","train_detection.py",recipe_path.relative_to(ROOT).as_posix()]
    if recipe.get("initialization") == "verified_idd_detector":
        source_files.extend(["detection_refinement.py","setup/report_detection_refinement.py"])
    for filename in source_files:
        destination=source/filename;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/filename,destination)
    command=[sys.executable,"-u",str(ROOT/"setup/launch_detection_training.py"),"--worker","--run-name",name,
             "--recipe",str(recipe_path)]
    if resume:
        command.append("--resume")
    if reset_optimizer:
        command.append("--reset-optimizer")
    with (owner/"training.log").open("ab" if resume else "xb",buffering=0) as log:
        if resume:
            log.write(("\nResuming verified epoch checkpoint at "+datetime.now(timezone.utc).isoformat()+"\n").encode())
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=os.environ.copy(),start_new_session=True)
    record={"status":"launched","pid":process.pid,"process_group":process.pid,"run_directory":str(run),
        "log":str(owner/"training.log"),"started_utc":datetime.now(timezone.utc).isoformat(),
        "user_requested_training":True,"test_evaluation_started":False}
    record["resumed"]=resume
    record["linux_boot_id"]=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    record["linux_process_start_ticks"]=Path(f"/proc/{process.pid}/stat").read_text().split(") ",1)[1].split()[19]
    save_json(owner/"owner.json",record)
    time.sleep(.25)
    if process.poll() is not None:
        raise RuntimeError("Trainer exited immediately; inspect the preserved log")
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--recipe",type=Path,default=ROOT/"detection_training_recipe.json")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--reset-optimizer",action="store_true")
    args=parser.parse_args()
    if args.reset_optimizer and not args.resume:
        parser.error("--reset-optimizer requires --resume")
    (worker if args.worker else launch)(args.run_name,args.recipe,resume=args.resume,reset_optimizer=args.reset_optimizer)
