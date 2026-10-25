"""Read-only checkpoint, metrics and GPU-health inspection; never resumes training."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    output.mkdir(exist_ok=False,parents=True)
    (output/"settings").mkdir();(output/"matplotlib").mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output/"settings"),MPLCONFIGDIR=str(output/"matplotlib"),
        YOLO_AUTOINSTALL="False",WANDB_DISABLED="true",MPLBACKEND="Agg")
    import torch
    torch.set_num_threads(4)
    from ultralytics import YOLO,settings
    settings.update({key:False for key in ("sync","wandb","comet","clearml","mlflow","neptune","raytune","tensorboard","dvc","hub") if key in settings})
    from idd_detection_data import file_sha,save_json
    from setup.verify_detection_runtime import protection
    run=ROOT/"runs/detection_train_20261024_04"
    owner=run.with_name(run.name+"_launcher")
    read=lambda path:json.loads(path.read_text(encoding="utf-8-sig"))
    names=read(ROOT/"detection_policy.json")["names"]
    before=protection()
    assert before==read(owner/"protected_before.json")
    with (run/"results.csv").open() as stream:
        rows=[{key.strip():float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    assert [row["epoch"] for row in rows]==list(range(1,len(rows)+1))
    best=max(rows,key=lambda row:row["metrics/mAP50-95(B)"])
    checkpoints={}
    for filename in ("best.pt","last.pt","last_resume_fp32.pt"):
        path=run/"weights"/filename
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        model=checkpoint.get("ema") or checkpoint.get("model")
        finite=all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point())
        raw=checkpoint.get("model")
        raw_finite=raw is None or all(torch.isfinite(t).all().item() for t in raw.state_dict().values() if t.is_floating_point())
        actual_names=[model.names[i] for i in range(len(names))]
        expected_epoch=int(best["epoch"] if filename=="best.pt" else rows[-1]["epoch"])
        state=checkpoint.get("optimizer",{}).get("state",{})
        moment_types=sorted({str(value.dtype) for entry in state.values() for key,value in entry.items()
            if key in {"exp_avg","exp_avg_sq"} and isinstance(value,torch.Tensor)})
        optimizer_finite=all(torch.isfinite(value).all().item() for entry in state.values()
            for value in entry.values() if isinstance(value,torch.Tensor) and value.is_floating_point())
        passed=finite and raw_finite and optimizer_finite and actual_names==names and checkpoint["epoch"]+1==expected_epoch
        if filename=="last_resume_fp32.pt":
            passed=passed and checkpoint.get("project_full_precision_resume") is True and moment_types==["torch.float32"] and raw is not None
        checkpoints[filename]={"passed":passed,"epoch_completed":checkpoint["epoch"]+1,"sha256":file_sha(path),
            "bytes":path.stat().st_size,"all_model_tensors_finite":finite and raw_finite,"optimizer_finite":optimizer_finite,
            "optimizer_state_entries":len(state),"moment_dtypes":moment_types,"ema_updates":checkpoint.get("updates"),
            "class_names_match":actual_names==names,"full_precision_resume":bool(checkpoint.get("project_full_precision_resume"))}
        del checkpoint,model,raw
    gpu={"cuda_available":torch.cuda.is_available(),"checkpoint_forward_pass":False,"scalar_backward_pass":False}
    if gpu["cuda_available"]:
        try:
            gpu["device"]=torch.cuda.get_device_name(0)
            x=torch.tensor(2.,device="cuda",requires_grad=True)
            (x*x).backward();torch.cuda.synchronize()
            gpu["scalar_backward_pass"]=float(x.grad)==4.
            import numpy as np
            model=YOLO(str(run/"weights/best.pt"),task="detect")
            prediction=model.predict(source=np.zeros((540,960,3),dtype=np.uint8),imgsz=960,device=0,verbose=False)
            gpu["checkpoint_forward_pass"]=len(prediction)==1 and len(model.names)==len(names)
            gpu["scope"]="one synthetic image and a scalar gradient check; not accuracy evaluation or training"
            del model,prediction,x
            torch.cuda.empty_cache()
        except Exception as error:
            gpu["error"]=str(error)
    baseline=read(ROOT/"runs/detection_improvement_20261024_01/diagnostics_960.json")["metrics"]
    keys=["metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/precision(B)","metrics/recall(B)"]
    current={key:rows[-1][key] for key in keys}
    status=read(run/"status.json")
    processes=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():continue
        try:
            arguments=(proc/"cmdline").read_bytes().split(bytes([0]))
            if str(ROOT/"train_detection.py").encode() in arguments:
                processes.append({"pid":int(proc.name),"name":(proc/"comm").read_text().strip()})
        except OSError:pass
    result={"checked_utc":datetime.now(timezone.utc).isoformat(),
        "checked_ist":datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
        "status":status["status"],"completed_epochs":len(rows),"maximum_epochs":30,
        "best_epoch":int(best["epoch"]),"latest_validation":current,"starting_model_960":{key:baseline[key] for key in keys},
        "gain_percentage_points":{key:100*(current[key]-baseline[key]) for key in keys},
        "checkpoints":checkpoints,"all_checkpoints_passed":all(item["passed"] for item in checkpoints.values()),
        "gpu_probe":gpu,"remaining_training_processes":processes,
        "protected_files_unchanged":protection()==before,"training_restarted":False,"official_test_used":False,
        "final_evaluation_available":(run/"final_validation.json").exists(),
        "exit":read(owner/"exit.json") if (owner/"exit.json").is_file() else None}
    save_json(output/"verification.json",result)
    save_json(ROOT/"docs/results/detection_progress_20261025.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output.resolve())
