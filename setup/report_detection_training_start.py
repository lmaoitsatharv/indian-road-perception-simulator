"""Verify successful real training startup without evaluating or changing a model."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from idd_detection_data import file_sha,save_json
from setup.verify_detection_runtime import protection


def main(name, report_name="detection_training_start"):
    if Path(name).name!=name or not name.startswith("detection_train_"):
        raise ValueError("Expected one local detector run name")
    if Path(report_name).name!=report_name or not report_name.startswith("detection_"):
        raise ValueError("Expected a detector report filename stem")
    run=ROOT/"runs"/name
    owner=ROOT/"runs"/(name+"_launcher")
    def read(path):return json.loads(path.read_text())
    first=read(run/"first_training_batches.json")
    ready=read(run/"readiness.json")
    recipe=read(run/"recipe.json")
    before=read(owner/"protected_before.json")
    after=protection()
    assert first["passed"] and first["total_batches_completed"]>=10 and first["optimizer_steps_completed"]>0
    assert first["sampled_losses_finite"] and all(math.isfinite(v) for v in first["losses"].values())
    assert ready["ready_for_explicit_training_start"] and ready["full_file_check_performed"]
    assert first["batch"]==recipe["batch"] and first["imgsz"]==recipe["imgsz"]
    assert before==after,"An existing protected file changed"
    assert not (owner/"exit.json").exists(),"Trainer already stopped; inspect its final state"
    provenance=read(ROOT/recipe["initial_weights_metadata"])
    assert file_sha(ROOT/recipe["initial_weights"])==provenance["sha256"]
    dataset=ROOT/recipe["dataset"]
    prepared=read(dataset/"preparation.json")
    cache_counts={}
    for split in ("train","val"):
        cache_path=dataset/"labels"/(split+".cache")
        cache=np.load(cache_path,allow_pickle=True).item()  # Locally generated framework cache.
        found,missing,negative,corrupt,total=cache["results"]
        assert found==total==len(cache["labels"])==prepared["counts"][split]
        assert missing==corrupt==0 and negative==prepared["negatives"][split]
        classes=np.concatenate([label["cls"].reshape(-1) for label in cache["labels"]]).astype(int)
        counts=np.bincount(classes,minlength=len(prepared["names"]))
        assert len(counts)==len(prepared["names"])
        by_class=dict(zip(prepared["names"],map(int,counts)))
        assert by_class==prepared["objects"][split],"Framework target counts differ from audited labels"
        cache_counts[split]={"images":int(total),"negatives":int(negative),"corrupt":int(corrupt),
            "boxes":len(classes),"objects_by_class":by_class,"cache_sha256":file_sha(cache_path)}
    save_json(owner/"protected_after_startup.json",after)
    sources={}
    for base in (run/"source",owner/"source"):
        for path in base.rglob("*"):
            if path.is_file():
                relative=path.relative_to(base).as_posix()
                assert file_sha(path)==file_sha(ROOT/relative),relative
                sources[str(path.relative_to(ROOT)).replace("\\","/")]=file_sha(path)
    checks={}
    for variant in ("windows_fixed","wsl_fixed"):
        path=ROOT/"runs/detection_startup_checks_20261024_01"/variant/"verification.json"
        verified=read(path)
        assert verified["passed"] and not verified["training_started"]
        checks[variant]={"passed":True,"check_groups":len(verified["checks"]),"sha256":file_sha(path)}
    result={"schema":"idd_detector_training_start_v1","passed":True,
        "checked_utc":datetime.now(timezone.utc).isoformat(),"user_date_asia_calcutta":"2026-10-24",
        "training_started":True,"status_at_verification":"training","run":"runs/"+name,
        "launcher":"runs/"+name+"_launcher","dataset":recipe["dataset"],
        "counts":recipe["expected_prepared_counts"],"model":recipe["model"],
        "maximum_epochs":recipe["epochs"],"early_stopping_patience":recipe["patience"],
        "batch":recipe["batch"],"imgsz":recipe["imgsz"],"optimizer":recipe["optimizer"],"lr0":recipe["lr0"],
        "seed":recipe["seed"],"recipe_sha256":ready["recipe_sha256"],
        "startup_real_batches":first["total_batches_completed"],"startup_optimizer_steps":first["optimizer_steps_completed"],
        "startup_losses":first["losses"],"gpu":first["gpu"],
        "gpu_peak_allocated_bytes_at_startup":first["gpu_peak_allocated_bytes"],
        "gpu_peak_reserved_bytes_at_startup":first["gpu_peak_reserved_bytes"],
        "full_file_readiness_passed":True,"file_audit_verification":checks,
        "framework_cache_matches_audited_counts":True,"framework_cache_counts":cache_counts,
        "protected_files_unchanged":True,"protected_fingerprints":len(after),
        "initial_checkpoint_unchanged":True,"source_hashes":sources,
        "official_test_used":False,"control_allowed":False,
        "scope":"Startup snapshot, not completed training or an accuracy result; read live batch/status files for current progress"}
    save_json(run/"startup_verification.json",result)
    save_json(ROOT/"docs/results"/(report_name+".json"),result)
    print(json.dumps({k:result[k] for k in ("passed","run","startup_real_batches","startup_optimizer_steps",
        "gpu_peak_allocated_bytes_at_startup","protected_files_unchanged")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--report-name",default="detection_training_start")
    args=parser.parse_args()
    main(args.run_name,args.report_name)
