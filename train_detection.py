"""Check detector readiness by default. Training requires --start-training.

Never loads a road model, inspects unlabelled official test images, or starts
training just because the archive download has finished.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import traceback

from idd_detection_data import file_sha, load_policy, save_json, source_path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def check_prepared_files(dataset, expected_counts, *, workers=6, progress=None):
    """Audit every regular file; resolve four fixed parents instead of each ancestor per row."""
    if not 1 <= workers <= 16:
        raise ValueError("Use 1 to 16 file-audit workers")
    dataset=Path(dataset).resolve()
    parents={}
    for folder in ("images/train","images/val","labels/train","labels/val"):
        resolved=source_path(dataset,folder)
        if resolved != dataset/folder or not resolved.is_dir():
            raise ValueError("Prepared image/label folders must be real directories under this dataset")
        parents[folder]=resolved
    def check(row):
        split=row["split"]
        if split not in {"train","val"}:
            raise ValueError("Training manifest includes an unauthorised split")
        paths=[]
        for field,folder,suffixes in (("prepared_image","images",{".jpg",".jpeg"}),
                                      ("prepared_label","labels",{".txt"})):
            value=row[field]
            relative=PurePosixPath(value)
            if (relative.is_absolute() or len(relative.parts)!=3 or ".." in relative.parts or
                "\\" in value or ":" in value or relative.parts[:2]!=(folder,split) or
                relative.suffix.lower() not in suffixes):
                raise ValueError("Prepared file paths must match their fixed dataset split folders")
            paths.append(parents[folder+"/"+split]/relative.name)
        image,label=paths
        if image.stem!=label.stem:
            raise ValueError("Prepared image and label identities differ")
        try:
            info=image.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_size!=row["image_bytes"] or
                info.st_mtime_ns//1000000!=row["image_mtime_ms"] or not stat.S_ISREG(label.lstat().st_mode)):
                return split,False
            flags=os.O_RDONLY | getattr(os,"O_BINARY",0) | getattr(os,"O_NOFOLLOW",0)
            with os.fdopen(os.open(label,flags),"rb") as stream:
                valid=(stat.S_ISREG(os.fstat(stream.fileno()).st_mode) and
                       hashlib.file_digest(stream,"sha256").hexdigest()==row["label_sha256"])
            return split,valid
        except OSError:
            return split,False
    counts={"train":0,"val":0}
    started=last=time.monotonic()
    valid=True
    with (dataset/"pairs.jsonl").open(encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as executor:
        records=(json.loads(line) for line in stream)
        while batch:=list(itertools.islice(records,96)):
            for split,passed in executor.map(check,batch):
                if not passed:
                    valid=False
                    break
                counts[split]+=1
            if not valid:
                break
            now=time.monotonic()
            if progress and now-last>=20:
                progress({"file_pairs_checked":sum(counts.values()),"expected":sum(expected_counts.values()),
                    "elapsed_seconds":round(now-started,1)})
                last=now
    for folder,parent in parents.items():
        if source_path(dataset,folder)!=parent:
            raise ValueError("Prepared parent directory changed during the file audit")
    return counts, valid and counts==expected_counts


def readiness(dataset=None, recipe_path=ROOT / "detection_training_recipe.json", *, inspect_files=True, file_progress=None):
    recipe = load(recipe_path)
    if recipe.get("schema") != "idd_detection_training_recipe_v1" or recipe.get("control_allowed") is not False:
        raise ValueError("Invalid observation-only training recipe")
    dataset = Path(dataset or ROOT / recipe["dataset"]).resolve()
    reasons, checks = [], {}
    raw = ROOT / "dataset/idd117k_raw"
    download = load(raw/"download_progress.json") if (raw/"download_progress.json").exists() else {}
    verification = load(raw/"download_verification.json") if (raw/"download_verification.json").exists() else {}
    checks["download_verified"] = download.get("status")=="completed" and verification.get("passed") is True
    if not checks["download_verified"]:
        reasons.append("The full five-part archive is not yet downloaded and verified")
    report_path = dataset / "preparation.json"
    report = load(report_path) if report_path.exists() else {}
    checks["prepared_real_dataset"] = (report.get("schema")=="idd_detection_dataset_v1" and
        report.get("scope")=="real_idd_detection" and report.get("passed") is True and report.get("training_allowed") is True)
    if not checks["prepared_real_dataset"]:
        reasons.append("A fully audited real IDD detection dataset has not been prepared")
    runtime_path = ROOT / recipe["runtime_verification"]
    runtime = load(runtime_path) if runtime_path.exists() else {}
    checks["gpu_forward_runtime_checked"] = bool(runtime.get("passed") is True and runtime.get("training_started") is False
        and runtime.get("protected_files_unchanged") is True and runtime.get("batch")==recipe["batch"]
        and runtime.get("imgsz")==recipe["imgsz"] and runtime.get("isolated_settings_file"))
    if not checks["gpu_forward_runtime_checked"]:
        reasons.append("The isolated GPU forward-runtime check has not passed")
    weights = ROOT / recipe["initial_weights"]
    provenance_path = ROOT / recipe["initial_weights_metadata"]
    provenance = load(provenance_path) if provenance_path.exists() else {}
    checkpoint_matches = (weights.is_file() and provenance.get("model")==recipe["model"]
        and file_sha(weights)==provenance.get("sha256")==runtime.get("checkpoint_sha256"))
    if recipe.get("initialization", "official_coco") == "verified_idd_detector":
        source_report_path = source_path(ROOT, provenance.get("verification_file", "missing"))
        source_report = load(source_report_path) if source_report_path.is_file() else {}
        verified = source_report.get("checkpoints", {}).get("best", {})
        checks["initial_weights_verified"] = bool(checkpoint_matches
            and provenance.get("idd_finetuning_completed") is True and provenance.get("control_allowed") is False
            and source_report.get("passed") is True and verified.get("all_tensors_finite") is True
            and source_report_path.is_file() and file_sha(source_report_path)==provenance.get("verification_sha256")
            and verified.get("sha256")==provenance.get("sha256")
            and verified.get("path")==recipe["initial_weights"]
            and verified.get("names")==report.get("names")==runtime.get("names"))
    else:
        checks["initial_weights_verified"] = (recipe.get("initialization", "official_coco")=="official_coco"
            and checkpoint_matches and provenance.get("training_dataset")=="COCO"
            and provenance.get("idd_finetuning_completed") is False)
    if not checks["initial_weights_verified"]:
        reasons.append("The initial checkpoint/provenance does not match its verified source and runtime proof")
    if checks["prepared_real_dataset"]:
        policy = load_policy()
        provenance = report.get("archive_provenance", {})
        archive_ok = set(provenance) == {"source_index", "archive_integrity", "extraction"}
        for row in provenance.values():
            path = source_path(dataset, row["file"])
            archive_ok = archive_ok and path.is_file() and file_sha(path) == row["sha256"]
            if path.is_file():
                archive_ok = archive_ok and load(path).get("passed") is True
        checks["full_archive_and_source_index_verified"] = archive_ok
        if not archive_ok:
            reasons.append("Full archive integrity, extraction and source-index evidence is missing or changed")
        consistent = (report.get("policy_sha256")==file_sha(ROOT/"detection_policy.json")
            and report.get("names")==policy["names"] and report.get("provider_counts")==recipe["expected_provider_counts"]
            and not report.get("quarantine") and not report.get("cross_split_duplicates")
            and not report.get("missing_train_or_validation_classes") and archive_ok)
        if recipe.get("expected_prepared_counts") is not None:
            consistent = consistent and report.get("counts") == recipe["expected_prepared_counts"]
        if not consistent:
            reasons.append("Prepared counts, class policy or data-quality guards differ from the recipe")
        for name, key in (("pairs.jsonl","manifest_sha256"),("data.yaml","data_yaml_sha256")):
            if not (dataset/name).is_file() or file_sha(dataset/name)!=report.get(key):
                reasons.append("Prepared manifest or dataset configuration changed")
                consistent = False
        if (raw/"download_verification.json").is_file() and file_sha(raw/"download_verification.json")!=report.get("download_verification_sha256"):
            reasons.append("The preparation does not match the completed download record")
            consistent = False
        if consistent and inspect_files:
            n,files_ok=check_prepared_files(dataset,{s:report["counts"][s] for s in ("train","val")},progress=file_progress)
            if not files_ok:
                reasons.append("A prepared image/label changed after its audit")
                consistent=False
            if n != {s:report["counts"][s] for s in n}:
                reasons.append("Prepared image counts differ from the audited manifest")
                consistent = False
        checks["dataset_manifest_and_files_unchanged"] = consistent
    if recipe.get("training_view_verification"):
        verification_path = source_path(ROOT, recipe["training_view_verification"])
        cleaned = load(verification_path) if verification_path.is_file() else {}
        independent_path = source_path(ROOT, recipe.get("independent_data_audit", "missing"))
        independently_checked = load(independent_path) if independent_path.is_file() else {}
        checks["clean_training_view_verified"] = bool(verification_path.is_file()
            and file_sha(verification_path)==recipe.get("training_view_verification_sha256")
            and cleaned.get("passed") is True and cleaned.get("training_pixels_unique") is True
            and cleaned.get("validation_membership_and_labels_unchanged") is True
            and cleaned.get("final_pairs_sha256")==report.get("manifest_sha256")
            and independent_path.is_file() and file_sha(independent_path)==recipe.get("independent_data_audit_sha256")
            and independently_checked.get("passed") is True and independently_checked.get("all_saved_labels_reopened") is True
            and independently_checked.get("counts")=={key:report.get("counts",{}).get(key) for key in ("train","val")})
        if not checks["clean_training_view_verified"]:
            reasons.append("The cleaned training-view or independent label audit is missing or changed")
    return {"ready_for_explicit_training_start":not reasons, "training_started":False,
        "checked_utc":datetime.now(timezone.utc).isoformat(), "checks":checks,"waiting_for":reasons,
        "dataset":str(dataset),"recipe_sha256":file_sha(recipe_path),
        "download_percent":download.get("percent"),"backward_memory_still_to_check":True,"control_allowed":False,
        "full_file_check_performed":bool(inspect_files and checks.get("dataset_manifest_and_files_unchanged") and not reasons)}


def train(dataset, output, recipe_path, *, resume=False, reset_optimizer=False):
    output = output.resolve()
    if output.parent!=ROOT/"runs" or not output.name.startswith("detection_train_"):
        raise ValueError("Use a runs/detection_train_* output")
    previous_state = load(output/"status.json") if resume else {}
    previous_batch = load(output/"batch_progress.json") if resume else {}
    if resume:
        previous_state.setdefault("optimizer_steps_completed",previous_batch.get("optimizer_steps_completed",0))
    resume_evidence = None
    if resume:
        expected_recipe = load(recipe_path)
        expected_recipe["data"] = str((ROOT/expected_recipe["dataset"]).resolve()/"data.yaml")
        if load(output/"recipe.json") != expected_recipe:
            raise ValueError("Resume requires the original unchanged training recipe")
        if previous_state.get("status") in {"completed", "epochs_and_final_validation_complete"}:
            raise ValueError("A completed run cannot be resumed as an interruption")
        resume_evidence = output/"recovery"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        resume_evidence.mkdir(parents=True)
        for name in ("weights/last.pt", "weights/best.pt", "weights/last_resume_fp32.pt", "results.csv", "status.json", "batch_progress.json", "progress.json", "args.yaml", "recipe.json"):
            candidate=output/name
            if candidate.is_file():
                target=resume_evidence/name; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(candidate,target)
        checkpoint_file="last_resume_fp32.pt" if (resume_evidence/"weights/last_resume_fp32.pt").is_file() else "last.pt"
        save_json(output/"active_recovery.json",{"directory":str(resume_evidence),"status":"checking_dataset",
            "checkpoint_file":checkpoint_file,"checkpoint_sha256":file_sha(resume_evidence/"weights"/checkpoint_file),"model_control":False})
        save_json(output/"status.json",dict(previous_state,status="checking_dataset",
            resumed_utc=datetime.now(timezone.utc).isoformat()))
    result = readiness(dataset,recipe_path,file_progress=lambda row:print("File readiness: "+json.dumps(row),flush=True))
    if not result["ready_for_explicit_training_start"]:
        raise RuntimeError("Training refused: " + "; ".join(result["waiting_for"]))
    if sys.platform != "linux":
        raise RuntimeError("Use the separate idd-detect WSL runtime")
    output.mkdir(exist_ok=resume)
    recipe = load(recipe_path)
    recipe["data"] = str(Path(result["dataset"])/"data.yaml")
    evidence=resume_evidence or output
    save_json(evidence/"recipe.json",recipe)
    save_json(evidence/"readiness.json",result)
    source = evidence/"source"; source.mkdir()
    source_files = {"train_detection.py", "idd_detection_data.py", "detection_policy.json", "requirements-detection-wsl.txt",
                    Path(recipe_path).resolve().relative_to(ROOT).as_posix()}
    if recipe.get("sampling"):
        source_files.add("detection_refinement.py")
    for name in source_files:
        destination = source/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    os.environ["YOLO_CONFIG_DIR"] = str(output/"ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(output/"matplotlib")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(exist_ok=resume)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=resume)
    os.environ["YOLO_AUTOINSTALL"]="False"
    os.environ["MPLBACKEND"]="Agg"
    os.environ["WANDB_DISABLED"]="true"
    from setup.verify_detection_runtime import protection
    before = protection()
    if resume and before != previous_state.get("protected_sha256"):
        raise ValueError("Protected model/inference files changed since this run started")
    initial_weights_sha256 = file_sha(ROOT/recipe["initial_weights"])
    state = {"status":"starting","training_started":True,"protected_sha256":before,"model_control":False,
             "started_utc":previous_state.get("started_utc",datetime.now(timezone.utc).isoformat()),
             "epochs_completed":previous_state.get("epochs_completed",0),"resumed":resume}
    save_json(output/"status.json",state)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO,settings
        settings.update({k:False for k in ("sync","wandb","comet","clearml","mlflow","neptune","raytune","tensorboard","dvc","hub") if k in settings})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not silently train on CPU")
        runtime=load(ROOT/recipe["runtime_verification"])
        if torch.__version__!=runtime["torch"] or ultralytics.__version__!=runtime["ultralytics"]:
            raise RuntimeError("Detector runtime changed; repeat readiness checks before training")
        resumed_epochs = 0
        checkpoint_path = output/"weights/last.pt"
        if resume and (output/"weights/last_resume_fp32.pt").is_file():
            checkpoint_path=output/"weights/last_resume_fp32.pt"
        if resume:
            checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
            resumed_epochs = int(checkpoint["epoch"])+1
            if not 0 < resumed_epochs < recipe["epochs"] or checkpoint.get("optimizer") is None or checkpoint.get("ema") is None:
                raise ValueError("Resume requires a partial checkpoint with optimizer and EMA state")
            critical=("data","epochs","batch","imgsz","optimizer","lr0","lrf","seed","workers","amp","cache","deterministic","cos_lr","warmup_epochs","warmup_bias_lr","mosaic","scale","close_mosaic","patience","save_period")
            if any(checkpoint["train_args"].get(key)!=recipe[key] for key in critical if key in recipe):
                raise ValueError("Checkpoint training settings differ from the original recipe")
            expected_names=load(ROOT/"detection_policy.json")["names"]
            actual_names=[checkpoint["ema"].names[index] for index in range(len(expected_names))]
            if actual_names!=expected_names or not all(torch.isfinite(v).all().item() for v in checkpoint["ema"].state_dict().values() if v.is_floating_point()):
                raise ValueError("Checkpoint labels or model tensors failed verification")
            resume_record={"passed":True,"checkpoint_sha256":file_sha(checkpoint_path),
                "completed_epochs_restored":resumed_epochs,"next_epoch":resumed_epochs+1,
                "optimizer_state_present":True,"ema_updates":checkpoint["updates"],
                "optimizer_state_restored":not reset_optimizer,"optimizer_reset":reset_optimizer,
                "optimizer_reset_reason":"Saved FP16 AdamW variance lost precision and ordinary resume diverged" if reset_optimizer else None,
                "full_precision_checkpoint":bool(checkpoint.get("project_full_precision_resume")),
                "original_settings_preserved":True,"protected_files_unchanged":True,
                "sampler_and_augmentation_rng_exact_replay":False,"model_control":False}
            save_json(evidence/"resume_verification.json",resume_record)
            del checkpoint
            state["epochs_completed"]=resumed_epochs
        model = YOLO(str(checkpoint_path if resume else ROOT/recipe["initial_weights"]),task="detect")
        counter = {"batches":0,"session_batches":0,"epoch_batches":0,
            "optimizer_steps":previous_state.get("optimizer_steps_completed",0) if resume else 0,
            "last_saved":0.0,"started":None}
        def optimizer_step_seen(optimizer, args, kwargs):
            counter["optimizer_steps"] += 1
        def training_started(trainer):
            counter["started"] = time.monotonic()
            counter["batches"] = resumed_epochs*len(trainer.train_loader)
            if resume:
                if trainer.start_epoch!=resumed_epochs or trainer.ema.updates!=resume_record["ema_updates"]:
                    raise ValueError("Framework did not restore the expected epoch and EMA state")
                save_json(evidence/"framework_resume.json",dict(resume_record,
                    restored_start_epoch=trainer.start_epoch,restored_ema_updates=trainer.ema.updates,
                    optimizer_state_entries=len(trainer.optimizer.state),checked_utc=datetime.now(timezone.utc).isoformat()))
            torch.cuda.reset_peak_memory_stats()
            trainer.optimizer.register_step_post_hook(optimizer_step_seen)
            state["status"] = "training"
            state["training_batches_started_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(output/"status.json",state)
        def epoch_started(trainer):
            counter["epoch_batches"] = 0
        def batch_finished(trainer):
            counter["batches"] += 1
            counter["session_batches"] += 1
            counter["epoch_batches"] += 1
            now = time.monotonic()
            if counter["session_batches"] > 10 and now-counter["last_saved"] < 20:
                return
            values = trainer.loss_items.detach().float().cpu().reshape(-1).tolist()
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError("Non-finite training loss; preserve this run for inspection")
            if resume and counter["session_batches"]>=4 and values[1]>20:
                raise FloatingPointError("Abnormal classification loss after recovery; preserve the original checkpoint")
            report = {"status":"training","checked_utc":datetime.now(timezone.utc).isoformat(),
                "epoch":trainer.epoch+1,"epochs_max":recipe["epochs"],
                "batches_completed_in_epoch":counter["epoch_batches"],"batches_per_epoch":len(trainer.train_loader),
                "total_batches_completed":counter["batches"],"optimizer_steps_completed":counter["optimizer_steps"],
                "losses":dict(zip(trainer.loss_names,values)),"sampled_losses_finite":True,
                "elapsed_training_seconds":round(now-counter["started"]+(previous_batch.get("elapsed_training_seconds",0) if resume else 0),2),
                "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "gpu":torch.cuda.get_device_name(0),"batch":recipe["batch"],"imgsz":recipe["imgsz"],
                "training_started":True,"model_control":False}
            temporary = output/"batch_progress.tmp"
            save_json(temporary,report)
            temporary.replace(output/"batch_progress.json")
            if counter["session_batches"] >= 10 and counter["optimizer_steps"] > (previous_state.get("optimizer_steps_completed",0) if resume else 0) and not (evidence/"first_training_batches.json").exists():
                save_json(evidence/"first_training_batches.json",dict(report,passed=True,
                    session_batches_completed=counter["session_batches"],resumed=resume,
                    session_optimizer_steps=counter["optimizer_steps"]-(previous_state.get("optimizer_steps_completed",0) if resume else 0),
                    scope="real_training_backward_and_optimizer_steps_not_accuracy"))
            counter["last_saved"] = now
        def progress(trainer):
            save_json(output/"progress.json",{"epoch_completed":trainer.epoch+1,"epochs_max":recipe["epochs"],
                "metrics":{str(k):float(v) for k,v in trainer.metrics.items()},"model_control":False})
            state["epochs_completed"] = trainer.epoch+1
            state["optimizer_steps_completed"] = counter["optimizer_steps"]
            save_json(output/"status.json",state)
        def training_ended(trainer):
            state["status"] = "epochs_and_final_validation_complete"
            state["best_checkpoint"] = "weights/best.pt"
            state["protected_files_unchanged"] = protection() == before
            save_json(output/"training_end_verification.json", {
                "checked_utc":datetime.now(timezone.utc).isoformat(),
                "epochs_completed":state["epochs_completed"], "optimizer_steps_completed":counter["optimizer_steps"],
                "protected_files_unchanged":state["protected_files_unchanged"],
                "best_checkpoint_sha256":file_sha(output/"weights/best.pt"),
                "last_checkpoint_sha256":file_sha(output/"weights/last.pt"), "model_control":False})
            save_json(output/"status.json",state)
            if recipe.get("initialization") == "verified_idd_detector":
                import numpy as np
                box = trainer.validator.metrics.box
                per_class = {}
                for index, class_id in enumerate(box.ap_class_index):
                    per_class[trainer.data["names"][int(class_id)]] = {
                        "precision":float(box.p[index]), "recall":float(box.r[index]),
                        "ap50":float(box.ap50[index]), "ap50_95":float(box.ap[index])}
                    eligible = np.flatnonzero(box.p_curve[index] >= 0.75)
                    selected = int(eligible[np.argmax(box.r_curve[index][eligible])]) if len(eligible) else None
                    per_class[trainer.data["names"][int(class_id)]]["at_precision_at_least_0_75"] = None if selected is None else {
                        "confidence":float(box.px[selected]), "precision":float(box.p_curve[index][selected]),
                        "recall":float(box.r_curve[index][selected])}
                save_json(output/"final_validation.json", {"metrics":{str(k):float(v) for k,v in trainer.metrics.items()},
                    "per_class":per_class, "split":"validation", "official_test_used":False,
                    "imgsz":recipe["imgsz"], "model_control":False})
        model.add_callback("on_train_start",training_started)
        model.add_callback("on_train_epoch_start",epoch_started)
        model.add_callback("on_train_batch_end",batch_finished)
        model.add_callback("on_fit_epoch_end",progress)
        model.add_callback("on_train_end",training_ended)
        state["status"]="preparing_training"; save_json(output/"status.json",state)
        kwargs={k:recipe[k] for k in ("data","imgsz","batch","epochs","patience","optimizer","lr0","lrf","seed","workers","amp","cache","close_mosaic","deterministic")}
        kwargs.update({key:recipe[key] for key in ("cos_lr", "warmup_epochs", "warmup_bias_lr", "mosaic", "scale",
            "fliplr", "nbs", "cls", "save_period", "max_det") if key in recipe})
        if recipe.get("sampling"):
            from detection_refinement import make_refinement_trainer
            kwargs["trainer"] = make_refinement_trainer(recipe["sampling"], Path(result["dataset"]), output,
                reset_optimizer=reset_optimizer)
        if resume:
            kwargs["resume"] = str(checkpoint_path)
        # This process alone just created this fresh folder. exist_ok does not
        # permit reusing any earlier run; it prevents Ultralytics renaming it.
        model.train(**kwargs,device=0,project=str(output.parent),name=output.name,exist_ok=True,
                    pretrained=True,save=True,plots=True)
        state["status"]="completed"
        state["best_checkpoint"]="weights/best.pt"
    except BaseException as error:
        state.update(status="failed_preserved",error_type=type(error).__name__)
        (output/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        state["protected_files_unchanged"] = protection()==before
        state["initial_checkpoint_unchanged"] = file_sha(ROOT/recipe["initial_weights"])==initial_weights_sha256
        if not state["protected_files_unchanged"] or not state["initial_checkpoint_unchanged"]:
            state.update(status="failed_preserved",error_type="ProtectedFileChanged")
        save_json(output/"status.json",state)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--recipe",type=Path,default=ROOT/"detection_training_recipe.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--start-training",action="store_true")
    parser.add_argument("--resume",action="store_true",help="Restore this interrupted run's last checkpoint and optimizer")
    parser.add_argument("--reset-optimizer",action="store_true",help="Explicitly replace unstable saved optimizer moments while keeping weights and epoch")
    args=parser.parse_args()
    if args.resume and not args.start_training:
        parser.error("--resume requires --start-training")
    if args.reset_optimizer and not args.resume:
        parser.error("--reset-optimizer requires --resume")
    if args.start_training:
        if args.output is None:
            parser.error("An explicit fresh --output is required for training")
        train(args.dataset,args.output,args.recipe,resume=args.resume,reset_optimizer=args.reset_optimizer)
    else:
        report=readiness(args.dataset,args.recipe)
        print(json.dumps(report,indent=2))
        if args.output:
            save_json(args.output,report)
        raise SystemExit(0 if report["ready_for_explicit_training_start"] else 2)
