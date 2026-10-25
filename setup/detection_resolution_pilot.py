"""Prepare, run and compare one authorized three-epoch resolution adaptation pilot."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json

PARENT = "detection_train_20261024_04"
PILOT = "detection_train_20261025_01"
PREP = ROOT / "runs/detection_pilot_prepare_20261025_01"


def phase(name, **extra):
    save_json(PREP / "progress.json", {"phase": name, "checked_utc": datetime.now(timezone.utc).isoformat(), **extra})


def paired_validation(checkpoint, output, names):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    active = [None]
    class CapturedValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            active[0] = self
    model = YOLO(str(checkpoint), task="detect")
    last_saved = [0.0]
    def seen(validator):
        if time.monotonic() - last_saved[0] > 20:
            phase(output.name, images_seen=int(validator.seen), images_expected=17675)
            last_saved[0] = time.monotonic()
    model.add_callback("on_val_batch_end", seen)
    metrics = model.val(data=str(ROOT / "dataset/idd117k_detection_v3/data.yaml"),
        validator=CapturedValidator, imgsz=1280, batch=4, workers=4, device=0,
        half=False, conf=.001, iou=.7, max_det=300, augment=False, plots=False,
        save_json=False, project=str(PREP), name=output.name, verbose=False)
    assert active[0].seen == 17675 and int(metrics.nt_per_class.sum()) == 319438
    result = {"images": 17675, "targets": 319438, "resolution": 1280, "precision": "GPU FP32",
        "checkpoint_sha256": file_sha(checkpoint),
        "metrics": {str(k): float(v) for k, v in metrics.results_dict.items()},
        "per_class": {names[int(c)]: {"ap50": float(metrics.box.ap50[i]),
            "ap50_95": float(metrics.box.ap[i]), "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i])} for i, c in enumerate(metrics.box.ap_class_index)},
        "speed_ms_per_image": metrics.speed, "official_test_used": False}
    assert set(result["per_class"]) == set(names)
    save_json(PREP / (output.name + ".json"), result)
    active[0] = None
    del model, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return result


def execute():
    os.environ.update(YOLO_CONFIG_DIR=str(PREP / "settings"), MPLCONFIGDIR=str(PREP / "matplotlib"),
        MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for folder in ("settings", "matplotlib"):
        (PREP / folder).mkdir(exist_ok=True)
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    import ultralytics.data.dataset as dataset_module
    from detection_refinement import parallel_dataset_hash
    from setup.verify_detection_runtime import protection
    from train_detection import readiness
    dataset_module.get_hash = parallel_dataset_hash
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if k in settings})
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225"
    assert torch.cuda.is_available()
    parent = ROOT / "runs" / PARENT
    parent_owner = parent.with_name(PARENT + "_launcher")
    completion = json.loads((parent / "refinement_result.json").read_text())
    assert completion["epochs_completed"] == 30 and completion["best_epoch"] == 30
    assert json.loads((parent_owner / "exit.json").read_text())["returncode"] == 0
    measured = json.loads((ROOT / "runs/detection_full_resolution_20261025_01/comparison.json").read_text())
    assert measured["passed"] and all(measured["invariants"].values())
    assert measured["metric_deltas_percentage_points"]["metrics/mAP50(B)"] > 1
    assert measured["metric_deltas_percentage_points"]["metrics/mAP50-95(B)"] > 1
    protected = protection()
    assert protected == json.loads((parent_owner / "protected_before.json").read_text())
    source = parent / "weights/best.pt"
    source_sha = completion["checkpoint_checks"]["best"]["sha256"]
    assert file_sha(source) == source_sha
    shutil.copy2(source, PREP / "parent_best_backup.pt")
    phase("verifying_parent_checkpoint")
    verification_dir = "report_verification_morning_pilot_01"
    subprocess.run([sys.executable, str(ROOT / "setup/verify_trained_detection_checkpoint.py"),
        "--run-name", PARENT, "--directory", verification_dir], cwd=ROOT, check=True)
    verification_path = parent / verification_dir / "checkpoint_verification.json"
    verified = json.loads(verification_path.read_text())
    names = json.loads((ROOT / "detection_policy.json").read_text())["names"]
    assert verified["passed"] and verified["checkpoints"]["best"]["sha256"] == source_sha
    assert verified["checkpoints"]["best"]["names"] == names
    provenance = {"model": "yolo11s", "training_dataset": "IDD117K clean project split",
        "idd_finetuning_completed": True, "control_allowed": False, "sha256": source_sha,
        "source_run": parent.relative_to(ROOT).as_posix(),
        "verification_file": verification_path.relative_to(ROOT).as_posix(),
        "verification_sha256": file_sha(verification_path)}
    save_json(PREP / "initial_checkpoint.json", provenance)
    phase("runtime_forward_check")
    model = YOLO(str(source), task="detect")
    model.model.to("cuda:0").float().eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prediction = model.model(torch.zeros((4, 3, 1280, 1280), device="cuda:0"))
        raw = prediction[0] if isinstance(prediction, tuple) else prediction
        assert list(raw.shape) == [4, 16, 33600] and bool(torch.isfinite(raw).all())
    runtime = {"passed": True, "training_started": False, "protected_files_unchanged": protection() == protected,
        "batch": 4, "imgsz": 1280, "isolated_settings_file": str(settings.file),
        "checkpoint_sha256": source_sha, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "names": names, "gpu": torch.cuda.get_device_name(0), "gpu_peak_forward_allocated_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_steps": 0, "checked_utc": datetime.now(timezone.utc).isoformat()}
    assert runtime["protected_files_unchanged"]
    save_json(PREP / "runtime_verification.json", runtime)
    del model, prediction, raw
    gc.collect()
    torch.cuda.empty_cache()
    phase("baseline_1280")
    baseline = paired_validation(source, PREP / "baseline_1280", names)
    save_json(PREP / "diagnostics_1280.json", baseline)
    analysis = {"passed": True, "checkpoint_sha256": source_sha, "resolutions": [1280],
        "baseline": "baseline_1280.json", "precision": "GPU FP32", "source_completed_epoch": 30,
        "hypothesis": "Short gentle training at 1280 may adapt small-object features to the resolution with proven paired gains; animal retention remains an explicit failure criterion.",
        "earlier_resolution_evidence": "runs/detection_full_resolution_20261025_01/comparison.json",
        "official_test_used": False, "training_started": False}
    save_json(PREP / "analysis.json", analysis)
    recipe = json.loads((ROOT / "detection_refinement_clean_recipe.json").read_text())
    recipe.update(initial_weights=source.relative_to(ROOT).as_posix(),
        initial_weights_metadata=(PREP / "initial_checkpoint.json").relative_to(ROOT).as_posix(),
        runtime_verification=(PREP / "runtime_verification.json").relative_to(ROOT).as_posix(),
        imgsz=1280, batch=4, epochs=3, patience=3, lr0=.00003, lrf=.5,
        warmup_epochs=0.0, warmup_bias_lr=0.0, mosaic=0.0, close_mosaic=0,
        save_period=1, baseline_analysis=(PREP / "analysis.json").relative_to(ROOT).as_posix(),
        baseline_analysis_sha256=file_sha(PREP / "analysis.json"), matched_comparison_resolution=1280)
    recipe["notes"] = [analysis["hypothesis"], "New three-epoch adaptation from the verified epoch-30 parent; not a restart of its completed run.",
        "Input 1280, existing rare-class sampling retained, mosaic remains off as in the parent's final stage.",
        "Learning rate starts at 0.00003, close to the parent's final rate, with no warmup jump.",
        "Every epoch and FP32 recovery state are saved. Parent best is preserved; no automatic deployment.",
        "Final independent GPU FP32 validation must be compared with this starting checkpoint at the same 1280 settings."]
    recipe_path = PREP / "recipe.json"
    save_json(recipe_path, recipe)
    ready = readiness(recipe_path=recipe_path, inspect_files=False)
    save_json(PREP / "metadata_readiness.json", ready)
    assert ready["ready_for_explicit_training_start"], ready["waiting_for"]
    assert file_sha(source) == source_sha and protection() == protected
    phase("launching_three_epoch_training", pilot=PILOT)
    subprocess.run([sys.executable, str(ROOT / "setup/launch_detection_training.py"), "--run-name", PILOT,
        "--recipe", str(recipe_path)], cwd=ROOT, check=True)
    phase("training", pilot=PILOT)
    owner = ROOT / "runs" / (PILOT + "_launcher")
    record = json.loads((owner / "owner.json").read_text())
    while not (owner / "exit.json").exists():
        stat_path = Path("/proc") / str(record["pid"]) / "stat"
        if not stat_path.exists() or stat_path.read_text().split(") ", 1)[1].split()[19] != record["linux_process_start_ticks"]:
            raise RuntimeError("Pilot supervisor disappeared without an exit record; preserve state for recovery")
        time.sleep(20)
    assert json.loads((owner / "exit.json").read_text())["returncode"] == 0
    phase("matched_final_1280")
    final = paired_validation(ROOT / "runs" / PILOT / "weights/best.pt", PREP / "matched_final_1280", names)
    deltas = {key: 100 * (final["metrics"][key] - value) for key, value in baseline["metrics"].items()}
    result = {"passed": protection() == protected and file_sha(source) == source_sha,
        "pilot": PILOT, "scope": "Matched 17675-image GPU FP32 validation, 1280 input; development only",
        "baseline": baseline, "final": final, "deltas_percentage_points": deltas,
        "class_ap50_deltas_percentage_points": {name: 100 * (final["per_class"][name]["ap50"] - baseline["per_class"][name]["ap50"]) for name in names},
        "parent_best_preserved": file_sha(PREP / "parent_best_backup.pt") == source_sha,
        "automatically_deployed": False, "official_test_used": False,
        "checked_utc": datetime.now(timezone.utc).isoformat()}
    assert result["passed"] and result["parent_best_preserved"]
    save_json(PREP / "matched_pilot_result.json", result)
    phase("completed", measured_gain=bool(deltas["metrics/mAP50(B)"] > 0 and deltas["metrics/mAP50-95(B)"] > 0))


def launch():
    assert sys.platform == "linux"
    assert not (ROOT / "runs" / PILOT).exists() and not (ROOT / "runs" / (PILOT + "_launcher")).exists()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith(b"/train_detection.py") for arg in command) and b"--start-training" in command:
            raise RuntimeError("An active detector trainer must be preserved")
    PREP.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), PREP / "pipeline_source.py")
    with (PREP / "console.log").open("xb", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(PREP / "owner.json", {"pid": child.pid,
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "linux_process_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split(") ", 1)[1].split()[19],
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot": PILOT,
        "authorization": "User authorized measured three-epoch follow-ups and continued training on 25 October 2026; 11 am ends checks, not training."})
    print(json.dumps({"launched": True, "pid": child.pid, "preparation": str(PREP), "pilot": PILOT}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        code = 1
        try:
            execute()
            code = 0
        finally:
            save_json(PREP / "exit.json", {"returncode": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        launch()
