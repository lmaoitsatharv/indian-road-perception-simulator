"""One bounded diagnostic recovery from the unchanged parent after a CUDA error."""
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
from setup import detection_resolution_pilot as pipeline

OLD = ROOT / "runs/detection_pilot_prepare_20261025_01"
OUTPUT = ROOT / "runs/detection_pilot_recovery_20261025_01"
PILOT = "detection_train_20261025_02"


def execute():
    os.environ.update(YOLO_CONFIG_DIR=str(OUTPUT / "settings"), MPLCONFIGDIR=str(OUTPUT / "matplotlib"),
        MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true",
        CUDA_LAUNCH_BLOCKING="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for folder in ("settings", "matplotlib"):
        (OUTPUT / folder).mkdir(exist_ok=True)
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from setup.verify_detection_runtime import protection
    from train_detection import readiness
    import ultralytics.data.dataset as dataset_module
    from detection_refinement import parallel_dataset_hash
    dataset_module.get_hash = parallel_dataset_hash
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if k in settings})
    torch.set_num_threads(4)
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225"
    assert torch.cuda.is_available()
    assert json.loads((OLD / "exit.json").read_text())["returncode"] != 0
    failed = ROOT / "runs/detection_train_20261025_01"
    assert not (failed / "results.csv").exists(), "Do not discard a saved completed epoch"
    assert "CUDA error: unknown error" in (failed / "error.txt").read_text()
    recipe = json.loads((OLD / "recipe.json").read_text())
    source = ROOT / recipe["initial_weights"]
    provenance = json.loads((OLD / "initial_checkpoint.json").read_text())
    source_sha = provenance["sha256"]
    assert file_sha(source) == source_sha == file_sha(OLD / "parent_best_backup.pt")
    verified = ROOT / provenance["verification_file"]
    assert file_sha(verified) == provenance["verification_sha256"]
    protected = protection()
    assert protected == json.loads((ROOT / "runs/detection_train_20261024_04_launcher/protected_before.json").read_text())
    for filename in ("parent_best_backup.pt", "initial_checkpoint.json", "baseline_1280.json", "diagnostics_1280.json"):
        shutil.copy2(OLD / filename, OUTPUT / filename)
    pipeline.PREP = OUTPUT
    pipeline.phase("diagnostic_runtime_check")
    model = YOLO(str(source), task="detect")
    model.model.to("cuda:0").float().eval().requires_grad_(False)
    assert all(bool(torch.isfinite(t).all()) for t in model.model.state_dict().values())
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = model.model(torch.zeros((2, 3, 1280, 1280), device="cuda:0"))
        raw = result[0] if isinstance(result, tuple) else result
        torch.cuda.synchronize()
        assert list(raw.shape) == [2, 16, 33600] and bool(torch.isfinite(raw).all())
    names = provenance_names = [model.names[i] for i in range(12)]
    assert names == json.loads((ROOT / "detection_policy.json").read_text())["names"]
    save_json(OUTPUT / "runtime_verification.json", {"passed": True, "training_started": False,
        "protected_files_unchanged": protection() == protected, "batch": 2, "imgsz": 1280,
        "isolated_settings_file": str(settings.file), "checkpoint_sha256": source_sha,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__, "names": names,
        "gpu": torch.cuda.get_device_name(0), "gpu_peak_forward_allocated_bytes": torch.cuda.max_memory_allocated(),
        "amp": False, "cuda_launch_blocking": True, "optimizer_steps": 0})
    del model, result, raw
    gc.collect()
    torch.cuda.empty_cache()
    baseline = json.loads((OUTPUT / "baseline_1280.json").read_text())
    assert baseline["checkpoint_sha256"] == source_sha and baseline["images"] == 17675
    save_json(OUTPUT / "analysis.json", {"passed": True, "checkpoint_sha256": source_sha,
        "resolutions": [1280], "precision": "GPU FP32", "baseline": "baseline_1280.json",
        "baseline_reused_from": str(OLD.relative_to(ROOT) / "baseline_1280.json"),
        "baseline_reused_sha256": file_sha(OLD / "baseline_1280.json"),
        "reason": "Same unchanged initial checkpoint and matched FP32 validation settings; no need to repeat its full baseline.",
        "hypothesis": "Resolution adaptation, with one diagnostic retry using FP32, smaller batches, no loader subprocesses and synchronous CUDA error reporting.",
        "root_cause_established": False, "official_test_used": False})
    recipe.update(batch=2, amp=False, workers=0,
        initial_weights_metadata=(OUTPUT / "initial_checkpoint.json").relative_to(ROOT).as_posix(),
        runtime_verification=(OUTPUT / "runtime_verification.json").relative_to(ROOT).as_posix(),
        baseline_analysis=(OUTPUT / "analysis.json").relative_to(ROOT).as_posix(),
        baseline_analysis_sha256=file_sha(OUTPUT / "analysis.json"))
    recipe["notes"].extend(["First pilot failed with an unknown CUDA error after 238 batches / 9 optimizer steps and zero saved epochs; all evidence retained.",
        "This is one changed diagnostic attempt from the unchanged parent, not an optimizer resume or a repeated identical run.",
        "FP32, batch 2, workers 0 and CUDA_LAUNCH_BLOCKING=1 reduce concurrency/AMP factors and improve error localization; they are not proof of the root cause."])
    path = OUTPUT / "recipe.json"
    save_json(path, recipe)
    ready = readiness(recipe_path=path, inspect_files=False)
    save_json(OUTPUT / "metadata_readiness.json", ready)
    assert ready["ready_for_explicit_training_start"], ready["waiting_for"]
    pipeline.phase("launching_diagnostic_training", pilot=PILOT)
    subprocess.run([sys.executable, str(ROOT / "setup/launch_detection_training.py"), "--run-name", PILOT,
        "--recipe", str(path)], cwd=ROOT, check=True)
    pipeline.phase("diagnostic_training", pilot=PILOT)
    owner = ROOT / "runs" / (PILOT + "_launcher")
    record = json.loads((owner / "owner.json").read_text())
    while not (owner / "exit.json").exists():
        proc = Path("/proc") / str(record["pid"]) / "stat"
        if not proc.exists() or proc.read_text().split(") ", 1)[1].split()[19] != record["linux_process_start_ticks"]:
            raise RuntimeError("Owned diagnostic trainer disappeared; preserve evidence without further automatic retries")
        time.sleep(20)
    assert json.loads((owner / "exit.json").read_text())["returncode"] == 0
    pipeline.phase("matched_final_1280")
    final = pipeline.paired_validation(ROOT / "runs" / PILOT / "weights/best.pt", OUTPUT / "matched_final_1280", names)
    deltas = {key: 100 * (final["metrics"][key] - value) for key, value in baseline["metrics"].items()}
    result = {"passed": protection() == protected and file_sha(source) == source_sha,
        "pilot": PILOT, "scope": "Matched full 17675-image GPU FP32 validation at 1280; development only",
        "baseline": baseline, "final": final, "deltas_percentage_points": deltas,
        "class_ap50_deltas_percentage_points": {name: 100 * (final["per_class"][name]["ap50"] - baseline["per_class"][name]["ap50"]) for name in names},
        "parent_best_preserved": file_sha(OUTPUT / "parent_best_backup.pt") == source_sha,
        "automatically_deployed": False, "official_test_used": False,
        "checked_utc": datetime.now(timezone.utc).isoformat()}
    assert result["passed"] and result["parent_best_preserved"]
    save_json(OUTPUT / "matched_pilot_result.json", result)
    pipeline.phase("completed", measured_gain=bool(deltas["metrics/mAP50(B)"] > 0 and deltas["metrics/mAP50-95(B)"] > 0))


def launch():
    assert sys.platform == "linux"
    assert not (ROOT / "runs" / PILOT).exists()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(arg.endswith(b"/train_detection.py") for arg in args) and b"--start-training" in args:
            raise RuntimeError("Preserve the active trainer")
    OUTPUT.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), OUTPUT / "pipeline_source.py")
    shutil.copy2(ROOT / "setup/detection_resolution_pilot.py", OUTPUT / "shared_pipeline_source.py")
    with (OUTPUT / "console.log").open("xb", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(OUTPUT / "owner.json", {"pid": child.pid,
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "linux_process_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split(") ", 1)[1].split()[19],
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot": PILOT,
        "authorization": "User authorized continued measured training and bounded recovery; monitoring deadline does not stop training."})
    print(json.dumps({"launched": True, "pid": child.pid, "pilot": PILOT}), flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        code = 1
        try:
            execute()
            code = 0
        finally:
            save_json(OUTPUT / "exit.json", {"returncode": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        launch()
