"""Detached CPU validation of a frozen detector while GPU training continues."""
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

PROBE = ROOT / "runs/detection_fast_probe_20261025_01"


def evaluate(output, smoke):
    os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(output / "settings"),
        MPLCONFIGDIR=str(output / "matplotlib"), MPLBACKEND="Agg", YOLO_AUTOINSTALL="False",
        WANDB_DISABLED="true", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for folder in ("settings", "matplotlib"):
        (output / folder).mkdir(exist_ok=True)
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.detect import DetectionValidator
    from setup.verify_detection_runtime import protection
    import ultralytics.data.dataset as dataset_module
    from detection_refinement import parallel_dataset_hash
    dataset_module.get_hash = parallel_dataset_hash
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    settings.update({key: False for key in ("sync", "wandb", "comet", "clearml", "mlflow",
        "neptune", "raytune", "tensorboard", "dvc", "hub") if key in settings})
    dataset = PROBE / "dataset" if smoke else ROOT / "dataset/idd117k_detection_v3"
    expected = 192 if smoke else 17675
    data = dataset / "data.yaml"
    cache = dataset / "labels/val.cache"
    # Both caches already exist from completed project validations. A mismatch
    # must be investigated rather than silently replacing the training cache.
    assert cache.is_file()
    cache_before = file_sha(cache)
    cache_rows = np.load(cache, allow_pickle=True).item()["labels"]
    assert len(cache_rows) == expected
    names = json.loads((ROOT / "detection_policy.json").read_text())["names"]
    expected_targets = sum(len(row["cls"]) for row in cache_rows)
    del cache_rows
    protected = protection()
    checkpoint = output / "checkpoint.pt"
    checkpoint_sha = file_sha(checkpoint)
    manifest_sha = file_sha(dataset / "pairs.jsonl") if not smoke else None
    data_sha = file_sha(data)
    results = {}
    for size in (960, 1280):
        started = time.monotonic()
        last_write = [0.0]
        active_validator = [None]
        class MeasuredValidator(DetectionValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                active_validator[0] = self
        model = YOLO(str(checkpoint), task="detect")
        epoch_metadata = model.ckpt.get("epoch")
        epoch = None if epoch_metadata is None or int(epoch_metadata) < 0 else int(epoch_metadata) + 1
        def progress(validator):
            now = time.monotonic()
            if now - last_write[0] >= 15:
                save_json(output / "progress.json", {"status": "validating", "resolution": size,
                    "images_seen": int(getattr(validator, "seen", 0)), "images_expected": expected,
                    "completed_resolutions": list(results), "checked_utc": datetime.now(timezone.utc).isoformat(),
                    "cpu_only": True, "training_started": False})
                last_write[0] = now
        model.add_callback("on_val_batch_end", progress)
        metrics = model.val(data=str(data), validator=MeasuredValidator, device="cpu", imgsz=size, batch=1, workers=0,
            half=False, conf=.001, iou=.7, max_det=300, augment=False, plots=False,
            save_json=False, project=str(output), name="validation_" + str(size), verbose=False)
        validator = active_validator[0]
        assert validator.seen == expected and int(metrics.nt_per_class.sum()) == expected_targets
        classes = {}
        for position, class_id in enumerate(metrics.box.ap_class_index):
            classes[names[int(class_id)]] = {"ap50": float(metrics.box.ap50[position]),
                "ap50_95": float(metrics.box.ap[position]), "precision": float(metrics.box.p[position]),
                "recall": float(metrics.box.r[position]), "targets": int(metrics.nt_per_class[int(class_id)])}
        assert set(classes) == set(names)
        record = {"resolution": size, "images": expected, "targets": expected_targets,
            "checkpoint_epoch_completed": epoch, "checkpoint_sha256": checkpoint_sha,
            "metrics": {str(key): float(value) for key, value in metrics.results_dict.items()},
            "per_class": classes, "cpu_speed_ms_per_image": metrics.speed,
            "elapsed_seconds": round(time.monotonic() - started, 2)}
        save_json(output / ("validation_" + str(size) + ".json"), record)
        results[str(size)] = record
        active_validator[0] = None
        del model, metrics, validator
        gc.collect()
    invariants = {"protected_files_unchanged": protection() == protected,
        "checkpoint_copy_unchanged": file_sha(checkpoint) == checkpoint_sha,
        "validation_cache_unchanged": file_sha(cache) == cache_before,
        "data_configuration_unchanged": file_sha(data) == data_sha,
        "manifest_unchanged": smoke or file_sha(dataset / "pairs.jsonl") == manifest_sha}
    assert all(invariants.values()), invariants
    report = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "192-image smoke check" if smoke else "full unchanged project validation; development only",
        "results": results, "invariants": invariants, "official_test_used": False,
        "cpu_only": True, "optimizer_steps": 0, "training_started": False, "control_allowed": False,
        "metric_deltas_percentage_points": {key: 100 * (results["1280"]["metrics"][key] - value)
            for key, value in results["960"]["metrics"].items()},
        "per_class_ap50_deltas_percentage_points": {name: 100 * (results["1280"]["per_class"][name]["ap50"]
            - results["960"]["per_class"][name]["ap50"]) for name in names}}
    save_json(output / "comparison.json", report)
    save_json(output / "progress.json", {"status": "completed", "checked_utc": report["checked_utc"],
        "completed_resolutions": [960, 1280], "cpu_only": True, "training_started": False})


def launch(output, smoke):
    assert sys.platform == "linux" and output.parent == ROOT / "runs"
    # Check all tagged workers, including their start tick, before creating a new one.
    for owner in (ROOT / "runs").glob("detection_full_resolution_*/owner.json"):
        record = json.loads(owner.read_text())
        proc = Path("/proc") / str(record["pid"])
        try:
            tick = (proc / "stat").read_text().split(") ", 1)[1].split()[19]
            if tick == record["linux_process_start_ticks"]:
                raise RuntimeError("A full resolution diagnostic is already active")
        except FileNotFoundError:
            pass
    selection = json.loads((PROBE / "selection.json").read_text())
    source = PROBE / "latest_snapshot.pt"
    assert file_sha(source) == selection["latest_snapshot_sha256"]
    output.mkdir(exist_ok=False)
    shutil.copy2(source, output / "checkpoint.pt")
    shutil.copy2(Path(__file__), output / "runner_source.py")
    shutil.copy2(PROBE / "selection.json", output / "checkpoint_provenance.json")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output", str(output)]
    if smoke:
        command.append("--smoke")
    with (output / "console.log").open("xb", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": command, "cpu_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "linux_process_start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()[19],
        "authorization": "User requested measured improvements and continued training through scheduled checks, 25 October 2026"}
    save_json(output / "owner.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        code = 1
        try:
            evaluate(output, args.smoke)
            code = 0
        finally:
            save_json(output / "exit.json", {"returncode": code,
                "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        launch(output, args.smoke)
