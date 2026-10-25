"""CPU-only paired modest-LR experiment, with matched data and augmented inputs."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json


def worker(output):
    os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(output / "settings"),
        MPLCONFIGDIR=str(output / "matplotlib"), MPLBACKEND="Agg", YOLO_AUTOINSTALL="False",
        WANDB_DISABLED="true", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for folder in ("settings", "matplotlib"):
        (output / folder).mkdir(exist_ok=True)
    mem = {k: int(v.strip().split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    assert mem["MemAvailable"] > 3 * 1024**2, "Defer the CPU probe if memory headroom is insufficient"
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
    from detection_refinement import save_full_precision_resume
    from setup.verify_detection_runtime import protection
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if k in settings})
    selected = json.loads((output / "selection.json").read_text())
    names = selected["names"]
    protected = protection()
    source = output / "starting_checkpoint.pt"
    source_sha = file_sha(source)
    assert source_sha == selected["starting_checkpoint_sha256"]
    def publish(phase, **extra):
        save_json(output / "progress.json", {"phase": phase, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_only": True, "main_gpu_training_modified": False, **extra})
    def evaluate(checkpoint, label):
        captured = [None]
        class CapturedValidator(DetectionValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                captured[0] = self
        model = YOLO(str(checkpoint), task="detect")
        metrics = model.val(data=str(output / "dataset/data.yaml"), validator=CapturedValidator,
            imgsz=1280, device="cpu", batch=1, workers=0, half=False, conf=.001, iou=.7,
            max_det=300, augment=False, plots=False, save_json=False, project=str(output), name=label,
            verbose=False)
        assert captured[0].seen == 192 and int(metrics.nt_per_class.sum()) == sum(selected["objects"]["val"].values())
        result = {"metrics": {str(k): float(v) for k, v in metrics.results_dict.items()},
            "per_class": {names[int(c)]: {"ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i])}
                for i, c in enumerate(metrics.box.ap_class_index)}, "images": 192,
            "resolution": 1280, "precision": "CPU FP32", "checkpoint_sha256": file_sha(checkpoint)}
        save_json(output / (label + ".json"), result)
        captured[0] = None
        del model, metrics
        gc.collect()
        return result
    publish("baseline_validation")
    baseline = evaluate(source, "baseline")
    arms = {}
    for label, rate in (("control", .00003), ("higher", .0000375)):
        torch.manual_seed(42)
        stream = hashlib.sha256()
        counter = {"batches": 0, "optimizer_steps": 0, "last_write": 0.0}
        class PairedTrainer(DetectionTrainer):
            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                for key in ("img", "cls", "bboxes", "batch_idx"):
                    array = batch[key].detach().cpu().contiguous().numpy()
                    stream.update(str((key, array.shape, str(array.dtype))).encode())
                    if array.size:
                        stream.update(memoryview(array).cast("B"))
                stream.update(json.dumps(batch["im_file"]).encode())
                return batch
            def save_model(self):
                super().save_model()
                save_full_precision_resume(self, Path(self.save_dir))
        def started(trainer):
            torch.set_num_threads(4)
            def stepped(optimizer, args, kwargs):
                counter["optimizer_steps"] += 1
            trainer.optimizer.register_step_post_hook(stepped)
        def batch_done(trainer):
            counter["batches"] += 1
            values = trainer.loss_items.detach().float().cpu().tolist()
            assert all(math.isfinite(v) for v in values), "Non-finite CPU probe loss"
            if counter["batches"] <= 3 or time.monotonic() - counter["last_write"] > 15:
                publish(label + "_training", learning_rate=rate, batches=counter["batches"], batches_expected=256,
                    optimizer_steps=counter["optimizer_steps"], losses=dict(zip(trainer.loss_names, values)))
                counter["last_write"] = time.monotonic()
        model = YOLO(str(source), task="detect")
        model.add_callback("on_train_start", started)
        model.add_callback("on_train_batch_end", batch_done)
        publish(label + "_training", learning_rate=rate, batches=0, batches_expected=256)
        started_at = time.monotonic()
        model.train(trainer=PairedTrainer, data=str(output / "dataset/data.yaml"), imgsz=1280,
            device="cpu", batch=1, nbs=16, workers=0, amp=False, epochs=1, patience=1,
            optimizer="AdamW", lr0=rate, lrf=1.0, cos_lr=False, warmup_epochs=0.0,
            warmup_bias_lr=0.0, weight_decay=.0005, seed=42, deterministic=True,
            cache=False, mosaic=0.0, close_mosaic=0, scale=.3, conf=.001, iou=.7,
            max_det=300, save=True, save_period=1, plots=False, verbose=False,
            project=str(output), name=label, exist_ok=False)
        assert counter["batches"] == 256 and counter["optimizer_steps"] > 0
        checkpoint = output / label / "weights/last_resume_fp32.pt"
        assert checkpoint.is_file()
        result = {"learning_rate": rate, "augmented_training_stream_sha256": stream.hexdigest(),
            "batches": counter["batches"], "optimizer_steps": counter["optimizer_steps"],
            "seconds": round(time.monotonic() - started_at, 2),
            "checkpoint": checkpoint.relative_to(ROOT).as_posix()}
        del model
        gc.collect()
        publish(label + "_matched_validation")
        result["validation"] = evaluate(checkpoint, label + "_matched")
        save_json(output / (label + "_result.json"), result)
        arms[label] = result
    assert arms["control"]["augmented_training_stream_sha256"] == arms["higher"]["augmented_training_stream_sha256"], "The arms did not receive identical augmented inputs"
    assert arms["control"]["optimizer_steps"] == arms["higher"]["optimizer_steps"]
    control, higher = (arms[key]["validation"] for key in ("control", "higher"))
    deltas = {key: 100 * (higher["metrics"][key] - value) for key, value in control["metrics"].items()}
    class_deltas = {name: 100 * (higher["per_class"][name]["ap50"] - control["per_class"][name]["ap50"]) for name in names}
    promising = deltas["metrics/mAP50-95(B)"] >= .1 and deltas["metrics/mAP50(B)"] >= 0 and class_deltas["person"] >= 0 and class_deltas["animal"] >= 0
    report = {"passed": protection() == protected and file_sha(source) == source_sha,
        "checked_utc": datetime.now(timezone.utc).isoformat(), "scope": "Small paired development screening; not a full validation or independent test result",
        "baseline": baseline, "arms": arms, "higher_minus_control_percentage_points": deltas,
        "class_ap50_deltas_percentage_points": class_deltas,
        "identical_augmented_inputs": True, "identical_optimizer_step_count": True,
        "higher_rate_promising_for_confirmation": bool(promising),
        "decision": "Consider larger confirmation" if promising else "No convincing screened win; retain current rate pending further evidence",
        "main_gpu_training_modified": False, "automatically_promoted": False, "official_test_used": False}
    assert report["passed"]
    save_json(output / "comparison.json", report)
    text = ("# Modest learning-rate comparison\n\n" + report["scope"] + ".\n\n"
        "Both arms start from the same epoch-30 checkpoint and receive identical augmented training inputs. Only the learning rate differs: 0.000030 versus 0.0000375 (+25%).\n\n"
        f"Higher minus control: mAP50 {deltas['metrics/mAP50(B)']:+.3f} points; mAP50-95 {deltas['metrics/mAP50-95(B)']:+.3f} points.\n\n"
        + report["decision"] + ". This does not establish a full-dataset improvement. Current GPU training and its learning rate were not modified.\n\n"
        "The screening cohort uses 256 training selections, 192 stratified validation images, CPU FP32, batch 1 and nominal batch 16. Its batch/coverage differ from the full GPU run; the result needs confirmation before promotion.\n")
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    publish("completed", higher_rate_promising_for_confirmation=bool(promising), result="comparison.json")


def launch(output):
    assert sys.platform == "linux" and output.parent == ROOT / "runs"
    assert (output / "selection.json").is_file() and not (output / "owner.json").exists()
    shutil.copy2(Path(__file__), output / "runner_source.py")
    with (output / "console.log").open("xb", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output", str(output)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(output / "owner.json", {"pid": child.pid, "cpu_only": True,
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "linux_process_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split(") ", 1)[1].split()[19],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User explicitly requested trying a small learning-rate increase, 25 October 2026"})
    print(json.dumps({"launched": True, "pid": child.pid, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        code = 1
        try:
            worker(output)
            code = 0
        finally:
            save_json(output / "exit.json", {"returncode": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        launch(output)
