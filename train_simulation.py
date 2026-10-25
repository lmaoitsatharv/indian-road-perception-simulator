"""Fresh, isolated simulation training. Never opens test image contents.

Requires the successful GPU/math/serialization verification report. Run on the
existing WSL GPU environment; original IDD trainers/checkpoints are read-only.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import tensorflow as tf

from idd_model import build_unet, WEIGHTS_NAME
from simulation_data import make_dataset, read_pairs
from simulation_model import compile_simulation_model, pack_targets
from setup.prepare_simulation_dataset import protected, save, sha

ROOT = Path(__file__).resolve().parent
MANIFEST_SHA = "c1b9ebbc1ba4768de91d413cb03170bdd5ab8f0a5de1dd3bc6ef344e98675e92"
SOURCES = ("simulation_model.py", "simulation_data.py", "train_simulation.py",
           "idd_model.py", "idd_data.py", "idd_experiments.py",
           "setup/verify_simulation_training.py")


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def utc():
    return datetime.now(timezone.utc).isoformat()


def configure_gpu():
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("A working TensorFlow GPU is required for simulation training")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    result = []
    for device in devices:
        details = tf.config.experimental.get_device_details(device)
        result.append({"name": device.name, "device_name": str(details.get("device_name", "")),
                       "compute_capability": [int(x) for x in details.get("compute_capability", ())]})
    return result


def dataset_checks(dataset):
    if sha(dataset / "splits/pair_splits.csv") != MANIFEST_SHA:
        raise RuntimeError("Simulation split manifest changed since preprocessing")
    report = json.loads((dataset / "completion_status.json").read_text())
    if report["status"] != "completed":
        raise RuntimeError("Dataset preparation is not complete")
    before = protected()
    if before != json.loads((dataset / "plan.json").read_text())["protected_sha256"]:
        raise RuntimeError("Protected real-model files differ from the saved dataset protocol")
    pairs = read_pairs(dataset / "splits")
    counts = {split: len(records) for split, records in pairs.items()}
    if counts != {"train": 4000, "val": 500, "test": 500}:
        raise RuntimeError(f"Unexpected simulation counts: {counts}")
    return pairs, before


class Progress(tf.keras.callbacks.Callback):
    def __init__(self, run_dir, batches, validation_batches):
        super().__init__()
        self.path = run_dir / "progress.json"
        self.batches, self.validation_batches = batches, validation_batches
        self.started, self.epoch = time.perf_counter(), 0

    def write(self, phase, batch=0, total=0, logs=None):
        save(self.path, {"phase": phase, "epoch": self.epoch,
             "epochs_maximum": self.params.get("epochs"), "batch": batch,
             "batches_in_phase": total, "updated_utc": utc(),
             "elapsed_seconds": round(time.perf_counter() - self.started, 2),
             "metrics": {k: float(v) for k, v in (logs or {}).items()}})

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.write("training", total=self.batches)
        print(f"Starting simulation epoch {self.epoch}/{self.params['epochs']}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 20 == 0 or batch + 1 == self.batches:
            self.write("training", batch + 1, self.batches, logs)

    def on_test_begin(self, logs=None):
        self.write("validation", total=self.validation_batches)

    def on_test_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 20 == 0 or batch + 1 == self.validation_batches:
            self.write("validation", batch + 1, self.validation_batches, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.write("epoch_completed", logs=logs)
        print(f"Simulation epoch {self.epoch}: validation IoU={logs['val_road_iou']:.5f}, "
              f"Dice={logs['val_road_dice']:.5f}", flush=True)


def train(dataset, run_dir, verification):
    dataset, run_dir = dataset.resolve(), run_dir.resolve()
    # All outputs stay under the separate simulation run namespace.
    if run_dir.parent != ROOT / "runs" or not run_dir.name.startswith("simulation_"):
        raise ValueError("Use a fresh runs/simulation_* directory")
    if run_dir.exists():
        raise FileExistsError(f"Preserve existing runs; directory already exists: {run_dir}")
    gpu = configure_gpu()
    pairs, real_before = dataset_checks(dataset)
    verified = json.loads(verification.read_text())
    hashes = source_hashes()
    if verified["status"] != "passed" or verified["source_sha256"] != hashes:
        raise RuntimeError("The current trainer has not passed its verification gate")
    if verified["manifest_sha256"] != MANIFEST_SHA or verified["gpu_batch_size"] != 4:
        raise RuntimeError("Verification belongs to a different dataset or batch size")
    recipe = json.loads((dataset / "training_recipe.json").read_text())
    phase = recipe["frozen_phase"]
    if (recipe["seed"], phase["batch_size"], phase["learning_rate"], phase["maximum_epochs"]) != (42, 4, .001, 20):
        raise RuntimeError("The verified initial training recipe changed")
    weights = ROOT / "models/cache/weights" / WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(weights)
    run_dir.mkdir()
    (run_dir / "source").mkdir()
    for name in SOURCES:
        destination = run_dir / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(verification, run_dir / "verification.json")
    shutil.copy2(dataset / "training_recipe.json", run_dir / "training_recipe.json")
    config = {"status": "starting", "model_domain": "simulation_only", "phase": "frozen",
              "started_utc": utc(), "pid": os.getpid(), "dataset": str(dataset),
              "manifest_sha256": MANIFEST_SHA, "split_counts": {s: len(p) for s, p in pairs.items()},
              "test_images_read": False, "test_model_evaluation_started": False,
              "recipe": recipe, "source_sha256": hashes, "protected_real_sha256": real_before,
              "encoder_weights_sha256": sha(weights), "initial_checkpoint": None,
              "normalization": "MobileNetV2 once to [-1,1]", "augmentation": "paired horizontal flip",
              "precision": "float32 (same as the original frozen phase)",
              "python": platform.python_version(), "tensorflow": tf.__version__,
              "keras": tf.keras.__version__, "gpu": gpu}
    save(run_dir / "run_config.json", config)
    save(run_dir / "status.json", {"status": "starting", "updated_utc": utc(), "pid": os.getpid()})
    try:
        model = compile_simulation_model(build_unet(weights), phase["learning_rate"])
        train_ds = make_dataset(pairs["train"], training=True, batch_size=4).map(pack_targets)
        val_ds = make_dataset(pairs["val"], batch_size=4).map(pack_targets)
        progress = Progress(run_dir, math.ceil(len(pairs["train"])/4), math.ceil(len(pairs["val"])/4))
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_road_iou", mode="max", save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_road_iou", mode="max", patience=5, min_delta=.001, restore_best_weights=False),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
            tf.keras.callbacks.TerminateOnNaN(), progress,
        ]
        config["status"] = "running"
        config["training_started"] = True
        config["parameters"] = model.count_params()
        config["trainable_parameters"] = sum(math.prod(v.shape) for v in model.trainable_weights)
        save(run_dir / "run_config.json", config)
        save(run_dir / "status.json", {"status": "running", "updated_utc": utc(), "pid": os.getpid()})
        history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=2)
        values = {k: [float(x) for x in v] for k, v in history.history.items()}
        if not values.get("val_road_iou") or not all(math.isfinite(x) for v in values.values() for x in v):
            raise RuntimeError("Training produced missing or nonfinite metrics")
        model.save(run_dir / "last.keras")
        save(run_dir / "history.json", values)
        best = max(range(len(values["val_road_iou"])), key=lambda i: values["val_road_iou"][i])
        config.update(status="completed", completed_utc=utc(), epochs_completed=len(values["loss"]),
                      best_epoch=best + 1, best_validation_road_iou=values["val_road_iou"][best],
                      best_validation_road_dice=values["val_road_dice"][best],
                      best_checkpoint_sha256=sha(run_dir / "best.keras"))
        # No test model selection/scoring or deployment is performed by this entrypoint.
        print(json.dumps({k: config[k] for k in ("status", "epochs_completed", "best_epoch", "best_validation_road_iou")}), flush=True)
    except BaseException as error:
        config.update(status="failed", failed_utc=utc(), error=f"{type(error).__name__}: {error}")
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        unchanged = protected() == real_before and sha(dataset / "splits/pair_splits.csv") == MANIFEST_SHA
        config["protected_real_and_manifest_unchanged"] = unchanged
        if not unchanged:
            config.update(status="failed", error="Protected files changed during training")
        save(run_dir / "run_config.json", config)
        save(run_dir / "status.json", {"status": config["status"], "updated_utc": utc(),
                                      "pid": os.getpid(), "protected_real_and_manifest_unchanged": unchanged})
        if not unchanged:
            raise RuntimeError("Protected real files or simulation manifest changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/simulation_india_v5")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    train(args.dataset, args.run_dir, args.verification)
