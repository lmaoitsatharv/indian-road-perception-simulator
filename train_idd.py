"""Explicit training entry point. Uses the completed loader; reserves test images.

Example on a verified GPU runtime:
python train_idd.py --run-dir runs/frozen_01

CPU training requires --allow-cpu. This file does not start a run when imported.
"""

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from idd_data import make_dataset, read_pairs
from idd_model import build_unet, compile_model, enable_last_encoder_blocks, get_encoder_weights


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TrainingProgress(tf.keras.callbacks.Callback):
    """Save readable progress so a long run can be monitored outside its terminal."""

    def __init__(self, run_dir, train_batches, val_batches):
        super().__init__()
        self.path = Path(run_dir) / "progress.json"
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.started = time.perf_counter()
        self.epoch = 0

    def write(self, phase, batch=0, total=0, logs=None):
        progress = {
            "phase": phase, "epoch": self.epoch,
            "epochs_maximum": self.params.get("epochs"),
            "batch": batch, "batches_in_phase": total,
            "elapsed_seconds": round(time.perf_counter() - self.started, 2),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": {key: float(value) for key, value in (logs or {}).items()},
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.epoch_started = time.perf_counter()
        self.write("training", total=self.train_batches)
        print(f"Starting epoch {self.epoch}/{self.params['epochs']}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 50 == 0 or batch + 1 == self.train_batches:
            self.write("training", batch + 1, self.train_batches, logs)

    def on_test_begin(self, logs=None):
        self.write("validation", total=self.val_batches)

    def on_test_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 50 == 0 or batch + 1 == self.val_batches:
            self.write("validation", batch + 1, self.val_batches, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.write("epoch_completed", logs=logs)
        elapsed = time.perf_counter() - self.epoch_started
        print(f"Epoch {epoch + 1} completed in {elapsed:.1f} seconds; "
              f"validation road IoU={logs['val_road_iou']:.4f}, "
              f"Dice={logs['val_road_dice']:.4f}", flush=True)

    def on_train_end(self, logs=None):
        self.write("training_ended", logs=logs)


def train(splits_dir, initial_model, run_dir, *, epochs=20, batch_size=4, allow_cpu=False, phase="frozen"):
    splits_dir, run_dir = Path(splits_dir).resolve(), Path(run_dir).resolve()
    initial_model = Path(initial_model).resolve() if initial_model is not None else None
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices and not allow_cpu:
        raise RuntimeError("TensorFlow sees no GPU. Use a verified GPU runtime, or explicitly choose CPU with --allow-cpu.")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    if run_dir.exists():
        raise FileExistsError(f"Use a new run directory to preserve previous checkpoints: {run_dir}")
    if initial_model is not None and not initial_model.is_file():
        raise FileNotFoundError(initial_model)
    if phase == "fine" and (initial_model is None or "UNTRAINED" in initial_model.name):
        raise ValueError("Fine-tuning must start from a validation-selected trained checkpoint, not the untrained decoder")

    pairs = read_pairs(splits_dir)
    counts = {split: len(records) for split, records in pairs.items()}
    if counts != {"train": 5507, "val": 687, "test": 689}:
        raise ValueError(f"Unexpected cleaned split counts: {counts}")
    manifest_hash = sha256(splits_dir / "pair_splits.csv")
    setup_config_path = Path(__file__).resolve().parent / "models" / "model_config.json"
    setup_config = json.loads(setup_config_path.read_text(encoding="utf-8"))
    if manifest_hash != setup_config["manifest_sha256"]:
        raise ValueError("The split manifest changed since model verification")
    if phase == "fine":
        previous_config_path = initial_model.parent / "run_config.json"
        previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
        if previous_config["manifest_sha256"] != manifest_hash:
            raise ValueError("Fine-tuning checkpoint belongs to a different split manifest")
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    # Do not construct or iterate a test dataset during training.
    train_ds = make_dataset(pairs["train"], training=True, batch_size=batch_size, seed=42)
    val_ds = make_dataset(pairs["val"], training=False, batch_size=batch_size, seed=42)
    if initial_model is None:
        # The disposable setup checkpoint was removed during project cleanup.
        # Fresh frozen training can reconstruct it from the retained ImageNet weights.
        weights_path = get_encoder_weights(Path(__file__).resolve().parent / "models" / "cache")
        model = build_unet(weights_path)
    else:
        model = tf.keras.models.load_model(initial_model, compile=False)
    if phase == "fine":
        enable_last_encoder_blocks(model)
        learning_rate = 1e-5
    else:
        model.get_layer("mobilenetv2_encoder").trainable = False
        learning_rate = 1e-3
    compile_model(model, learning_rate=learning_rate)

    run_dir.mkdir(parents=True, exist_ok=False)
    source_dir = run_dir / "source"
    source_dir.mkdir()
    for source_name in ("train_idd.py", "idd_data.py", "idd_model.py"):
        shutil.copy2(Path(__file__).resolve().parent / source_name, source_dir / source_name)
    initial_hash = sha256(initial_model) if initial_model is not None else None
    config = {
        "phase": phase, "epochs_maximum": epochs, "batch_size": batch_size,
        "resolution": [512, 512], "seed": 42, "learning_rate": learning_rate,
        "initial_checkpoint": str(initial_model) if initial_model is not None else None,
        "initial_checkpoint_sha256": initial_hash,
        "initialization": "saved_checkpoint" if initial_model is not None else "fresh_decoder_with_cached_ImageNet_encoder",
        "manifest_sha256": manifest_hash, "split_counts": counts,
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__, "gpu_devices": [str(device) for device in gpu_devices],
        "model_and_metric_settings": setup_config,
        "trainable_encoder_layers": [layer.name for layer in model.get_layer("mobilenetv2_encoder").layers if layer.trainable and layer.weights],
        "encoder_batchnorm_mode": "inference; encoder call fixed at training=False",
        "test_set_reserved": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_details": [tf.config.experimental.get_device_details(device) for device in gpu_devices],
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "source_sha256": {path.name: sha256(path) for path in source_dir.iterdir()},
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.keras"), monitor="val_road_iou", mode="max", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_road_iou", mode="max", patience=5, min_delta=0.001,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
        TrainingProgress(run_dir, math.ceil(counts["train"] / batch_size),
                         math.ceil(counts["val"] / batch_size)),
    ]
    started = time.perf_counter()
    state = {"status": "running", "test_set_evaluated": False,
             "started_at_utc": config["started_at_utc"]}
    status_path = run_dir / "run_status.json"
    status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            callbacks=callbacks, verbose=2)
        if any(not math.isfinite(float(value)) for values in history.history.values() for value in values):
            raise RuntimeError("Training produced a non-finite logged loss or metric; inspect history.csv")
        model.save(run_dir / "last.keras")
        (run_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
        best_index = max(range(len(history.epoch)), key=lambda index: history.history["val_road_iou"][index])
        state.update(status="completed", epochs_completed=len(history.epoch),
                     elapsed_seconds=time.perf_counter() - started,
                     best_epoch=history.epoch[best_index] + 1,
                     best_validation_road_iou=float(history.history["val_road_iou"][best_index]),
                     best_validation_road_dice=float(history.history["val_road_dice"][best_index]))
        print(f"Validation-selected checkpoint: {run_dir / 'best.keras'}")
        print("Use validation data for threshold selection. Test data remains reserved for final evaluation.")
        return history
    except BaseException as error:
        state.update(status="interrupted_or_failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        state.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     manifest_unchanged=sha256(splits_dir / "pair_splits.csv") == manifest_hash,
                     initial_checkpoint_unchanged=(sha256(initial_model) == initial_hash) if initial_model is not None else None)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=root / "dataset" / "splits")
    parser.add_argument("--initial-model", type=Path, default=None,
                        help="Saved checkpoint for continuation; omit for a fresh decoder with cached pretrained encoder weights.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--phase", choices=["frozen", "fine"], default="frozen")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    train(args.splits_dir, args.initial_model, args.run_dir, epochs=args.epochs,
          batch_size=args.batch_size, phase=args.phase, allow_cpu=args.allow_cpu)
