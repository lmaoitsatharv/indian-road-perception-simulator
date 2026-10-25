"""One bounded overnight trial; retains complete validated checkpoints atomically."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from idd_data import decode_pair, normalize_pair, make_dataset, read_pairs
from idd_experiments import light_augment, experiment_dataset, mixed_model_from_checkpoint
from idd_model import compile_model
from train_idd import TrainingProgress

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_model(model, path):
    temporary = path.with_name(path.stem + "_writing.keras")
    model.save(temporary)
    temporary.replace(path)


class BoundedCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, run, deadline, session):
        super().__init__()
        self.run, self.deadline, self.session = run, deadline, session
        self.best, self.best_epoch, self.best_dice = -1., 0, None
        self.stop_reason = None
        self.last_epoch = 0

    def check(self):
        if (self.session / "STOP.json").exists():
            self.stop_reason = "session_stop_requested"
        elif time.time() >= self.deadline - 120:
            self.stop_reason = "deadline_grace_period"
        else:
            try:
                lease = json.loads((self.session / "monitor_lease.json").read_text(encoding="utf-8-sig"))
                if datetime.fromisoformat(lease["expires_at_utc"]).timestamp() < time.time():
                    self.stop_reason = "monitor_lease_expired"
            except (OSError, ValueError, KeyError):
                self.stop_reason = "monitor_lease_unavailable"
        if self.stop_reason:
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self.check()

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch = epoch + 1
        score = float(logs["val_road_iou"])
        if math.isfinite(score) and score > self.best:
            atomic_model(self.model, self.run / "best.keras")
            self.best, self.best_epoch = score, epoch + 1
            self.best_dice = float(logs["val_road_dice"])
            write_json(self.run / "best_checkpoint.json", {
                "best_epoch": self.best_epoch, "road_iou": self.best, "road_dice": self.best_dice,
                "sha256": sha(self.run / "best.keras")})
        memory = tf.config.experimental.get_memory_info("GPU:0")
        with (self.run / "memory_by_epoch.csv").open("a", newline="") as stream:
            writer = csv.writer(stream)
            if epoch == 0:
                writer.writerow(["epoch", "tensorflow_current_mib", "tensorflow_peak_mib"])
            writer.writerow([epoch + 1, memory["current"] / 1024**2, memory["peak"] / 1024**2])
        self.check()


def smoke(session):
    from PIL import Image, ImageDraw
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("No GPU")
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    pairs = read_pairs(ROOT / "dataset/splits")
    assert {k: len(v) for k, v in pairs.items()} == {"train": 5507, "val": 687, "test": 689}
    # Preview uses training images only. Photometric edits preserve the paired-flipped mask.
    sheet = Image.new("RGB", (3 * 256, 3 * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(pairs["train"][:3]):
        raw, mask = decode_pair(record["image"], record["mask"])
        seed = tf.constant([42, row], tf.int32)
        plain, plain_mask = light_augment(raw, mask, seed, "none")
        augmented, augmented_mask = light_augment(raw, mask, seed, "shadow")
        np.testing.assert_array_equal(plain_mask.numpy(), augmented_mask.numpy())
        again, _ = light_augment(raw, mask, seed, "shadow")
        np.testing.assert_allclose(again.numpy(), augmented.numpy(), atol=3e-5, rtol=1e-5)
        normalized, checked_mask = normalize_pair(augmented, augmented_mask)
        assert float(tf.reduce_min(normalized)) >= -1 and float(tf.reduce_max(normalized)) <= 1
        assert set(np.unique(checked_mask.numpy())) == {0, 1}
        panels = [np.uint8(np.clip(plain.numpy(), 0, 255)),
                  np.uint8(np.clip(augmented.numpy(), 0, 255)),
                  np.repeat(np.uint8(augmented_mask.numpy() * 255), 3, axis=2)]
        for col, panel in enumerate(panels):
            sheet.paste(Image.fromarray(panel).resize((256, 256)), (col * 256, row * 280 + 24))
            draw.text((col * 256 + 5, row * 280 + 5), ["Paired flip", "Lighting variation", "Unchanged paired mask"][col], fill="black")
    sheet.save(session / "augmentation_preview.png")
    batch = next(iter(make_dataset(pairs["train"], batch_size=8)))
    model = mixed_model_from_checkpoint(ROOT, ROOT / "runs/frozen_01/best.keras", first_block=13)
    compile_model(model, learning_rate=1e-5)
    before = model.get_layer("mobilenetv2_encoder").get_layer("block_16_project").kernel.numpy().copy()
    for _ in range(3):
        values = model.train_on_batch(*batch, return_dict=True)
    assert all(np.isfinite(float(v)) for v in values.values())
    assert not np.array_equal(before, model.get_layer("mobilenetv2_encoder").get_layer("block_16_project").kernel.numpy())
    temporary = session / "smoke_model.keras"
    atomic_model(model, temporary)
    loaded = tf.keras.models.load_model(temporary, compile=False)
    @tf.function
    def predict(m, images):
        return m(images, training=False)
    np.testing.assert_allclose(predict(model, batch[0]).numpy(), predict(loaded, batch[0]).numpy(), atol=1e-6, rtol=1e-5)
    assert loaded.output.dtype == "float32"
    write_json(session / "smoke_result.json", {"status": "passed", "paired_masks_preserved": True,
               "normalization_once": True, "mixed_save_reload_verified": True,
               "fine_tuning_updates_verified": True, "test_images_used": False})
    print("Overnight preparation verified", flush=True)


def train(spec_path, session):
    spec = json.loads(spec_path.read_text())
    run = ROOT / spec["run_dir"]
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    checkpoint = ROOT / spec["initial_checkpoint"]
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    manifest_hash, initial_hash = sha(manifest), sha(checkpoint)
    if manifest_hash != "512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90":
        raise RuntimeError("Split changed")
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError("Expected verified GPU")
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
    tf.keras.utils.set_random_seed(spec["seed"])
    tf.config.experimental.enable_op_determinism()
    pairs = read_pairs(ROOT / "dataset/splits")
    counts = {k: len(v) for k, v in pairs.items()}
    assert counts == {"train": 5507, "val": 687, "test": 689}
    training = experiment_dataset(pairs["train"], batch_size=spec["batch_size"], seed=spec["seed"], profile=spec["augmentation"])
    validation = make_dataset(pairs["val"], batch_size=spec["batch_size"])
    freeze_decoder_bn = spec.get("freeze_decoder_batchnorm", False)
    model = mixed_model_from_checkpoint(ROOT, checkpoint, spec["first_encoder_block"], freeze_decoder_bn)
    decoder_bn = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    initial_decoder_bn = [[value.numpy().copy() for value in layer.weights] for layer in decoder_bn]
    compile_model(model, learning_rate=spec["learning_rate"])
    source = run / "source"
    source.mkdir()
    for name in ["overnight_idd_worker.py", "idd_experiments.py", "idd_model.py", "idd_data.py", "train_idd.py", "review_idd_validation.py", "overnight_idd_controller.py", "overnight_idd_continuation.py"]:
        shutil.copy2(ROOT / name, source / name)
    config = {**spec, "phase": "fine" if spec["first_encoder_block"] is not None else "frozen",
              "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": initial_hash,
              "manifest_sha256": manifest_hash, "epochs_maximum": spec["epochs"], "split_counts": counts,
              "gpu_details": [tf.config.experimental.get_device_details(d) for d in devices],
              "source_sha256": {p.name: sha(p) for p in source.iterdir()},
              "precision": "mixed_float16 with float32 probability head/loss and loss scaling",
              "encoder_batchnorm_mode": "frozen and inference",
              "decoder_batchnorm_mode": "frozen and inference" if freeze_decoder_bn else "trainable",
              "decoder_batchnorm_layers": len(decoder_bn), "test_set_reserved": True,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "run_config.json", config)
    state = {"status": "running", "started_at_utc": config["started_at_utc"], "test_set_evaluated": False}
    write_json(run / "run_status.json", state)
    bounded = BoundedCheckpoint(run, spec["deadline_epoch"], session)
    callbacks = [bounded,
                 tf.keras.callbacks.CSVLogger(str(run / "history.csv")),
                 tf.keras.callbacks.EarlyStopping(monitor="val_road_iou", mode="max", patience=4, min_delta=.0005),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-7),
                 tf.keras.callbacks.TerminateOnNaN(),
                 TrainingProgress(run, math.ceil(5507 / spec["batch_size"]), math.ceil(687 / spec["batch_size"]))]
    started = time.perf_counter()
    try:
        history = model.fit(training, validation_data=validation, epochs=spec["epochs"], callbacks=callbacks, verbose=2)
        if any(not np.isfinite(v) for values in history.history.values() for v in values):
            raise RuntimeError("Non-finite training history")
        decoder_bn_unchanged = all(np.array_equal(before, after.numpy())
                                   for values, layer in zip(initial_decoder_bn, decoder_bn)
                                   for before, after in zip(values, layer.weights))
        if freeze_decoder_bn and not decoder_bn_unchanged:
            raise RuntimeError("Frozen decoder BatchNorm values changed")
        write_json(run / "history.json", history.history)
        if bounded.best_epoch == 0:
            raise RuntimeError("No complete validation checkpoint")
        state.update(status="completed", epochs_completed=len(history.epoch), best_epoch=bounded.best_epoch,
                     best_validation_road_iou=bounded.best, best_validation_road_dice=bounded.best_dice,
                     decoder_batchnorm_unchanged=decoder_bn_unchanged,
                     stop_reason=bounded.stop_reason or "epoch_budget_or_early_stopping")
    except BaseException as error:
        state.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state.update(elapsed_seconds=time.perf_counter() - started,
                     finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     manifest_unchanged=sha(manifest) == manifest_hash,
                     initial_checkpoint_unchanged=sha(checkpoint) == initial_hash)
        write_json(run / "run_status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.session)
    else:
        train(args.spec, args.session)
