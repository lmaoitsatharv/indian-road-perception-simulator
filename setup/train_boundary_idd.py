"""Bounded matched continuation of trial13: ordinary or boundary-weighted loss."""
import argparse
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import tensorflow as tf
from idd_data import make_dataset, read_pairs
from idd_experiments import experiment_dataset, mixed_model_from_checkpoint
from idd_model import BCEDiceLoss, RoadIoU, RoadDice
from idd_boundary import BoundaryIoU, BoundaryWeightedBCEDice, inner_boundary
from train_idd import TrainingProgress
from boundary_metrics import boundary
from review_boundary_masks import atomic, sha

SESSION = ROOT / "runs/boundary_20261003"
INITIAL = ROOT / "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras"
EXPECTED_MANIFEST = "512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90"
EXPECTED_INITIAL = "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"


def setup_gpu():
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError("Use the verified WSL RTX4060 TensorFlow environment")
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    return tf.config.experimental.get_device_details(devices[0])


def model_for(alpha):
    model = mixed_model_from_checkpoint(ROOT, INITIAL, first_block=13, freeze_decoder_batchnorm=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6, clipnorm=1.),
                  loss=BoundaryWeightedBCEDice(alpha=alpha),
                  metrics=[RoadIoU(), RoadDice(), BoundaryIoU()], jit_compile=False)
    return model


def atomic_model(model, path):
    temp = path.with_name(path.stem + "_writing.keras")
    model.save(temp)
    temp.replace(path)


class SelectedCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, run, road_floor, timeout_seconds=2700):
        super().__init__()
        self.run, self.road_floor = run, road_floor
        self.best, self.best_road, self.best_epoch = -1., -1., 0
        self.started, self.timeout = time.monotonic(), timeout_seconds
        self.stop_reason = None

    def on_train_batch_end(self, batch, logs=None):
        if (SESSION / "STOP.json").exists() or time.monotonic() - self.started > self.timeout:
            self.stop_reason = "requested_stop_or_45_minute_run_bound"
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        # Never select a checkpoint from an incomplete validation pass.
        if self.stop_reason:
            return
        road, score = float(logs["val_road_iou"]), float(logs["val_boundary_iou_5px"])
        if not all(math.isfinite(float(v)) for v in logs.values()):
            raise RuntimeError("Nonfinite training metrics")
        if road > self.best_road:
            atomic_model(self.model, self.run / "best_road.keras")
            self.best_road = road
        if road >= self.road_floor and score > self.best:
            atomic_model(self.model, self.run / "best.keras")
            self.best, self.best_epoch = score, epoch + 1
            atomic(self.run / "best_checkpoint.json", dict(epoch=epoch + 1, boundary_iou_5px=score,
                   road_iou=road, road_dice=float(logs["val_road_dice"]), sha256=sha(self.run / "best.keras")))
        memory = tf.config.experimental.get_memory_info("GPU:0")
        atomic(self.run / "memory.json", {key + "_mib": value / 1024**2 for key, value in memory.items()})


def smoke():
    setup_gpu()
    # Independent CPU and TensorFlow implementations must agree, including frame padding.
    rectangle = np.zeros((32, 32), np.float32)
    rectangle[8:25, 5:26] = 1
    edge = rectangle.copy()
    edge[-5:, :10] = 1
    for image in (rectangle, edge, np.ones((32, 32), np.float32)):
        tb = inner_boundary(image[None, ..., None], 5).numpy()[0, ..., 0] > .5
        np.testing.assert_array_equal(tb, boundary(image.astype(bool), 5))
    truth = tf.constant(rectangle[None, ..., None])
    pred = truth * .8 + .1
    np.testing.assert_allclose(BoundaryWeightedBCEDice(alpha=0)(truth, pred).numpy(),
                               BCEDiceLoss()(truth, pred).numpy(), atol=1e-6)
    # Loss gradients are finite and custom loss survives .keras serialization.
    with tf.GradientTape() as tape:
        tape.watch(pred)
        value = BoundaryWeightedBCEDice()(truth, pred)
    assert np.isfinite(tape.gradient(value, pred).numpy()).all()
    pairs = read_pairs(ROOT / "dataset/splits")
    batch = next(iter(make_dataset(pairs["train"][:4], batch_size=4)))
    assert set(np.unique(batch[1].numpy())) == {0, 1}
    assert float(tf.reduce_min(batch[0])) >= -1 and float(tf.reduce_max(batch[0])) <= 1
    model = model_for(3.)
    before = model.get_layer("full_resolution_refine").kernel.numpy().copy()
    result = model.train_on_batch(*batch, return_dict=True)
    assert all(math.isfinite(float(v)) for v in result.values())
    assert not np.array_equal(before, model.get_layer("full_resolution_refine").kernel.numpy())
    path = SESSION / "smoke_model.keras"
    atomic_model(model, path)
    loaded = tf.keras.models.load_model(path)
    assert isinstance(loaded.loss, BoundaryWeightedBCEDice)
    np.testing.assert_allclose(model(batch[0], training=False).numpy(),
                               loaded(batch[0], training=False).numpy(), atol=1e-6, rtol=1e-5)
    atomic(SESSION / "smoke_result.json", dict(status="passed", independent_boundary_implementation_matches=True,
           alpha_zero_matches_old_loss=True, gradients_finite=True, save_reload_verified=True,
           decoder_update_verified=True, masks_binary=True, normalization_once=True, test_images_used=False))
    print("Boundary training verification passed; smoke model is never used to initialize trials", flush=True)


def train(name, alpha):
    run = SESSION / name
    run.mkdir(parents=True, exist_ok=False)
    initial_hash, manifest_hash = sha(INITIAL), sha(ROOT / "dataset/splits/pair_splits.csv")
    assert initial_hash == EXPECTED_INITIAL and manifest_hash == EXPECTED_MANIFEST
    gpu = setup_gpu()
    pairs = read_pairs(ROOT / "dataset/splits")
    assert {k: len(v) for k, v in pairs.items()} == dict(train=5507, val=687, test=689)
    training = experiment_dataset(pairs["train"], batch_size=16, seed=42, profile="shadow")
    validation = make_dataset(pairs["val"], batch_size=16, training=False)
    model = model_for(alpha)
    # BatchNorm kernels/statistics are captured for an actual end-of-run check.
    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    bn_layers += [l for l in model.get_layer("mobilenetv2_encoder").layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    before_bn = [l.get_weights() for l in bn_layers]
    before_conv = model.get_layer("full_resolution_refine").kernel.numpy().copy()
    reference = json.loads((SESSION / "reference_trial13/metrics.json").read_text())
    road_floor = reference["global"]["road_iou"] - .002
    source = run / "source"
    source.mkdir()
    for path in [ROOT / f for f in ("idd_data.py", "idd_model.py", "idd_experiments.py", "idd_boundary.py", "train_idd.py")] + [Path(__file__), ROOT / "setup/boundary_metrics.py"]:
        shutil.copy2(path, source / path.name)
    config = dict(name=name, alpha=alpha, loss="normalized weighted BCE + existing soft Dice",
                  band_radius_px=3, band_weight=1+alpha, resolution=[512, 512], epochs_maximum=6,
                  batch_size=16, seed=42, learning_rate=3e-6, lr_schedule="constant for both matched runs",
                  initial_checkpoint=str(INITIAL), initial_checkpoint_sha256=initial_hash,
                  manifest_sha256=manifest_hash, split_counts={k: len(v) for k, v in pairs.items()},
                  precision="mixed_float16, float32 head and loss, Adam automatic loss scaling",
                  first_encoder_block=13, augmentation="same shadow profile as trial13",
                  encoder_and_decoder_batchnorm="all fixed, inference mode",
                  checkpoint_selection="largest global validation Boundary IoU at 5px subject to road IoU guard",
                  road_iou_guard_minimum=road_floor, guard_maximum_drop_pp=.2, threshold=.5,
                  test_set_reserved=True, python=platform.python_version(), tensorflow=tf.__version__, keras=tf.keras.__version__,
                  gpu=gpu, parameters=model.count_params(), started_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={p.name: sha(p) for p in source.iterdir()},
                  boundary_metric_reference="https://github.com/bowenc0221/boundary-iou-api",
                  comparison_plan="ordinary loss and boundary loss both restart from trial13 with identical seed/config; fixed 6 epochs")
    atomic(run / "run_config.json", config)
    selected = SelectedCheckpoints(run, road_floor)
    state = dict(status="running", test_set_evaluated=False, started_at_utc=config["started_at_utc"])
    atomic(run / "run_status.json", state)
    started = time.perf_counter()
    try:
        history = model.fit(training, validation_data=validation, epochs=6, verbose=2, callbacks=[
            selected, tf.keras.callbacks.CSVLogger(str(run / "history.csv")),
            tf.keras.callbacks.TerminateOnNaN(), TrainingProgress(run, math.ceil(5507/16), math.ceil(687/16))])
        assert selected.best_epoch > 0, "No completed checkpoint satisfying region guard; preserve evidence and retain trial13"
        atomic(run / "history.json", history.history)
        atomic_model(model, run / "last.keras")
        assert all(np.array_equal(a, b) for weights, layer in zip(before_bn, bn_layers) for a, b in zip(weights, layer.get_weights()))
        assert not np.array_equal(before_conv, model.get_layer("full_resolution_refine").kernel.numpy())
        state.update(status="completed", epochs_completed=len(history.epoch), selected_epoch=selected.best_epoch,
                     best_boundary_iou_5px=selected.best, best_road_iou=selected.best_road,
                     batchnorm_unchanged=True, decoder_weights_updated=True, stop_reason=selected.stop_reason or "six_epoch_budget")
    except BaseException as error:
        state.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        state.update(elapsed_seconds=time.perf_counter()-started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     initial_checkpoint_unchanged=sha(INITIAL)==initial_hash,
                     manifest_unchanged=sha(ROOT / "dataset/splits/pair_splits.csv")==manifest_hash)
        atomic(run / "run_status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name", choices=["control", "boundary_weight4"])
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        train(args.name, 0. if args.name == "control" else 3.)
