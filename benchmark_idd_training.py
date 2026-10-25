"""Disposable GPU capacity benchmark; uses train images, never saves model updates."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from idd_data import make_dataset, read_pairs
from idd_model import build_unet, compile_model, enable_last_encoder_blocks

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(args):
    destination = ROOT / "setup" / f"benchmark_{args.phase}_{args.precision}_b{args.batch_size}.json"
    checkpoint = ROOT / "runs/frozen_01/best.keras"
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    before = {str(path): digest(path) for path in (checkpoint, manifest)}
    report = {"phase": args.phase, "precision": args.precision, "batch_size": args.batch_size,
              "status": "running", "gpu_budget_mib": 6144, "updates_saved": False,
              "test_images_used": False, "timing_scope": "Repeated real train batch already in RAM; excludes decoding, graph warmup, validation and checkpoint writes. Not an epoch estimate."}
    try:
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise RuntimeError(f"Expected one GPU, saw {devices}")
        tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        pairs = read_pairs(ROOT / "dataset/splits")
        batch = next(iter(make_dataset(pairs["train"], training=False, batch_size=args.batch_size)))
        report["train_ids"] = [row["id"] for row in pairs["train"][:args.batch_size]]
        with tf.device("/CPU:0"):
            saved = tf.keras.models.load_model(checkpoint, compile=False)
            weights = saved.get_weights()
        del saved
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy(args.precision)
        with tf.device("/GPU:0"):
            model = build_unet(ROOT / "models/cache/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
            if args.precision == "mixed_float16":
                # Build a float32 probability head; casting an already-computed
                # float16 sigmoid would not recover lost probability precision.
                config = model.layers[-1].get_config()
                config["dtype"] = "float32"
                head = tf.keras.layers.Conv2D.from_config(config)
                model = tf.keras.Model(model.input, head(model.layers[-2].output), name=model.name)
            model.set_weights(weights)
            del weights
            if args.phase == "fine":
                enable_last_encoder_blocks(model)
            compile_model(model, learning_rate=1e-5 if args.phase == "fine" else 1e-3)
            decoder_before = model.get_layer("road_probability").kernel.numpy().copy()
            encoder = model.get_layer("mobilenetv2_encoder")
            encoder_before = encoder.get_layer("block_16_project").kernel.numpy().copy()
            prediction = model(batch[0], training=False)
            assert "GPU:0" in prediction.device
            assert prediction.dtype == tf.float32
            report["prediction_device"] = prediction.device
            report["optimizer"] = type(model.optimizer).__name__
            for _ in range(3):
                warmup = model.train_on_batch(*batch, return_dict=True)
            if not all(np.isfinite(float(value)) for value in warmup.values()):
                raise ValueError("Non-finite warmup metric")
            tf.config.experimental.reset_memory_stats("GPU:0")
            times = []
            for _ in range(12):
                started = time.perf_counter()
                values = model.train_on_batch(*batch, return_dict=True)
                times.append(time.perf_counter() - started)
                if not all(np.isfinite(float(value)) for value in values.values()):
                    raise ValueError("Non-finite timed metric")
            memory = tf.config.experimental.get_memory_info("GPU:0")
            decoder_changed = not np.array_equal(decoder_before, model.get_layer("road_probability").kernel.numpy())
            encoder_changed = not np.array_equal(encoder_before, encoder.get_layer("block_16_project").kernel.numpy())
            if not decoder_changed or encoder_changed != (args.phase == "fine"):
                raise ValueError("Expected disposable weight updates did not occur")
        report.update(status="passed", timed_steps=len(times),
                      median_step_seconds=float(np.median(times)),
                      images_per_second=args.batch_size / float(np.mean(times)),
                      tensorflow_current_mib=memory["current"] / 1024**2,
                      tensorflow_peak_mib=memory["peak"] / 1024**2,
                      decoder_weights_changed_in_ram=decoder_changed,
                      last_encoder_kernel_changed_in_ram=encoder_changed,
                      finite_metrics=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["checkpoint_and_manifest_unchanged"] = all(digest(Path(path)) == value for path, value in before.items())
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=[4, 8, 16], required=True)
    parser.add_argument("--precision", choices=["float32", "mixed_float16"], required=True)
    parser.add_argument("--phase", choices=["frozen", "fine"], required=True)
    raise SystemExit(main(parser.parse_args()))
