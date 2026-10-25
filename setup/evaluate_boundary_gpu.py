"""Saved-checkpoint validation exports and batch-one latency; no test images."""
import gc
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import tensorflow as tf
import idd_model
import idd_boundary
from idd_data import make_dataset, read_pairs
from review_boundary_masks import evaluate, atomic, sha

SESSION = ROOT / "runs/boundary_20261003"
MODELS = {"original": ROOT / "runs/frozen_01/best.keras",
          "morning_best": ROOT / "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras",
          "boundary_weight4": SESSION / "boundary_weight4/best.keras"}

# Historical control metrics remain in comparison_results.json and latency.json.
# Its binary was removed during cleanup, so subsequent exports use the retained
# three models. This one-time exporter still refuses existing prediction folders.
if (SESSION / "control/best.keras").is_file():
    MODELS["control"] = SESSION / "control/best.keras"


def main():
    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) == 1
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
    tf.config.experimental.enable_op_determinism()
    pairs = read_pairs(ROOT / "dataset/splits")
    assert {k: len(v) for k, v in pairs.items()} == dict(train=5507, val=687, test=689)
    records = pairs["val"]
    manifest_hash = sha(ROOT / "dataset/splits/pair_splits.csv")
    hashes = {name: sha(path) for name, path in MODELS.items()}
    # Eight spaced validation examples for timing; disk reads are outside timing.
    raw = []
    for r in records[::85][:8]:
        with Image.open(r["image"]) as im:
            raw.append(np.asarray(im.convert("RGB"))[None, ...])
    latency = {}
    architecture = None
    for name, path in MODELS.items():
        print(f"Reload and verify {name}", flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        signature = [(layer.name, type(layer).__name__) for layer in model.layers]
        if architecture is None:
            architecture = signature
            parameter_count = model.count_params()
        assert signature == architecture and model.count_params() == parameter_count

        @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
        def predict(images):
            return model(images, training=False)

        if name in ("control", "boundary_weight4"):
            target = SESSION / name / "predicted_masks"
            target.mkdir(exist_ok=False)
            offset = 0
            for images, masks in make_dataset(records, batch_size=16, training=False):
                probability = predict(images)
                device = probability.device
                values = probability.numpy()[..., 0]
                assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
                assert np.isin(masks.numpy(), [0, 1]).all()
                for j, mask in enumerate(values >= .5):
                    Image.fromarray(mask.astype(np.uint8)).save(target / f"Prediction_{records[offset+j]['id']}.png")
                offset += len(values)
            assert offset == 687 and "GPU:0" in device
            reviewed = evaluate(target, SESSION / name / "validation_review")
            selected = json.loads((SESSION / name / "best_checkpoint.json").read_text())
            for key in ("road_iou", "road_dice", "boundary_iou_5px"):
                assert abs(reviewed["global"][key] - selected[key]) < 1e-12, (key, reviewed["global"][key], selected[key])
            atomic(SESSION / name / "reload_verification.json", dict(status="passed", checkpoint_sha256=sha(path),
                   independently_recomputed_metrics_match_training=True, prediction_device=device,
                   masks_binary=True, test_images_used=False))

        # Same batch-one graph for every model; synchronize by copying output to CPU.
        with tf.device("/GPU:0"):
            resident = [tf.constant(image.astype(np.float32)/127.5-1.) for image in raw]
        with tf.device("/CPU:0"):
            host_raw = [tf.constant(image) for image in raw]
        assert all("CPU:0" in image.device for image in host_raw)
        assert all("GPU:0" in image.device for image in resident)

        @tf.function(input_signature=[tf.TensorSpec([1, 512, 512, 3], tf.uint8)])
        def rgb_to_mask(image):
            with tf.device("/GPU:0"):
                normalized = tf.cast(image, tf.float32)/127.5-1.
                return tf.cast(model(normalized, training=False) >= .5, tf.uint8)

        for i in range(15):
            predict(resident[i % 8]).numpy()
            rgb_to_mask(host_raw[i % 8]).numpy()
        timings = {"resident_normalized_forward_plus_probability_copy_ms": [],
                   "rgb_in_cpu_ram_to_binary_mask_in_cpu_ram_ms": []}
        # Alternating timing paths, fixed image order and 64 synchronized samples each.
        for i in range(64):
            start = time.perf_counter()
            result = predict(resident[i % 8])
            prediction_device = result.device
            result.numpy()
            timings["resident_normalized_forward_plus_probability_copy_ms"].append((time.perf_counter()-start)*1000)
            start = time.perf_counter()
            rgb_to_mask(host_raw[i % 8]).numpy()
            timings["rgb_in_cpu_ram_to_binary_mask_in_cpu_ram_ms"].append((time.perf_counter()-start)*1000)
        assert "GPU:0" in prediction_device
        latency[name] = dict(checkpoint_sha256=hashes[name], parameters=parameter_count,
                            precision_policy=str(model.layers[1].dtype_policy), prediction_device=prediction_device,
                            samples=64, warmup_per_path=15, image_ids=[r["id"] for r in records[::85][:8]],
                            **{key: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                                     "mean": float(np.mean(v)), "samples_ms": v} for key, v in timings.items()})
        atomic(SESSION / "latency.json", dict(status="in_progress", models=latency, test_images_used=False))
        print(f"{name}: RAM-to-mask median {latency[name]['rgb_in_cpu_ram_to_binary_mask_in_cpu_ram_ms']['median']:.2f} ms", flush=True)
        del model, predict, rgb_to_mask, resident, host_raw
        tf.keras.backend.clear_session()
        gc.collect()
    assert hashes == {name: sha(path) for name, path in MODELS.items()}
    assert manifest_hash == sha(ROOT / "dataset/splits/pair_splits.csv")
    atomic(SESSION / "latency.json", dict(status="completed", models=latency, batch_size=1,
           resolution=[512, 512], gpu=tf.config.experimental.get_device_details(devices[0]),
           identical_architecture_and_parameter_count=True, no_boundary_processing_in_inference=True,
           scope="Warm tf.function GPU inference. RAM-to-mask includes normalization, host/device transfer, threshold and sync. Excludes camera capture, tracking, planning, display, cold initialization, disk reads.",
           limitation="One sequential session; thermal/load differences can affect small latency differences. FPS inverse is not full-system FPS.",
           immutable_checkpoints_and_manifest_verified=True, test_images_used=False))


if __name__ == "__main__":
    main()
