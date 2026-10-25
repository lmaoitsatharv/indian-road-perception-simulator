"""One fixed-model test inference pass; export masks, independently verify metrics.

Requires EVALUATION_LOCK.json created before test-content access. No training,
threshold selection, checkpoint writes, model rebuilding or label modifications.
"""
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/test_evaluation_20261003_morning_best"
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import tensorflow as tf
import idd_model
import idd_boundary
from idd_data import make_dataset, read_pairs
from idd_model import BCEDiceLoss, RoadDice, RoadIoU
from idd_boundary import BoundaryIoU
from boundary_metrics import counts, scores, summarize


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf-8")
    temporary.replace(path)


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(model):
    result = hashlib.sha256()
    for value in model.get_weights():
        result.update(value.tobytes())
    return result.hexdigest()


def gpu_sample():
    fields = "name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw"
    result = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=10)
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "returncode": result.returncode,
            "fields": fields, "values": result.stdout.strip(), "error": result.stderr.strip()}


def fixture_checks():
    # Shifted 2x2 squares: TP=2, FP=2, FN=2 -> IoU=1/3, Dice=1/2.
    truth, prediction = np.zeros((32, 32), bool), np.zeros((32, 32), bool)
    truth[8:10, 8:10], prediction[8:10, 9:11] = True, True
    c = counts(truth, prediction)
    assert (c["tp"], c["fp"], c["fn"]) == (2, 2, 2)
    assert scores(c)["road_iou"] == 1/3 and scores(c)["road_dice"] == .5
    result = {}
    for cls, target in ((RoadIoU, 1/3), (RoadDice, .5)):
        metric = cls()
        metric.update_state(truth[None, ..., None], prediction[None, ..., None])
        assert abs(float(metric.result())-target) < 1e-12
        result[cls.__name__] = True
    metric = BoundaryIoU()
    metric.update_state(truth[None, ..., None], truth[None, ..., None])
    assert float(metric.result()) == scores(counts(truth, truth))["boundary_iou_5px"] == 1
    result["perfect_boundary_iou"] = True
    assert scores(counts(np.zeros_like(truth), np.zeros_like(truth)))["road_iou"] == 1
    result["empty_union_convention"] = True
    return result


def bootstrap(rows, lock):
    keys = ("tp", "fp", "fn", "tn", "boundary_intersection_5px", "boundary_union_5px",
            "boundary_intersection_14px", "boundary_union_14px", "contour_precision_matches",
            "contour_pred_pixels", "contour_recall_matches", "contour_true_pixels")
    values = np.array([[r[k] for k in keys] for r in rows], dtype=np.int64)
    rng = np.random.default_rng(lock["bootstrap"]["seed"])
    samples = {k: [] for k in ("road_iou", "road_dice", "road_precision", "road_recall", "boundary_iou_5px", "boundary_iou_14px", "visible_boundary_f1_3px")}
    for _ in range(lock["bootstrap"]["resamples"]):
        total = dict(zip(keys, map(int, values[rng.integers(0, len(rows), len(rows))].sum(axis=0))))
        for key, value in scores(total).items():
            if key in samples:
                samples[key].append(value)
    return {k: {"lower": float(np.percentile(v, 2.5)), "upper": float(np.percentile(v, 97.5))} for k, v in samples.items()}


def main():
    lock = json.loads((OUT / "EVALUATION_LOCK.json").read_text())
    existing = json.loads((OUT / "session_status.json").read_text())
    if existing["status"] != "prepared" or existing.get("test_inference_started"):
        raise RuntimeError("This evaluation is already started/completed; use saved masks and records rather than another test inference pass.")
    assert lock["checkpoint_sha256"] == "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"
    assert lock["threshold"] == .5 and lock["batch_size"] == 16 and lock["training_prohibited"]
    assert sha(ROOT / lock["checkpoint"]) == lock["checkpoint_sha256"]
    assert sha(ROOT / "dataset/splits/pair_splits.csv") == lock["manifest_sha256"]
    for rel, expected in lock["source_sha256"].items():
        assert sha(ROOT / rel) == expected, rel
    source = OUT / "source"
    source.mkdir(exist_ok=False)
    for rel in lock["source_sha256"]:
        destination = source / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    status = {"status": "running", "phase": "preflight", "completed": 0, "total": 689,
              "pid": os.getpid(), "process_group": os.getpgrp(), "test_inference_started": False,
              "training_started": False, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    gpu_samples = []

    def progress(phase, completed=0):
        status.update(phase=phase, completed=completed, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=round(time.perf_counter()-started, 3))
        atomic(OUT / "session_status.json", status)
        if time.perf_counter()-started > 1800:
            raise TimeoutError("Bounded evaluation exceeded 30 minutes; stop this process without changing data or checkpoints.")

    try:
        progress("gpu_setup")
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise RuntimeError("The verified WSL GPU runtime must expose the RTX 4060.")
        tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        gpu_details = tf.config.experimental.get_device_details(devices[0])
        assert "4060" in gpu_details.get("device_name", "")
        gpu_samples.append(gpu_sample())
        fixtures = fixture_checks()
        pairs = read_pairs(ROOT / "dataset/splits")
        assert {k: len(v) for k, v in pairs.items()} == lock["split_counts"]
        records = pairs["test"]
        assert [r["id"] for r in records] == lock["test_ids"]
        with (ROOT / "dataset/splits/pair_splits.csv").open(newline="", encoding="utf-8-sig") as stream:
            manifest = {int(r["id"]): r for r in csv.DictReader(stream)}
        # First test-content access occurs only after the immutable lock exists.
        file_rows = []
        for index, record in enumerate(records):
            image_hash, mask_hash = sha(record["image"]), sha(record["mask"])
            original = manifest[record["id"]]
            assert image_hash == original["image_sha256"] and mask_hash == original["mask_sha256"], record["id"]
            with Image.open(record["image"]) as image:
                assert image.format == "PNG" and image.size == (512, 512)
                mode = image.mode
                image.load()
            with Image.open(record["mask"]) as mask:
                target = np.asarray(mask)
                assert mask.format == "PNG" and target.shape == (512, 512)
                assert target.dtype == np.uint8 and set(np.unique(target)) == {0, 1}, record["id"]
                road_pixels = int(target.sum())
            file_rows.append(dict(id=record["id"], split="test", image_sha256=image_hash, mask_sha256=mask_hash,
                                  image_mode=mode, width=512, height=512, mask_dtype="uint8", mask_values="0;1",
                                  truth_road_pixels=road_pixels))
            if (index+1) % 100 == 0:
                progress("checking_test_files", index+1)
                print(f"File integrity: {index+1}/689", flush=True)
        table(OUT / "test_file_manifest.csv", file_rows)
        atomic(OUT / "data_integrity.json", dict(status="passed", images=689, live_test_hashes_match_original_manifest=True,
               all_shapes_512x512=True, mask_values=[0, 1], empty_truth_masks=0,
               split_ids_disjoint=True, stored_exact_image_hash_overlap=lock["stored_manifest_exact_image_hash_overlap"],
               sequence_metadata_available=False, class_conversion_recovered=False, excluded_test_pairs=0))
        progress("loading_locked_model")
        model = tf.keras.models.load_model(ROOT / lock["checkpoint"], compile=False)
        assert model.input_shape == (None, 512, 512, 3) and model.output_shape == (None, 512, 512, 1)
        assert model.layers[-1].compute_dtype == "float32"
        weights_before = fingerprint(model)
        iou_metric, dice_metric, edge_metric = RoadIoU(), RoadDice(), BoundaryIoU()
        loss = BCEDiceLoss()

        @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32), tf.TensorSpec([None, 512, 512, 1], tf.float32)])
        def predict_and_count(images, masks):
            with tf.device("/GPU:0"):
                probabilities = model(images, training=False)
                iou_metric.update_state(masks, probabilities)
                dice_metric.update_state(masks, probabilities)
                edge_metric.update_state(masks, probabilities)
                return probabilities, loss.call(masks, probabilities)

        # Warm up with validation only, then reset accumulators before the test pass.
        warm_images, warm_masks = next(iter(make_dataset(pairs["val"][:16], training=False, batch_size=16, augment=False)))
        for _ in range(3):
            p, _ = predict_and_count(warm_images, warm_masks)
            p.numpy()
        assert "GPU:0" in p.device
        for metric in (iou_metric, dice_metric, edge_metric):
            metric.reset_state()
        prediction_dir = OUT / "predicted_masks"
        prediction_dir.mkdir(exist_ok=False)
        rows, batches = [], []
        minimum, maximum, offset = 1.0, 0.0, 0
        status["test_inference_started"] = True
        progress("test_inference")
        evaluation_start = time.perf_counter()
        for images, masks in make_dataset(records, training=False, batch_size=16, augment=False):
            assert float(tf.reduce_min(images)) >= -1 and float(tf.reduce_max(images)) <= 1
            assert np.isin(masks.numpy(), [0, 1]).all()
            batch_start = time.perf_counter()
            probabilities, image_losses = predict_and_count(images, masks)
            prediction_device = probabilities.device
            values, losses = probabilities.numpy()[..., 0], image_losses.numpy()
            forward_seconds = time.perf_counter()-batch_start
            assert "GPU:0" in prediction_device
            assert values.dtype == np.float32 and np.isfinite(values).all() and np.isfinite(losses).all()
            assert values.min() >= 0 and values.max() <= 1
            minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
            truth = masks.numpy()[..., 0].astype(bool)
            prediction = values >= lock["threshold"]
            for j, predicted in enumerate(prediction):
                record = records[offset+j]
                target = truth[j]
                c = counts(target, predicted)
                s = scores(c)
                if c["tp"]+c["fp"] == 0 and target.any():
                    s["road_precision"] = 0.
                c["tn"] = int((~target & ~predicted).sum())
                assert c["tp"]+c["fn"] == file_rows[offset+j]["truth_road_pixels"]
                assert sum(c[k] for k in ("tp", "fp", "fn", "tn")) == 512*512
                path = prediction_dir / f"Prediction_{record['id']}.png"
                Image.fromarray(predicted.astype(np.uint8)).save(path)
                rows.append(dict(id=record["id"], split="test", **c, **s,
                                 pixel_accuracy=(c["tp"]+c["tn"])/(512*512),
                                 truth_road_fraction=float(target.mean()), predicted_road_fraction=float(predicted.mean()),
                                 bce_plus_soft_dice_loss=float(losses[j]), prediction_sha256=sha(path)))
            offset += len(values)
            batches.append(dict(batch=len(batches)+1, size=len(values), completed=offset,
                                gpu_forward_metrics_and_probability_copy_seconds=forward_seconds))
            progress("test_inference", offset)
            if len(batches) % 5 == 0 or offset == 689:
                gpu_samples.append(gpu_sample())
                print(f"Fixed model inference: {offset}/689", flush=True)
        assert offset == 689 and len({r["id"] for r in rows}) == 689
        assert len(batches) == 44 and batches[-1]["size"] == 1
        inference_wall = time.perf_counter()-evaluation_start
        table(OUT / "per_image_metrics.csv", rows)
        table(OUT / "inference_batches.csv", batches)
        report = summarize(rows)
        report["mean_per_image"]["road_precision"] = float(np.mean([r["road_precision"] for r in rows]))
        report["counts"]["tn"] = sum(r["tn"] for r in rows)
        total = report["counts"]
        report["global"]["pixel_accuracy"] = (total["tp"]+total["tn"])/(689*512*512)
        report["global"]["bce_plus_soft_dice_loss"] = float(np.mean([r["bce_plus_soft_dice_loss"] for r in rows]))
        tf_counts = {k: int(getattr(iou_metric, k).numpy()) for k in ("tp", "fp", "fn")}
        assert tf_counts == {k: total[k] for k in tf_counts}
        assert int(edge_metric.intersection.numpy()) == total["boundary_intersection_5px"]
        assert int(edge_metric.union.numpy()) == total["boundary_union_5px"]
        assert abs(float(iou_metric.result())-report["global"]["road_iou"]) < 1e-12
        assert abs(float(dice_metric.result())-report["global"]["road_dice"]) < 1e-12
        assert abs(float(edge_metric.result())-report["global"]["boundary_iou_5px"]) < 1e-12
        progress("independent_export_check", 689)
        # A CPU audit of exported PNGs reuses saved predictions; it is not another
        # model forward pass and makes no change to inference settings.
        for index, row in enumerate(rows):
            with Image.open(records[index]["mask"]) as im:
                target = np.asarray(im)
            path = prediction_dir / f"Prediction_{row['id']}.png"
            with Image.open(path) as im:
                prediction = np.asarray(im)
            assert prediction.shape == target.shape == (512, 512)
            assert np.isin(prediction, [0, 1]).all() and sha(path) == row["prediction_sha256"]
            check = counts(target, prediction)
            check["tn"] = int((~target.astype(bool) & ~prediction.astype(bool)).sum())
            assert all(row[k] == v for k, v in check.items())
            assert sha(records[index]["image"]) == file_rows[index]["image_sha256"]
            assert sha(records[index]["mask"]) == file_rows[index]["mask_sha256"]
        assert {p.name for p in prediction_dir.iterdir()} == {f"Prediction_{r['id']}.png" for r in rows}
        atomic(OUT / "metric_verification.json", dict(status="passed", synthetic_fixtures=fixtures,
               numpy_counts_match_tensorflow_metrics=True, exported_binary_masks_independently_recounted=689,
               input_test_png_hashes_verified_before_and_after=True, final_partial_batch_included=True,
               tensorflow_global_metrics={"road_iou": float(iou_metric.result()), "road_dice": float(dice_metric.result()), "boundary_iou_5px": float(edge_metric.result())},
               tensorflow_pixel_counts=tf_counts))
        progress("uncertainty_summary", 689)
        report["image_bootstrap_95pct"] = bootstrap(rows, lock)
        values = np.array([r["road_iou"] for r in rows])
        report["per_image_road_iou_distribution"] = {"minimum": float(values.min()), "p05": float(np.percentile(values, 5)),
                    "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "maximum": float(values.max())}
        report["lowest_road_iou_ids"] = [r["id"] for r in sorted(rows, key=lambda r: (r["road_iou"], r["id"]))[:12]]
        report["predetermined_illustrative_ids"] = lock["illustrative_test_ids"]
        report["empty_predictions"] = sum(r["tp"]+r["fp"] == 0 for r in rows)
        progress("batch_one_latency_on_predeclared_validation_images", 689)
        latency_records = {r["id"]: r for r in pairs["val"]}
        raw = []
        for i in lock["latency_protocol"]["validation_ids"]:
            with Image.open(latency_records[i]["image"]) as im:
                raw.append(np.asarray(im.convert("RGB"))[None, ...])
        with tf.device("/CPU:0"):
            host_raw = [tf.constant(image) for image in raw]
        with tf.device("/GPU:0"):
            resident = [tf.constant(image.astype(np.float32)/127.5-1.) for image in raw]

        @tf.function(input_signature=[tf.TensorSpec([1, 512, 512, 3], tf.float32)])
        def forward(image):
            with tf.device("/GPU:0"):
                return model(image, training=False)

        @tf.function(input_signature=[tf.TensorSpec([1, 512, 512, 3], tf.uint8)])
        def rgb_to_mask(image):
            with tf.device("/GPU:0"):
                return tf.cast(model(tf.cast(image, tf.float32)/127.5-1., training=False) >= .5, tf.uint8)

        for i in range(15):
            forward(resident[i % 8]).numpy()
            rgb_to_mask(host_raw[i % 8]).numpy()
        timing = {"normalized_forward_and_probability_copy_ms": [], "raw_rgb_cpu_ram_to_binary_mask_cpu_ram_ms": []}
        for i in range(64):
            t = time.perf_counter()
            output = forward(resident[i % 8]); output.numpy()
            assert "GPU:0" in output.device
            timing["normalized_forward_and_probability_copy_ms"].append((time.perf_counter()-t)*1000)
            t = time.perf_counter()
            rgb_to_mask(host_raw[i % 8]).numpy()
            timing["raw_rgb_cpu_ram_to_binary_mask_cpu_ram_ms"].append((time.perf_counter()-t)*1000)
        latency = {key: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(np.mean(v)), "samples_ms": v}
                   for key, v in timing.items()}
        latency.update(protocol=lock["latency_protocol"], gpu=gpu_details, scope=lock["latency_protocol"]["scope"],
                       checkpoint_sha256=lock["checkpoint_sha256"], no_test_reinference_for_latency=True)
        atomic(OUT / "latency.json", latency)
        gpu_samples.append(gpu_sample())
        atomic(OUT / "gpu_samples.json", gpu_samples)
        assert weights_before == fingerprint(model), "In-memory parameters or BatchNorm state changed"
        assert sha(ROOT / "dataset/splits/pair_splits.csv") == lock["manifest_sha256"]
        assert {p: sha(ROOT/p) for p in lock["retained_checkpoint_hashes_before"]} == lock["retained_checkpoint_hashes_before"]
        report.update(status="completed", split="test", model="morning_best", checkpoint=lock["checkpoint"],
                      checkpoint_sha256=lock["checkpoint_sha256"], manifest_sha256=lock["manifest_sha256"], threshold=.5,
                      prediction_execution="saved checkpoint, training=False, tf.function graph, GPU:0",
                      prediction_device=prediction_device, probability_range=[minimum, maximum], loss_definition="Mean per-image BCE plus soft Dice with 1e-6 smoothing",
                      mask_values=[0, 1], single_normalization=True, no_augmentation_or_postprocessing=True,
                      test_forward_passes_per_image=1, test_pairs_excluded=0,
                      inference_batch_size=16, inference_batches=len(batches), last_batch_size=batches[-1]["size"],
                      inference_phase_wall_seconds_including_io_cpu_metrics_and_export=inference_wall,
                      summed_batch_gpu_forward_count_and_copy_seconds=sum(b["gpu_forward_metrics_and_probability_copy_seconds"] for b in batches),
                      all_model_file_hashes_unchanged=True, in_memory_weights_and_bn_state_unchanged=True,
                      in_memory_model_weights_sha256=weights_before,
                      peak_tensorflow_gpu_memory_mib=tf.config.experimental.get_memory_info("GPU:0")["peak"]/1024**2,
                      parameters=model.count_params(), saved_layer_dtype_policies={l.name: l.dtype_policy.name for l in model.layers},
                      runtime={"python": platform.python_version(), "tensorflow": tf.__version__, "keras": tf.keras.__version__, "numpy": np.__version__, "gpu": gpu_details},
                      locked_protocol_file="EVALUATION_LOCK.json", independent_verification="metric_verification.json",
                      dataset_limitations=lock["known_limitations"], finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter()-started)
        atomic(OUT / "test_metrics.json", report)
        status.update(status="completed", phase="metrics_saved", completed=689, finished_at_utc=report["finished_at_utc"],
                      elapsed_seconds=report["elapsed_seconds"], gpu_worker_finished=True, active_worker_pid=None)
        atomic(OUT / "session_status.json", status)
        print(json.dumps({"global": report["global"], "images": 689, "independent_verification": "passed",
                          "peak_gpu_mib": report["peak_tensorflow_gpu_memory_mib"], "seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=datetime.now(timezone.utc).isoformat())
        atomic(OUT / "session_status.json", status)
        raise


if __name__ == "__main__":
    main()
