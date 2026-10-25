"""Reload the best checkpoint and review validation only; never open test images."""

import argparse
import csv
import hashlib
import heapq
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from idd_data import make_dataset, read_pairs
import idd_model  # Registers the saved model's custom loss and metrics.

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_recorded_checkpoint(recorded, root):
    """Recover project-local lineage across recorded Windows/WSL paths."""
    if recorded is None:
        return None
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("initial_checkpoint must be a path string or null.")
    root = Path(root).resolve()
    normalized = recorded.replace("\\", "/")
    path = Path(normalized).expanduser()
    windows_drive = len(normalized) > 2 and normalized[1:3] == ":/"
    candidate = path if path.is_absolute() or windows_drive else root / path
    if candidate.is_file():
        return candidate.resolve()
    # Original run records contain /mnt/e/... or E:/... paths. Recover only
    # recognized project folders, then verify the recorded digest before loading.
    for marker in ("/runs/", "/models/"):
        if marker in normalized:
            portable = (root / marker.strip("/") / normalized.rsplit(marker, 1)[1]).resolve()
            if not portable.is_relative_to(root):
                raise ValueError("Recorded checkpoint escapes the project.")
            return portable
    return candidate.resolve()


def verify_initial_comparison(config, model, root):
    """Unavailable initial artifacts yield unknown equivalence, never success."""
    initial_checkpoint = resolve_recorded_checkpoint(config.get("initial_checkpoint"), root)
    result = {"status": "unavailable", "encoder_weights_unchanged": None,
              "decoder_weights_changed": None, "initial_checkpoint_sha256": None,
              "resolved_initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None}
    if initial_checkpoint is None:
        result["reason"] = "Fresh initialization did not save an initial checkpoint for equivalence comparison."
        return result
    if not initial_checkpoint.is_file():
        result["reason"] = "The recorded initial checkpoint is no longer available; use its historical verification evidence."
        return result
    actual_hash = sha256(initial_checkpoint)
    if actual_hash != config.get("initial_checkpoint_sha256"):
        raise ValueError("The initial checkpoint does not match the recorded hash.")
    with tf.device("/CPU:0"):
        initial = tf.keras.models.load_model(initial_checkpoint, compile=False, safe_mode=True)
    before = initial.get_layer("mobilenetv2_encoder").weights
    after = model.get_layer("mobilenetv2_encoder").weights
    encoder_unchanged = len(before) == len(after) and all(
        np.array_equal(left.numpy(), right.numpy()) for left, right in zip(before, after))
    decoder_changed = not np.array_equal(initial.get_layer("road_probability").kernel.numpy(),
                                          model.get_layer("road_probability").kernel.numpy())
    if config["phase"] == "frozen" and (not encoder_unchanged or not decoder_changed):
        raise RuntimeError("Frozen encoder / trained decoder verification failed.")
    result.update(status="verified", encoder_weights_unchanged=encoder_unchanged,
                  decoder_weights_changed=decoder_changed, initial_checkpoint_sha256=actual_hash,
                  reason="Compared to the existing initial artifact with its recorded SHA-256.")
    return result


def review(run_dir, *, output_dir=None):
    root = ROOT
    run_dir = Path(run_dir).resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "validation_review"
    if output.exists():
        raise FileExistsError("Choose a new validation output folder; saved reviews are preserved.")
    config = json.loads((run_dir / "run_config.json").read_text())
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Review requires a completed training run")
    manifest = root / "dataset" / "splits" / "pair_splits.csv"
    if sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("The dataset split changed")
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        raise RuntimeError("Run checkpoint review in the verified WSL GPU environment")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    records = read_pairs(root / "dataset" / "splits")["val"]
    if len(records) != 687:
        raise ValueError("Unexpected validation count")
    dataset = make_dataset(records, training=False, batch_size=config["batch_size"])
    checkpoint = run_dir / "best.keras"
    model = tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True)
    initial_verification = verify_initial_comparison(config, model, root)
    # Use the same compiled validation path as fit; graph optimizations can give
    # slightly different probabilities than an eager forward call near 0.5.
    idd_model.compile_model(model, learning_rate=config["learning_rate"])
    keras_evaluation = {key: float(value) for key, value in
                        model.evaluate(dataset, verbose=0, return_dict=True).items()}

    @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
    def predict_graph(images):
        return model(images, training=False)

    output.mkdir(parents=True, exist_ok=False)
    mask_dir = output / "predicted_masks"
    mask_dir.mkdir()
    selected_ids = {records[index]["id"] for index in np.linspace(0, len(records) - 1, 6, dtype=int)}
    previews, worst, per_image = {}, [], []
    tp_total = fp_total = fn_total = offset = 0
    probability_min, probability_max = 1.0, 0.0
    prediction_device = None
    started = time.perf_counter()
    for images, masks in dataset:
        probabilities = predict_graph(images)
        prediction_device = probabilities.device
        values = probabilities.numpy()
        if not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 1:
            raise RuntimeError("Invalid output probabilities")
        probability_min = min(probability_min, float(values.min()))
        probability_max = max(probability_max, float(values.max()))
        truth, prediction = masks.numpy()[..., 0] >= 0.5, values[..., 0] >= 0.5
        for index in range(len(values)):
            record = records[offset + index]
            target, predicted = truth[index], prediction[index]
            tp = int(np.count_nonzero(target & predicted))
            fp = int(np.count_nonzero(~target & predicted))
            fn = int(np.count_nonzero(target & ~predicted))
            union, denominator = tp + fp + fn, 2 * tp + fp + fn
            iou = tp / union if union else 1.0
            dice = 2 * tp / denominator if denominator else 1.0
            row = {"id": record["id"], "road_iou": iou, "road_dice": dice,
                   "truth_road_fraction": float(target.mean()),
                   "predicted_road_fraction": float(predicted.mean())}
            per_image.append(row)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            if record["id"] in selected_ids:
                previews[record["id"]] = predicted.astype(np.uint8)
            entry = (iou, record["id"], predicted.astype(np.uint8))
            if len(worst) < 6:
                heapq.heappush(worst, (-iou, -record["id"], entry))
            elif (iou, record["id"]) < (worst[0][2][0], worst[0][2][1]):
                heapq.heapreplace(worst, (-iou, -record["id"], entry))
        offset += len(values)
    if offset != 687:
        raise RuntimeError("Validation iteration did not visit every image")
    worst_ids = []
    for _, _, entry in sorted(worst, key=lambda item: (item[2][0], item[2][1])):
        _, pair_id, prediction = entry
        previews[pair_id] = prediction
        worst_ids.append(pair_id)
    for pair_id, prediction in previews.items():
        # Store binary prediction values as 0/1; display scaling belongs to plotting.
        Image.fromarray(prediction).save(mask_dir / f"Prediction_{pair_id}.png")
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    global_iou = tp_total / (tp_total + fp_total + fn_total)
    global_dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    difference = abs(keras_evaluation["road_iou"] - status["best_validation_road_iou"])
    prediction_difference = abs(global_iou - keras_evaluation["road_iou"])
    report = {
        "split": "validation", "images_evaluated": offset, "threshold": 0.5,
        "global_road_iou": global_iou, "global_road_dice": global_dice,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "best_epoch": status["best_epoch"],
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(manifest),
        "saved_checkpoint_matches_logged_iou": difference < 1e-5,
        "keras_evaluation": keras_evaluation,
        "prediction_execution": "tf.function graph, matching training validation execution",
        "graph_predictions_match_keras_evaluation": prediction_difference < 1e-5,
        "graph_prediction_iou_difference": prediction_difference,
        "encoder_weights_unchanged": initial_verification["encoder_weights_unchanged"],
        "decoder_weights_changed": initial_verification["decoder_weights_changed"],
        "initial_checkpoint_sha256": initial_verification["initial_checkpoint_sha256"],
        "initial_checkpoint_verification": initial_verification,
        "logged_iou_difference": difference,
        "prediction_device": prediction_device,
        "probability_range": [probability_min, probability_max],
        "illustrative_ids": sorted(selected_ids), "lowest_iou_ids": worst_ids,
        "test_set_evaluated": False,
        "elapsed_seconds_including_io": time.perf_counter() - started,
        "notes": ["Global road overlap at threshold 0.5; threshold has not been tuned.",
                  "Illustrative images are evenly spaced by sorted validation ID, not a random quality estimate.",
                  "Lowest-IoU cases were selected after evaluating validation only.",
                  "Original images and masks were not rewritten."],
    }
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if difference >= 1e-5 or prediction_difference >= 1e-5:
        raise RuntimeError("Reloaded checkpoint does not reproduce logged validation IoU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New validation review folder; existing evidence is refused.")
    args = parser.parse_args()
    review(args.run_dir, output_dir=args.output_dir)
