"""Verify deployment inference on six predetermined validation examples only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from idd_inference import LockedRoadSegmenter, normalize_raw_rgb, require_raw_rgb, require_normalized_rgb, read_rgb, write_prediction, file_sha256

IDS = [9, 48, 103, 606, 4160, 6514]


def fingerprint(model):
    h = hashlib.sha256()
    for value in model.get_weights():
        h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError("Use the saved verification evidence; this directory already exists.")
    with (ROOT / "dataset/splits/pair_splits.csv").open(newline="", encoding="utf-8-sig") as f:
        pairs = {int(row["id"]): row for row in csv.DictReader(f)}
    assert all(pairs[i]["split"] == "val" for i in IDS)
    lock = json.loads((ROOT / "models/segmentation_final_lock.json").read_text(encoding="utf-8"))
    before_hashes = {path: file_sha256(ROOT / path) for path in lock["retained_checkpoint_hashes_before"]}
    assert before_hashes == lock["retained_checkpoint_hashes_before"]
    manifest_before = file_sha256(ROOT / "dataset/splits/pair_splits.csv")
    test_metrics_before = file_sha256(ROOT / "runs/test_evaluation_20261003_morning_best/test_metrics.json")
    OUT.mkdir(parents=True)
    checks = {}
    for name, callback, data in (
        ("normalized_floats_rejected_by_raw_entry", require_raw_rgb, np.zeros((512,512,3), np.float32)),
        ("wrong_shape_rejected", require_raw_rgb, np.zeros((256,512,3), np.uint8)),
        ("raw_float255_rejected_by_normalized_entry", require_normalized_rgb, np.full((512,512,3),255,np.float32)),
        ("nan_rejected", require_normalized_rgb, np.full((512,512,3),np.nan,np.float32)),
    ):
        try:
            callback(data)
        except ValueError:
            checks[name] = True
        else:
            raise AssertionError(name)
    fixture = np.broadcast_to(np.array([0,127,255], np.uint8), (512,512,3)).copy()
    normalized = normalize_raw_rgb(fixture)
    assert normalized.dtype == np.float32 and normalized[0,0,0] == -1 and normalized[0,0,2] == 1
    assert require_normalized_rgb(normalized) is normalized
    checks["normalization_black_minus1_white_plus1"] = True
    checks["already_normalized_values_not_rescaled"] = True
    segmenter = LockedRoadSegmenter(device="gpu", gpu_memory_mib=3072)
    tf = segmenter.tf
    original_weights = fingerprint(segmenter.model)
    max_norm_diff = float(np.max(np.abs(normalized-tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(fixture,tf.float32)).numpy())))
    assert max_norm_diff == 0
    checks["normalization_exactly_matches_existing_tensorflow_preprocessing"] = True
    segmenter.warmup()
    # Reproduce the saved batch-16 protocol separately from live batch-one use.
    # Frozen BatchNorm makes outputs independent of other images in the batch.
    fixtures = [read_rgb(ROOT / f"dataset/splits/val/images/Image_{i}.png") for i in IDS]
    batch = [normalize_raw_rgb(rgb) for rgb, _ in fixtures]
    batch += [batch[0]] * (16-len(batch))

    @tf.function(input_signature=[tf.TensorSpec([None,512,512,3],tf.float32)])
    def reference_batch(images):
        return segmenter.model(images, training=False)

    batch_values = reference_batch(np.stack(batch)).numpy()[...,0]
    rows = []
    for index, (i, (rgb, metadata)) in enumerate(zip(IDS, fixtures)):
        raw_prediction = segmenter.predict_rgb(rgb)
        normalized_prediction = segmenter.predict_normalized(normalize_raw_rgb(rgb))
        assert np.array_equal(raw_prediction.mask, normalized_prediction.mask)
        assert np.array_equal(raw_prediction.probability, normalized_prediction.probability)
        expected_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(rgb,tf.float32))
        assert np.array_equal(normalize_raw_rgb(rgb), expected_tensor.numpy())
        graph_reference = segmenter._infer(expected_tensor[None]).numpy()[0,...,0]
        assert np.array_equal(graph_reference, raw_prediction.probability)
        assert "GPU:0" in raw_prediction.output_device
        previous_path = ROOT / f"runs/model_comparisons/baseline_vs_trial13_validation/predicted_masks/trial13/Prediction_{i}.png"
        previous = np.asarray(Image.open(previous_path))
        mismatch = raw_prediction.mask != previous
        count = int(mismatch.sum())
        # Exact reproduction at the original batch shape checks protocol fidelity.
        # Batch-one rounding is measured, not hidden by changing the threshold.
        assert np.array_equal((batch_values[index] >= .5).astype(np.uint8), previous)
        delta = np.abs(raw_prediction.probability-batch_values[index])
        distance_at_flip = float(np.max(np.abs(raw_prediction.probability[mismatch]-.5))) if count else None
        assert np.array_equal(raw_prediction.mask != (batch_values[index]>=.5), mismatch)
        folder = write_prediction(OUT, f"Image_{i}", rgb, raw_prediction, metadata, model_lock=segmenter.lock)
        restored = np.asarray(Image.open(folder / "road_mask_01.png"))
        assert restored.dtype == np.uint8 and set(np.unique(restored)).issubset({0,1})
        assert np.array_equal(restored, raw_prediction.mask)
        assert np.array_equal(np.load(folder / "road_probability.npy", allow_pickle=False), raw_prediction.probability)
        truth = np.asarray(Image.open(ROOT / f"dataset/splits/val/masks/Mask_{i}.png")) > 0
        pred = raw_prediction.mask > 0
        tp, fp, fn = int((truth&pred).sum()), int((~truth&pred).sum()), int((truth&~pred).sum())
        rows.append({"id": i, "split": "val", "road_iou": tp/(tp+fp+fn),
                     "raw_vs_normalized_probability_exact": True, "matches_existing_preprocessing_graph_exactly": True,
                     "previous_batch16_mask_disagreement_pixels": count,
                     "saved_batch16_mask_exactly_reproduced": True,
                     "batch1_vs_batch16_max_probability_difference": float(delta.max()),
                     "batch1_vs_batch16_mean_probability_difference": float(delta.mean()),
                     "maximum_distance_from_threshold_at_disagreement": distance_at_flip,
                     "mask_values": [int(v) for v in np.unique(restored)], "inference_ms": raw_prediction.inference_ms})
        print(f"Verified validation Image {i}: {count} pixel differences from saved batch-16 reference", flush=True)
    assert fingerprint(segmenter.model) == original_weights
    assert all(file_sha256(ROOT/path) == h for path,h in before_hashes.items())
    assert file_sha256(ROOT / "dataset/splits/pair_splits.csv") == manifest_before
    assert file_sha256(ROOT / "runs/test_evaluation_20261003_morning_best/test_metrics.json") == test_metrics_before
    report = {"status": "passed", "ids": IDS, "split": "val", "images": len(IDS), "runtime": segmenter.runtime,
              "checks": checks, "normalization_max_absolute_difference": max_norm_diff,
              "per_image": rows, "in_memory_model_and_bn_unchanged": True, "all_three_checkpoint_hashes_unchanged": True,
              "manifest_unchanged": True, "test_results_unchanged": True, "test_image_contents_read": False,
              "new_training_started": False, "new_checkpoint_written": False,
              "batch_comparison_note": "Identical locked weights, preprocessing and threshold. Batch-16 masks exactly reproduce prior exports; batch-one uses different floating point computation order in saved mixed-precision layers. All differences are explicitly recorded.",
              "previous_reference_total_pixel_disagreements": sum(r["previous_batch16_mask_disagreement_pixels"] for r in rows)}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "images": len(IDS), "device":segmenter.device,
                      "saved_reference_disagreements": report["previous_reference_total_pixel_disagreements"]},indent=2))


if __name__ == "__main__":
    main()
