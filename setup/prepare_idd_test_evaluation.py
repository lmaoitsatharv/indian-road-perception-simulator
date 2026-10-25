"""Lock the selected model and protocol before opening any test PNG contents."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/test_evaluation_20261003_morning_best"
EXPECTED_MODEL = "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"
EXPECTED_MANIFEST = "512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing evaluation; inspect its status instead of rerunning or overwriting it.")
    catalog = json.loads((ROOT / "models/retained_models.json").read_text())
    assert catalog["recommended_current_model"] == "morning_best"
    selected = next(m for m in catalog["models"] if m["name"] == "morning_best")
    assert selected["checkpoint_sha256"] == digest(ROOT / selected["checkpoint"]) == EXPECTED_MODEL
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    assert digest(manifest) == EXPECTED_MANIFEST
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len({int(r["id"]) for r in rows}) == len(rows) == 6883
    ids = {split: sorted(int(r["id"]) for r in rows if r["split"] == split) for split in ("train", "val", "test")}
    counts = {split: len(values) for split, values in ids.items()}
    assert counts == {"train": 5507, "val": 687, "test": 689}
    stored_hashes = {split: {r["image_sha256"] for r in rows if r["split"] == split} for split in ids}
    overlaps = {f"{a}_vs_{b}": len(stored_hashes[a] & stored_hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(value == 0 for value in overlaps.values())
    for split in ids:
        for folder, prefix in (("images", "Image"), ("masks", "Mask")):
            actual = {p.name for p in (ROOT / f"dataset/splits/{split}/{folder}").iterdir()}
            assert actual == {f"{prefix}_{i}.png" for i in ids[split]}
    retained_hashes = {m["checkpoint"]: digest(ROOT / m["checkpoint"]) for m in catalog["models"]}
    assert retained_hashes == {m["checkpoint"]: m["checkpoint_sha256"] for m in catalog["models"]}
    sources = ("idd_data.py", "idd_model.py", "idd_boundary.py", "setup/boundary_metrics.py", "setup/evaluate_idd_test.py")
    source_hashes = {name: digest(ROOT/name) for name in sources}
    training_config = json.loads((ROOT / selected["run_config"]).read_text())
    assert source_hashes["idd_data.py"] == training_config["source_sha256"]["idd_data.py"]
    assert source_hashes["idd_model.py"] == training_config["source_sha256"]["idd_model.py"]
    lock = {
        "status": "locked_before_test_content_access", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested proper evaluation of the locked model on all 689 reserved test images.",
        "selection_basis": "Morning-best trial 13 selected using training/validation results before opening the test set.",
        "model": "morning_best", "checkpoint": selected["checkpoint"], "checkpoint_sha256": EXPECTED_MODEL,
        "selected_epoch": 1, "threshold": .5, "threshold_comparator": ">=", "input_shape": [512, 512, 3],
        "input_preprocessing": "Existing idd_data.make_dataset(training=False): raw RGB normalized once to [-1,1].",
        "mask_labels": [0, 1], "inference_training_flag": False, "batch_size": 16, "drop_remainder": False,
        "augmentation": False, "test_time_augmentation": False, "postprocessing": "threshold only; no morphology or component filtering",
        "precision": "Saved checkpoint dtype policies; mixed_float16 feature layers with float32 probability head. No rebuild or precision conversion.",
        "seed": 42, "deterministic_operations": True, "manifest_sha256": EXPECTED_MANIFEST,
        "split_counts": counts, "test_ids": ids["test"], "last_batch_size": 1,
        "primary_metric": "Global class-1 road IoU: sum(TP)/(sum(TP)+sum(FP)+sum(FN)).",
        "other_metrics": ["global road Dice", "road precision", "road recall", "supplementary pixel accuracy",
                          "mean/median/per-image road IoU", "Boundary IoU at 5px and 14px", "visible contour F1 at 3px tolerance", "BCE plus soft Dice loss"],
        "boundary_metric_definition": "Inner foreground bands, square erosion, image frame included. Contour F1 uses many-to-one square 3px matching and excludes the frame.",
        "boundary_reference": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py",
        "bootstrap": {"resamples": 3000, "seed": 42, "unit": "image", "interval": "percentile 2.5/97.5", "limitation": "Descriptive image-resampling intervals; unknown sequence correlation is not accounted for."},
        "latency_protocol": {"batch_size": 1, "samples": 64, "warmup_per_path": 15,
                             "validation_ids": ids["val"][::85][:8], "selection": "Fixed spaced validation IDs, not selected from test outcomes.",
                             "scope": "Warm normalized forward plus probability copy, and CPU-RAM raw RGB through normalization/GPU/threshold to CPU-RAM binary mask; excludes camera/planning/display."},
        "illustrative_test_ids": [ids["test"][int(i*(len(ids["test"])-1)/7)] for i in range(8)],
        "failure_selection": "Lowest 12 test per-image road IoUs, shown alongside predetermined examples; no exclusions or tuning.",
        "stored_manifest_exact_image_hash_overlap": overlaps,
        "stored_manifest_hash_check_note": "Cross-split comparison uses stored original image hashes; all test PNG hashes will be verified live before and after evaluation.",
        "known_limitations": ["Original sequence IDs are unavailable; sequence-independent generalization cannot be claimed.",
                              "Third-party positive-class conversion and ignore-label handling remain unconfirmed; score supplied binary road labels.",
                              "110 all-zero-mask pairs were previously excluded from split copies; these test results apply to the retained nonempty-mask subset.",
                              "Test results are not to be used to select another of the three models, tune the threshold or train further on this test split."],
        "retained_checkpoint_hashes_before": retained_hashes, "source_sha256": source_hashes,
        "training_prohibited": True, "parameter_or_bn_updates_prohibited": True, "test_png_contents_opened_during_preparation": False,
    }
    OUT.mkdir()
    (OUT / "EVALUATION_LOCK.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    (ROOT / "models/segmentation_final_lock.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    (OUT / "session_status.json").write_text(json.dumps({"status": "prepared", "completed": 0, "total": 689,
          "test_inference_started": False, "locked_at_utc": lock["locked_at_utc"]}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"locked_model": selected["checkpoint"], "threshold": .5, "split_counts": counts,
                      "stored_exact_image_overlap": overlaps, "test_contents_read": False, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
