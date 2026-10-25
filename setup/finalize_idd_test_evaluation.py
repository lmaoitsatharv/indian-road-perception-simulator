"""Record completed test evaluation and verify saved artifacts, without inference."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/test_evaluation_20261003_morning_best"
REL = OUT.relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    result = read(OUT / "test_metrics.json")
    lock_path = OUT / "EVALUATION_LOCK.json"
    lock = read(lock_path)
    lock_hash = sha(lock_path)
    status = read(OUT / "session_status.json")
    verification = read(OUT / "metric_verification.json")
    assert result["status"] == status["status"] == "completed"
    assert verification["status"] == "passed"
    assert result["images"] == 689 and not status["training_started"]
    assert result["test_forward_passes_per_image"] == 1 and result["test_pairs_excluded"] == 0
    assert sha(ROOT / "models/segmentation_final_lock.json") == lock_hash

    hashes = {path: sha(ROOT / path) for path in lock["retained_checkpoint_hashes_before"]}
    assert hashes == lock["retained_checkpoint_hashes_before"]
    assert sha(ROOT / "dataset/splits/pair_splits.csv") == lock["manifest_sha256"]
    assert all(sha(ROOT / path) == expected for path, expected in lock["source_sha256"].items())
    assert all(sha(OUT / "source" / path) == expected for path, expected in lock["source_sha256"].items())

    with (ROOT / "dataset/splits/pair_splits.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    assert dict(Counter(row["split"] for row in pairs)) == lock["split_counts"]
    ids = {int(row["id"]) for row in pairs if row["split"] == "test"}
    assert ids == set(lock["test_ids"]) and len(ids) == 689
    with (OUT / "per_image_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 689 and {int(row["id"]) for row in rows} == ids
    assert all(row["split"] == "test" for row in rows)
    recounted = {key: sum(int(row[key]) for row in rows) for key in result["counts"]}
    assert recounted == result["counts"]
    tp, fp, fn, tn = (recounted[k] for k in ("tp", "fp", "fn", "tn"))
    assert tp+fp+fn+tn == 689*512*512
    assert tp/(tp+fp+fn) == result["global"]["road_iou"]
    assert 2*tp/(2*tp+fp+fn) == result["global"]["road_dice"]
    assert sum(float(r["road_iou"]) < .5 for r in rows) == result["road_iou_below_50pct"]
    assert sum(float(r["road_iou"]) < .75 for r in rows) == result["road_iou_below_75pct"]

    predictions = list((OUT / "predicted_masks").glob("Prediction_*.png"))
    comparisons = list((OUT / "gallery/comparisons").glob("Image_*.jpg"))
    assert len(predictions) == len(comparisons) == 689
    assert {int(p.stem.split("_")[1]) for p in predictions} == ids
    assert {int(p.stem.split("_")[1]) for p in comparisons} == ids
    for row in rows:
        assert sha(OUT / f"predicted_masks/Prediction_{row['id']}.png") == row["prediction_sha256"]
    for path in comparisons:
        with Image.open(path) as image:
            assert image.size == (2048, 612) and image.mode == "RGB"
            image.verify()
    gallery_manifest = read(OUT / "gallery/manifest.json")
    assert gallery_manifest["images"] == 689 and set(gallery_manifest["ids"]) == ids
    with (OUT / "inference_batches.csv").open(newline="", encoding="utf-8") as stream:
        batches = list(csv.DictReader(stream))
    assert len(batches) == 44 and result["last_batch_size"] == 1
    assert read(OUT / "power_request_released.json")["released"]
    assert read(OUT / "artifact_status.json")["status"] == "completed"
    for path in ("TEST_REPORT.md", "gallery/index.html", "plots/test_summary.png",
                 "plots/test_worst_cases.png", "plots/test_predetermined_examples.png"):
        assert (OUT / path).is_file() and (OUT / path).stat().st_size > 0
    binaries = sorted(p.relative_to(ROOT).as_posix() for directory in ("runs", "models", "setup")
                      for p in (ROOT / directory).rglob("*.keras"))
    assert binaries == sorted(hashes)

    now = datetime.now(timezone.utc).isoformat()
    test_record = {"status": "completed", "model": "morning_best", "images": 689,
                   "checkpoint_sha256": result["checkpoint_sha256"], "threshold": .5,
                   "finished_at_utc": result["finished_at_utc"],
                   "metrics_file": REL+"/test_metrics.json", "report": REL+"/TEST_REPORT.md",
                   "gallery": REL+"/gallery/index.html", "protocol_lock": "models/segmentation_final_lock.json",
                   "global_metrics": result["global"], "training_performed": False,
                   "test_used_for_reporting": True,
                   "future_test_informed_selection_requires_new_independent_data": True}
    catalog_path = ROOT / "models/retained_models.json"
    catalog = read(catalog_path)
    catalog.update(status="morning_best_final_test_evaluated", updated_at_utc=now,
                   test_evaluated=True, test_evaluated_models=["morning_best"], final_test_evaluation=test_record)
    for model in catalog["models"]:
        model["test_evaluated"] = model["name"] == "morning_best"
        if model["test_evaluated"]:
            model["test_metrics_file"] = test_record["metrics_file"]
            model["global_test_metrics"] = result["global"]
    assert catalog["recommended_current_model"] == "morning_best"
    save(catalog_path, catalog)

    pointer_path = ROOT / "models/overnight_20261003_selected_model.json"
    pointer = read(pointer_path)
    pointer.update(status="validation_selected_final_test_evaluated", test_evaluated=True,
                   test_evaluation=test_record, updated_at_utc=now)
    save(pointer_path, pointer)
    boundary_path = ROOT / "models/boundary_20261003_selected_model.json"
    boundary = read(boundary_path)
    assert not boundary["test_evaluated"]
    boundary.update(status="validation_boundary_selected_not_test_evaluated",
                    test_evaluation_note="The project test set was evaluated only with locked morning best. This retained boundary checkpoint was not tested.")
    save(boundary_path, boundary)
    config_path = ROOT / "models/model_config.json"
    config = read(config_path)
    config.update(final_test_evaluation=test_record, inference_protocol_lock="models/segmentation_final_lock.json")
    save(config_path, config)

    notebook_path = ROOT / "IDD_UNet_Model.ipynb"
    notebook = read(notebook_path)
    code_before = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    notebook["cells"][0]["source"] = (
        "## Current project status — final test evaluation complete, 3 October 2026\n\n"
        "The original, morning-best trial 13 and boundary-best epoch 4 models are trained and preserved. "
        "**Morning best is locked and has been evaluated on all 689 test pairs: road IoU 91.65%, Dice 95.65%.** "
        "See the [final test report](runs/test_evaluation_20261003_morning_best/TEST_REPORT.md), "
        "[test gallery](runs/test_evaluation_20261003_morning_best/gallery/index.html), "
        "[model catalog](models/README.md) and [project index](README.md).\n\n"
        "The setup explanations, code cells and stored outputs below are historical construction/training records. "
        "They were not rerun during final evaluation. The disposable untrained setup binary was removed; "
        "setup cells can rebuild a fresh decoder from retained ImageNet weights. `RUN_TRAINING` remains false. "
        "Read the final result section at the end for the current status.\n\n"
        "The completed preprocessing notebook is unchanged. Split **5507/687/689**, normalization once to "
        "**[-1,1]** and masks **0/1** remain unchanged. The test set has now been used for reporting; "
        "future test-informed model changes need independent evaluation data.\n"
    ).splitlines(keepends=True)
    heading = "## Completed locked-model test evaluation — 3 October 2026"
    if not any(heading in "".join(c.get("source", [])) for c in notebook["cells"]):
        markdown = heading+"\n\n"+(
            "All 689 test pairs were evaluated once with the fixed morning-best trial-13 epoch-1 model "
            "on the RTX 4060 in WSL2. Threshold 0.5, no training, augmentation or mask cleanup.\n\n"
            "| Test metric | Result |\n|---|---:|\n"
            "| Road IoU | **91.65%** |\n| Dice / road F1 | **95.65%** |\n"
            "| Road precision | **95.19%** |\n| Road recall | **96.11%** |\n"
            "| Strict Boundary IoU, 5px | **55.23%** |\n| Visible contour F1, 3px | **70.70%** |\n\n"
            "Road IoU is 0.47 percentage points below validation. There are 14 images below 50% IoU and "
            "42 below 75%; no test case was removed. Agreement with supplied masks is measured; "
            "sequence identities and exact class conversion remain unavailable.\n\n"
            "Warm RGB-RAM-to-mask time: median 13.18 ms, p95 16.99 ms, excluding capture, file reading, "
            "tracking, planning and display. Saved binary predictions were independently recounted, "
            "and all model/data hashes remain unchanged. Original and boundary checkpoints were not tested.\n\n"
            "[Full final report](runs/test_evaluation_20261003_morning_best/TEST_REPORT.md) · "
            "[All 689 test comparisons](runs/test_evaluation_20261003_morning_best/gallery/index.html) · "
            "[Per-image scores](runs/test_evaluation_20261003_morning_best/per_image_metrics.csv)\n\n"
            "![Test evaluation summary](runs/test_evaluation_20261003_morning_best/plots/test_summary.png)\n\n"
            "### Evaluation summary\n\n"
            "We selected the checkpoint using validation data, then fixed its preprocessing and threshold "
            "before opening the test set. On all 689 test images it achieved 91.65% road IoU and 95.65% "
            "Dice. We checked boundaries, timing and weak cases, and verified saved predictions independently. "
            "This supports our segmentation prototype; driving safety still requires closed-loop simulation.\n\n"
            "Next work: fixed reusable inference and camera-to-metric-map geometry, followed by Hybrid A*, "
            "RL STOP / SLOW / PROCEED and MATLAB/Simulink visualization. These integration stages remain unfinished.\n"
        )
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)})
    assert [c for c in notebook["cells"] if c["cell_type"] == "code"] == code_before
    save(notebook_path, notebook)

    source = OUT / "source/setup"
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    # Keep the original executed gallery builder snapshot; record the small final wording revision separately.
    shutil.copy2(ROOT / "setup/build_idd_test_report.py", source / "build_idd_test_report_final_wording.py")
    assert sha(lock_path) == lock_hash
    final = {"status": "passed", "verified_at_utc": now, "no_new_model_inference": True,
             "csv_counts_match_final_metrics": True, "all_689_prediction_hashes_match": True,
             "gallery_images": 689, "gallery_sizes": [2048, 612], "report_exists": True,
             "test_ids_match_locked_manifest": True, "final_partial_batch_included": True,
             "retained_model_files": binaries, "retained_model_hashes_unchanged": True,
             "manifest_hash_unchanged": True, "evaluation_sources_and_lock_unchanged": True,
             "notebook_code_and_saved_outputs_unchanged": True, "temporary_power_request_released": True,
             "worker_status": status["status"], "active_worker_pid": status["active_worker_pid"],
             "selected_model": "morning_best", "test_evaluated_models": ["morning_best"],
             "report_sha256": sha(OUT / "TEST_REPORT.md"), "gallery_index_sha256": sha(OUT / "gallery/index.html"),
             "report_wording_revision": "Added visual weak-case observations and active-versus-reserved GPU memory clarification; numeric results unchanged."}
    save(OUT / "final_artifact_verification.json", final)
    print(json.dumps({"status": final["status"], "images": 689, "model": "morning_best",
                      "test_road_iou": result["global"]["road_iou"], "models_retained": len(binaries)}, indent=2))


if __name__ == "__main__":
    main()
