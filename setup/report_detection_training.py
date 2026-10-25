"""Build a detector result report from preserved local training evidence."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(name):
    if Path(name).name != name or not name.startswith("detection_train_"):
        raise ValueError("Expected one local detector run name")
    run = ROOT / "runs" / name
    owner = run.with_name(name + "_launcher")
    verification = run / "report_verification_02/checkpoint_verification.json"
    checkpoint = read(verification)
    protection = read(run / "completion_protection.json")
    recipe = read(run / "recipe.json")
    prepared = read(ROOT / recipe["dataset"] / "preparation.json")
    progress = read(run / "progress.json")
    state = read(run / "status.json")
    exit_record = read(owner / "exit.json")
    assert checkpoint["passed"] and protection["passed"]
    assert checkpoint["checkpoint_files_unchanged"] and protection["protected_files_unchanged"]
    assert protection["protected_sha256"] == read(owner / "protected_before.json")
    for label, verified in checkpoint["checkpoints"].items():
        assert sha(ROOT / verified["path"]) == verified["sha256"]
    with (run / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    assert [row["epoch"] for row in rows] == list(range(1, recipe["epochs"] + 1))
    assert progress["epoch_completed"] == state["epochs_completed"] == len(rows)
    best = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    best_epoch = int(best["epoch"])
    for key, value in checkpoint["checkpoints"]["best"]["train_metrics"].items():
        if key in best:
            assert abs(value - best[key]) < 1e-10
    metrics = progress["metrics"]
    log = owner / "training.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 80000))
        tail = stream.read().decode("utf-8", errors="replace").replace("\r", "\n")
    lines = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", tail).splitlines()
    number = r"([0-9.eE+-]+)"
    pattern = re.compile(r"^\s*(all|" + "|".join(prepared["names"]) + r")\s+(\d+)\s+(\d+)\s+"
                         + r"\s+".join([number] * 4) + r"\s*$")
    classes = {}
    excerpt = []
    for line in lines:
        match = pattern.fullmatch(line)
        if match:
            label, images, objects, precision, recall, ap50, ap = match.groups()
            classes[label] = {"images_with_class": int(images), "instances": int(objects),
                "precision": float(precision), "recall": float(recall),
                "ap50": float(ap50), "ap50_95": float(ap)}
            excerpt.append(line)
        elif line.startswith(("Speed:", "Results saved to")) or "2210/2210" in line:
            excerpt.append(line)
    assert set(classes) == set(prepared["names"]) | {"all"}
    assert classes["all"]["images_with_class"] == prepared["counts"]["val"]
    assert sum(classes[label]["instances"] for label in prepared["names"]) == classes["all"]["instances"]
    assert all(classes[label]["instances"] == prepared["objects"]["val"][label] for label in prepared["names"])
    for short, key in (("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)"),
                       ("ap50", "metrics/mAP50(B)"), ("ap50_95", "metrics/mAP50-95(B)")):
        assert abs(classes["all"][short] - metrics[key]) < 0.0006
    assert any("2210/2210" in line for line in excerpt)
    assert any(line.startswith("Results saved to") for line in excerpt)
    excerpt_path = run / "final_validation_excerpt.txt"
    excerpt_path.write_text("\n".join(excerpt) + "\n", encoding="utf-8")
    evidence = [run / path for path in ("results.csv", "progress.json", "recipe.json", "args.yaml",
        "readiness.json", "status.json", "startup_verification.json", "completion_protection.json")]
    evidence += [verification, owner / "exit.json", owner / "keep_awake.json", log, excerpt_path]
    report = {
        "schema": "idd_detector_training_report_v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "run": run.relative_to(ROOT).as_posix(), "report_evidence_verified": True,
        "status": "epochs_and_final_validation_complete_worker_killed_after_results",
        "training_process_clean_exit": exit_record["returncode"] == 0,
        "launcher_exit": exit_record, "shutdown_cause": "not established",
        "trainer_status_json_is_stale": state["status"] == "training",
        "epochs_completed": len(rows), "maximum_epochs": recipe["epochs"],
        "best_epoch": best_epoch, "checkpoint_selection": "highest validation mAP50-95 in pinned Ultralytics 8.3.225",
        "optimizer_steps_recorded": state["optimizer_steps_completed"],
        "framework_training_and_epoch_validation_seconds": rows[-1]["time"],
        "dataset": recipe["dataset"], "counts": prepared["counts"],
        "train_instances": sum(prepared["objects"]["train"].values()),
        "validation_instances": classes["all"]["instances"], "names": prepared["names"],
        "evaluation_split": "validation", "official_test_used": False,
        "protocol": "project filtered validation set and custom 12-class mapping; not official IDD benchmark",
        "final_validation_complete": True, "final_validation_batches": 2210,
        "final_best_checkpoint_metrics": metrics,
        "per_class_from_rounded_final_log": {label: classes[label] for label in prepared["names"]},
        "per_class_precision_note": "Printed source values have approximately three significant digits",
        "precision_recall_aggregation": "macro mean across classes at the evaluator selected F1 operating point; not a deployed threshold",
        "epoch20_metrics_before_stripped_checkpoint_validation": {key: rows[-1][key] for key in metrics},
        "improvement_from_epoch1_percentage_points": {
            "mAP50": 100 * (rows[-1]["metrics/mAP50(B)"] - rows[0]["metrics/mAP50(B)"]),
            "mAP50_95": 100 * (rows[-1]["metrics/mAP50-95(B)"] - rows[0]["metrics/mAP50-95(B)"]),
        },
        "recipe": recipe, "checkpoints": checkpoint["checkpoints"],
        "checkpoint_verification": verification.relative_to(ROOT).as_posix(),
        "protected_files_unchanged": True, "protected_fingerprints": protection["protected_fingerprints"],
        "initial_checkpoint_unchanged": True, "recipe_unchanged": True,
        "reported_validation_speed_ms_per_image": {"preprocess": 0.1, "inference": 1.7, "postprocess": 1.1},
        "speed_scope": "batched GPU validation components; not live camera latency or end-to-end FPS",
        "additional_optimizer_steps_for_report": 0, "distance_accuracy_measured": False,
        "tracking_evaluated": False, "nearby_only_accuracy_measured": False,
        "real_deployment_approved": False, "model_control": False,
        "evidence_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in evidence},
    }
    save(run / "training_report.json", report)
    save(ROOT / "docs/results/detection_training.json", report)
    print(json.dumps({key: report[key] for key in ("report_evidence_verified", "status", "epochs_completed",
        "best_epoch", "validation_instances", "final_best_checkpoint_metrics", "protected_files_unchanged")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    main(parser.parse_args().run_name)
