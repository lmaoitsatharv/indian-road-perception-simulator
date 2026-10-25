"""Read-only training inspection plus a dated portable results snapshot."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.prepare_simulation_dataset import protected, save, sha


def report(run_dir):
    run_dir = run_dir.resolve()
    if run_dir.parent != ROOT / "runs" or not run_dir.name.startswith("simulation_"):
        raise ValueError("Expected a local runs/simulation_* directory")
    config = json.loads((run_dir / "run_config.json").read_text())
    state = json.loads((run_dir / "status.json").read_text())
    progress = json.loads((run_dir / "progress.json").read_text())
    rows = []
    if (run_dir / "history.csv").exists():
        with (run_dir / "history.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                parsed = {k: float(v) for k, v in row.items()}
                if all(math.isfinite(v) for v in parsed.values()):
                    rows.append(parsed)
    best = max(rows, key=lambda r: r["val_road_iou"]) if rows else None
    evaluation = None
    if (run_dir / "test_evaluation.json").is_file():
        evaluation = json.loads((run_dir / "test_evaluation.json").read_text())
    output = {"status": state["status"], "snapshot_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": config.get("training_started", False),
              "run_directory": run_dir.relative_to(ROOT).as_posix(), "model_domain": "simulation_only",
              "dataset": "dataset/simulation_india_v5", "split_counts": config["split_counts"],
              "manifest_sha256": config["manifest_sha256"], "gpu": config["gpu"],
              "progress": progress, "completed_epochs": len(rows),
              "best_validation": None if best is None else {"epoch": int(best["epoch"]) + 1,
                  "road_iou": best["val_road_iou"], "road_dice": best["val_road_dice"], "loss": best["val_loss"]},
              "protected_real_unchanged": protected() == config["protected_real_sha256"],
              "trainer_sources_unchanged": all(sha(ROOT / name) == value for name,value in config["source_sha256"].items()),
              "test_model_evaluation_started": evaluation is not None or config["test_model_evaluation_started"],
              "test_images_read": bool(evaluation["test_images_read"]) if evaluation else config["test_images_read"],
              "test_evaluation": evaluation, "fine_tuning_started": False,
              "verification": (run_dir / "verification.json").relative_to(ROOT).as_posix(),
              "report": "docs/SIMULATION_TRAINING.md",
              "live_status": (run_dir / "status.json").relative_to(ROOT).as_posix(),
              "live_progress": (run_dir / "progress.json").relative_to(ROOT).as_posix()}
    if best is not None and (run_dir / "best.keras").is_file():
        output["best_checkpoint_sha256_at_snapshot"] = sha(run_dir / "best.keras")
    if not output["protected_real_unchanged"] or not output["trainer_sources_unchanged"]:
        raise RuntimeError("Protected model or verified trainer sources changed")
    save(ROOT / "docs/results/simulation_training.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/simulation_frozen_20261022_01")
    args = parser.parse_args()
    report(args.run_dir)
