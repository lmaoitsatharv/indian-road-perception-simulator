"""Fine-tune the detector on dataset/idd117k_detection_v4_landanimals (project decision, 29 Oct 2026; overnight run scheduled to finish before 08:00).

Reuses train_detection.train() unchanged (callbacks, FP32 recovery checkpoints, rare-class sampler, protected-file checks).
Only the v3-specific readiness gate is replaced by checks for the v4 view.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_detection
from idd_detection_data import file_sha, save_json

V4 = (ROOT / "dataset/idd117k_detection_v4_landanimals").resolve()


def v4_readiness(dataset=None, recipe_path=None, **_):
    recipe = json.loads(Path(recipe_path).read_text())
    verification = json.loads((V4 / "verification.json").read_text())
    runtime = json.loads((ROOT / recipe["runtime_verification"]).read_text())
    checks = {
        "v4_independent_verification_passed": verification.get("all_passed") is True,
        "recipe_points_at_v4": (ROOT / recipe["dataset"]).resolve() == V4,
        "initial_weights_hash_matches": file_sha(ROOT / recipe["initial_weights"]) == recipe["initial_weights_sha256"],
        "train_label_count_matches": sum(1 for _ in (V4 / "labels/train").glob("*.txt")) == recipe["expected_prepared_counts"]["train"],
        "val_label_count_matches": sum(1 for _ in (V4 / "labels/val").glob("*.txt")) == recipe["expected_prepared_counts"]["val"],
        "data_yaml_and_pairs_present": (V4 / "data.yaml").is_file() and (V4 / "pairs.jsonl").is_file(),
        "runtime_previously_verified": runtime.get("passed") is True and runtime.get("batch") == recipe["batch"] and runtime.get("imgsz") == recipe["imgsz"],
        "vehicle_control_not_allowed": recipe.get("control_allowed") is False,
    }
    return {"ready_for_explicit_training_start": all(checks.values()), "waiting_for": [k for k, v in checks.items() if not v],
            "checks": checks, "dataset": str(V4), "official_test_used": False}


train_detection.readiness = v4_readiness


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    code = 1
    try:
        train_detection.train(V4, a.output, a.recipe.resolve())
        shutil.copy2(Path(__file__), a.output.resolve() / "source" / Path(__file__).name)
        code = 0
    finally:
        save_json(a.output.resolve().parent / (a.output.name + "_exit.json"),
                  {"returncode": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
