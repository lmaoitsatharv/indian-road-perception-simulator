"""Freeze a DEV-only simulator choice; never grant shared real-image deployment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/model_retention_20261021_01"
RUNS = ["road_boundary_adaptation_20261020_02", "road_boundary_adaptation_20261020_03", "road_rehearsal_20261021_01"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def local(value):
    return Path("E:/" + value[7:]) if value.startswith("/mnt/e/") else Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "select"))
    args = parser.parse_args()
    if args.mode == "plan":
        plan = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "pool_runs": RUNS,
                "scope": "recorded_carla_only", "criterion": "Maximum pooled DEV road IoU, >0.5pp over original, using identical60 DEV frame hashes",
                "real_deployment": "Original locked model only; failures of any real guard prohibit a shared replacement",
                "fresh_seed137_pixels_scored": False,
                "historical_trial02": "Previously rejected by real-retention gates; eligible only for the separate simulator path",
                "independent_confirmation": "Compare frozen choice, locked original and prior03 on seed137 afterward; never reselect on it"}
        save_new(OUT / "simulation_pool_plan.json", plan)
        print(json.dumps(plan))
        return
    plan = load(OUT / "simulation_pool_plan.json")
    pool, expected_dev, baseline_dev = [], None, None
    for run_name in plan["pool_runs"]:
        run = ROOT / "runs" / run_name
        state, manifest, history = (load(run / name) for name in ("run_status.json", "manifest.json", "history.json"))
        if state["phase"] != "completed" or not state["protected_files_unchanged"]:
            raise ValueError("Incomplete or unprotected trial: " + run_name)
        dev = [(r["name"], r["image_sha256"], r["mask_sha256"]) for r in manifest["development"]]
        if len(dev) != 60 or (expected_dev is not None and dev != expected_dev):
            raise ValueError("Compared trials have different DEV images/masks/order")
        expected_dev = dev
        for record in manifest["development"]:
            for key in ("image", "mask"):
                if sha(local(record[key])) != record[key + "_sha256"]:
                    raise ValueError("Changed DEV source")
        base = load(run / "evaluations/baseline_development/metrics.json")["metrics"]["road_iou"]
        if baseline_dev is not None and base != baseline_dev:
            raise ValueError("Baseline DEV comparison is not identical")
        baseline_dev = base
        for row in history:
            epoch = row["epoch"]
            model = run / f"epoch_{epoch:02d}.keras"
            fingerprint = sha(model)
            if fingerprint != row["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after evaluation")
            report = run / f"evaluations/epoch_{epoch:02d}_development/metrics.json"
            score = load(report)["metrics"]["road_iou"]
            pool.append({"run": run_name, "epoch": epoch, "checkpoint": model.relative_to(ROOT).as_posix(),
                         "checkpoint_sha256": fingerprint, "dev_road_iou": score,
                         "development_metrics_sha256": sha(report),
                         "real_retention_passed_in_trial": row.get("real_retention_passed", row.get("real_retention_guard_passed")),
                         "strict_current_shared_checks_available": "strict_gates" in row})
    candidates = [r for r in pool if r["dev_road_iou"] > baseline_dev + .005]
    if not candidates:
        raise ValueError("No simulator candidate meets the predeclared DEV improvement")
    chosen = max(candidates, key=lambda r: r["dev_road_iou"])
    selection = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(OUT / "simulation_pool_plan.json"),
                 "chosen": chosen, "pool": pool, "baseline_development_iou": baseline_dev,
                 "fresh_seed137_pixels_scored": False, "scope": "recorded_carla_only",
                 "shared_real_deployment_approved": False,
                 "historical_model_reused": chosen["run"] != "road_rehearsal_20261021_01"}
    selection_file = OUT / "simulation_selection.json"
    save_new(selection_file, selection)
    lock = load(ROOT / "models/segmentation_final_lock.json")
    spec = {"schema": "simulator_road_model_v1", "scope": "recorded_carla_only", "training_allowed": False,
            "input_shape": [512, 512, 3], "normalization": "RGB /127.5 -1 exactly once", "threshold": .5,
            "shared_real_deployment_approved": False, "base_checkpoint_sha256": lock["checkpoint_sha256"],
            "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["checkpoint_sha256"],
            "selection_artifact": selection_file.relative_to(ROOT).as_posix(), "selection_artifact_sha256": sha(selection_file)}
    save_new(OUT / "simulation_model.json", spec)
    print(json.dumps({"chosen": chosen, "baseline_dev": baseline_dev, "shared_real_deployment_approved": False}))


if __name__ == "__main__":
    main()
