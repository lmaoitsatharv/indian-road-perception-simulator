"""One fixed-checkpoint evaluation of the reserved simulation test split.

Freezes checkpoint/protocol before test decoding. Saves every binary prediction
and per-image counts for independent recounting. Never trains or tunes a model.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import tensorflow as tf

from simulation_data import make_dataset
from simulation_model import MaskedBCEDiceLoss, MaskedRoadIoU, MaskedRoadDice, pack_targets
from train_simulation import ROOT, MANIFEST_SHA, configure_gpu, dataset_checks, utc
from setup.prepare_simulation_dataset import save, sha, protected


def scores(counts):
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    return {"road_iou": tp/(tp+fp+fn) if tp+fp+fn else 1.,
            "road_dice": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "pixel_accuracy": (tp+tn)/(tp+fp+fn+tn)}


def weight_hash(model):
    digest = hashlib.sha256()
    for weight in model.weights:
        digest.update(weight.numpy().tobytes())
    return digest.hexdigest()


def evaluate(dataset, run_dir, output):
    dataset, run_dir, output = dataset.resolve(), run_dir.resolve(), output.resolve()
    if output.parent != ROOT / "runs" or not output.name.startswith("simulation_test_"):
        raise ValueError("Use a new local runs/simulation_test_* output directory")
    config = json.loads((run_dir / "run_config.json").read_text())
    if config["status"] != "completed" or config["model_domain"] != "simulation_only":
        raise ValueError("Select a completed separate simulation model before scoring test")
    receipt = run_dir / "test_evaluation.json"
    if receipt.exists() or output.exists():
        raise FileExistsError("Test evaluation already attempted; preserve its evidence")
    checkpoint = run_dir / "best.keras"
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != config["best_checkpoint_sha256"]:
        raise RuntimeError("The validation-selected checkpoint changed")
    history = json.loads((run_dir / "history.json").read_text())
    selected = max(range(len(history["val_road_iou"])), key=lambda i: history["val_road_iou"][i]) + 1
    if selected != config["best_epoch"]:
        raise RuntimeError("Checkpoint selection disagrees with validation history")
    if not all(sha(ROOT / name) == value for name,value in config["source_sha256"].items()):
        raise RuntimeError("The verified training/loader sources changed")
    pairs, real_before = dataset_checks(dataset)  # Names only; no image contents.
    with (dataset / "splits/pair_splits.csv").open(newline="") as stream:
        metadata = {int(row["id"]): row for row in csv.DictReader(stream) if row["split"] == "test"}
    assert len(metadata) == len(pairs["test"]) == 500
    output.mkdir()
    (output / "predictions").mkdir()
    protocol = {"locked_utc": utc(), "model_domain": "simulation_only",
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": checkpoint_sha,
                "selection": "Maximum simulation validation road IoU; user requested testing this best checkpoint",
                "selected_epoch": selected, "optional_fine_tuning_performed": False,
                "threshold": .5, "comparison": "probability >= threshold", "batch_size": 4,
                "split": "test", "expected_images": 500, "manifest_sha256": MANIFEST_SHA,
                "normalization": "MobileNetV2 once to [-1,1]", "augmentation": "none",
                "ignored_pixels": "valid == 0 excluded from loss and all confusion counts",
                "aggregation": "Global integer TP/FP/FN/TN; mean per-image masked BCE+soft Dice loss",
                "prediction_encoding": "uint8 PNG 0/1; predictions retained on all pixels; validity applied when scoring",
                "no_test_tuning": True, "evaluator_sha256": sha(Path(__file__)),
                "training_source_sha256": config["source_sha256"], "protected_real_sha256": real_before}
    save(output / "protocol.json", protocol)
    state = {"status": "running", "started_utc": utc(),
             "output": output.relative_to(ROOT).as_posix(), "protocol_sha256": sha(output / "protocol.json"),
             "test_model_evaluation_started": True, "test_images_read": 0}
    # Exclusive creation prevents accidental repeated selection/test scoring.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
    save(output / "status.json", state)
    started = time.perf_counter()
    result = None
    try:
        gpu = configure_gpu()
        model = tf.keras.models.load_model(checkpoint, compile=False)
        before_weights = weight_hash(model)
        infer = tf.function(lambda image: model(image, training=False),
                            input_signature=[tf.TensorSpec([None,512,512,3],tf.float32)])
        objective = MaskedBCEDiceLoss()
        iou, dice = MaskedRoadIoU(), MaskedRoadDice()
        counts = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
        valid_pixels = 0
        loss_sum = 0.
        seen = 0
        by_group = {key: {} for key in ("episode_id", "layout", "weather", "case")}
        fields = ["id", "episode_id", "layout", "weather", "case", "tp", "fp", "fn", "tn",
                  "valid_pixels", "ignored_pixels", "road_iou", "road_dice", "masked_bce_dice", "prediction_sha256"]
        with (output / "per_image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for batch_index, triple in enumerate(make_dataset(pairs["test"], batch_size=4)):
                image, packed = pack_targets(*triple)
                probability = infer(image)
                array = probability.numpy()
                if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                    raise RuntimeError("Nonfinite/out-of-range model probability")
                iou.update_state(packed, probability)
                dice.update_state(packed, probability)
                loss = objective.call(packed, probability).numpy()
                if not np.isfinite(loss).all():
                    raise RuntimeError("Nonfinite test loss")
                truth, valid = triple[1].numpy() >= .5, triple[2].numpy() >= .5
                predictions = array >= .5
                for j in range(len(array)):
                    record = pairs["test"][seen]
                    meta = metadata[record["id"]]
                    t, p, v = truth[j,...,0], predictions[j,...,0], valid[j,...,0]
                    tally = {"tp": int(np.count_nonzero(t & p & v)),
                             "fp": int(np.count_nonzero(~t & p & v)),
                             "fn": int(np.count_nonzero(t & ~p & v)),
                             "tn": int(np.count_nonzero(~t & ~p & v))}
                    trusted = int(np.count_nonzero(v))
                    assert sum(tally.values()) == trusted > 0
                    path = output / "predictions" / f"Prediction_{record['id']}.png"
                    Image.fromarray(p.astype(np.uint8)).save(path)
                    row = {k: meta[k] for k in ("id", "episode_id", "layout", "weather", "case")}
                    row.update(tally, valid_pixels=trusted, ignored_pixels=v.size-trusted,
                               road_iou=scores(tally)["road_iou"], road_dice=scores(tally)["road_dice"],
                               masked_bce_dice=float(loss[j]), prediction_sha256=sha(path))
                    writer.writerow(row)
                    for key, value in tally.items():
                        counts[key] += value
                    for field, groups in by_group.items():
                        group = groups.setdefault(meta[field], {"images": 0, "valid_pixels": 0,
                                                               **dict.fromkeys(counts, 0)})
                        group["images"] += 1
                        group["valid_pixels"] += trusted
                        for key,value in tally.items():
                            group[key] += value
                    valid_pixels += trusted
                    loss_sum += float(loss[j])
                    seen += 1
                if batch_index == 0 or (batch_index + 1) % 10 == 0 or seen == 500:
                    stream.flush()
                    state.update(test_images_read=seen, updated_utc=utc(), elapsed_seconds=time.perf_counter()-started)
                    save(output / "status.json", state)
                    print(f"Test images: {seen}/500", flush=True)
        assert seen == 500 and sum(counts.values()) == valid_pixels
        metrics = scores(counts)
        np.testing.assert_allclose(iou.result().numpy(), metrics["road_iou"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(dice.result().numpy(), metrics["road_dice"], atol=1e-12, rtol=0)
        for groups in by_group.values():
            for group in groups.values():
                group.update(scores(group))
        assert weight_hash(model) == before_weights
        result = {"status": "completed", "completed_utc": utc(), "images": seen,
                  "checkpoint": protocol["checkpoint"], "checkpoint_sha256": checkpoint_sha,
                  "selected_epoch": selected, "threshold": .5, "manifest_sha256": MANIFEST_SHA,
                  "counts": counts, "metrics": {**metrics, "masked_bce_dice": loss_sum/seen},
                  "valid_pixels": valid_pixels, "total_pixels": seen*512*512,
                  "ignored_pixels": seen*512*512-valid_pixels, "valid_coverage": valid_pixels/(seen*512*512),
                  "by_group": by_group, "tensorflow_metrics_match_numpy_counts": True,
                  "in_memory_model_weights_unchanged": True,
                  "elapsed_seconds": time.perf_counter()-started, "gpu": gpu,
                  "test_split_now_inspected": True, "real_test_images_read": False,
                  "independent_verification": "pending"}
        state.update(status="completed", completed_utc=utc(), test_images_read=seen)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        unchanged = (protected() == real_before and sha(checkpoint) == checkpoint_sha and
                     sha(dataset / "splits/pair_splits.csv") == MANIFEST_SHA)
        state["protected_real_checkpoint_and_manifest_unchanged"] = unchanged
        if not unchanged:
            state.update(status="failed", error="Protected files changed during evaluation")
        save(output / "status.json", state)
        save(receipt, state)
        if result is not None:
            result["protected_real_checkpoint_and_manifest_unchanged"] = unchanged
            if not unchanged:
                result["status"] = "failed"
            save(output / "test_results.json", result)
            print(json.dumps(result, indent=2), flush=True)
        if not unchanged:
            raise RuntimeError("Protected files changed during test evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/simulation_india_v5")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/simulation_frozen_20261022_01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.run_dir, args.output)
