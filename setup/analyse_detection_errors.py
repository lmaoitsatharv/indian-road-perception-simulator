"""Compare fixed detector resolutions and audit misses on labelled validation only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json
from setup.verify_detection_runtime import protection


def main(output, sizes):
    output = output.resolve()
    if output.parent != ROOT / "runs":
        raise ValueError("Use a fresh run directory")
    output.mkdir(exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(output / "settings"), MPLCONFIGDIR=str(output / "matplotlib"),
                      MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true")
    for folder in ("settings", "matplotlib"):
        (output / folder).mkdir()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou

    settings.update({key: False for key in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if key in settings})
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    dataset = ROOT / "dataset/idd117k_detection_v2"
    prepared = json.loads((dataset / "preparation.json").read_text())
    names = prepared["names"]
    before = protection()
    checkpoint = ROOT / "runs/detection_train_20261024_02/weights/best.pt"
    checkpoint_sha = file_sha(checkpoint)
    assert checkpoint_sha == "57f1d54b93c794677d26fd22773284de2752d878477bb9ac04a761cc7f035153"
    native_stats = {}
    for split in ("train", "val"):
        cache_path = dataset / "labels" / (split + ".cache")
        cache = np.load(cache_path, allow_pickle=True).item()  # Locally generated framework cache.
        assert cache["results"][0] == prepared["counts"][split] == len(cache["labels"])
        counts = np.zeros((len(names), 3), dtype=np.int64)
        image_counts = np.zeros(len(names), dtype=np.int64)
        for row in cache["labels"]:
            cls = row["cls"].reshape(-1).astype(int)
            height, width = row["shape"]
            wh = row["bboxes"][:, 2:4] * np.array([width, height])
            scale_area = np.sqrt(wh.prod(axis=1)) * 640 / max(width, height)
            bins = np.digitize(np.round(scale_area, 2), [32, 96])
            np.add.at(counts, (cls, bins), 1)
            image_counts[np.unique(cls)] += 1
        assert counts.sum(axis=1).tolist() == [prepared["objects"][split][name] for name in names]
        native_stats[split] = {"images": len(cache["labels"]), "cache_sha256": file_sha(cache_path),
            "size_bins": "sqrt(native box area) scaled to longest image side 640: <32, 32-96, >=96 pixels",
            "classes": {name: {"boxes": int(counts[i].sum()), "images": int(image_counts[i]),
                "small": int(counts[i, 0]), "medium": int(counts[i, 1]), "large": int(counts[i, 2])}
                for i, name in enumerate(names)}}
    save_json(output / "dataset_object_sizes.json", native_stats)

    thresholds = np.array([0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    class DiagnosticValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            DiagnosticValidator.latest = self
            self.gt_sizes = np.zeros((len(names), 3), dtype=np.int64)
            self.found_sizes = np.zeros((len(thresholds), len(names), 3), dtype=np.int64)
            self.pred_counts = np.zeros((len(thresholds), len(names)), dtype=np.int64)
            self.worst_people = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for index, pred in enumerate(preds):
                truth = self._prepare_batch(index, batch)
                true_cls = truth["cls"].cpu().numpy().astype(int)
                pred_cls = pred["cls"].cpu().numpy().astype(int)
                confidences = pred["conf"].cpu().numpy()
                native_boxes = ops.scale_boxes(truth["imgsz"], truth["bboxes"].clone(),
                    truth["ori_shape"], ratio_pad=truth["ratio_pad"])
                wh = (native_boxes[:, 2:4] - native_boxes[:, 0:2]).cpu().numpy()
                bins = np.digitize(np.round(np.sqrt(wh.prod(axis=1)) * 640 / max(truth["ori_shape"]), 2), [32, 96])
                np.add.at(self.gt_sizes, (true_cls, bins), 1)
                ious = box_iou(truth["bboxes"], pred["bboxes"]).cpu().numpy()
                ious *= true_cls[:, None] == pred_cls[None, :]
                for ti, threshold in enumerate(thresholds):
                    kept = np.flatnonzero(confidences >= threshold)
                    np.add.at(self.pred_counts[ti], pred_cls[kept], 1)
                    matches = np.array(np.nonzero(ious[:, kept] >= 0.5)).T
                    if len(matches) > 1:
                        scores = ious[matches[:, 0], kept[matches[:, 1]]]
                        matches = matches[scores.argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    found = matches[:, 0] if len(matches) else np.zeros(0, dtype=int)
                    np.add.at(self.found_sizes[ti], (true_cls[found], bins[found]), 1)
                    if threshold == 0.25:
                        misses = int((true_cls == 0).sum() - (true_cls[found] == 0).sum())
                        if misses:
                            record = {"image": Path(truth["im_file"]).relative_to(ROOT).as_posix(),
                                "missed_people": misses, "labelled_people": int((true_cls == 0).sum())}
                            self.worst_people.append(record)
                            self.worst_people = sorted(self.worst_people, key=lambda r: (-r["missed_people"], r["image"]))[:24]

    results = {}
    for size in sizes:
        print("Starting full validation diagnostic at resolution " + str(size), flush=True)
        started = time.monotonic()
        model = YOLO(str(checkpoint), task="detect")
        metrics = model.val(data=str(dataset / "data.yaml"), validator=DiagnosticValidator,
            imgsz=size, batch=8, workers=4, device=0, conf=0.001, iou=0.7, max_det=300,
            plots=False, save_json=False, project=str(output), name="validation_" + str(size), verbose=False)
        validator = DiagnosticValidator.latest
        assert validator.seen == prepared["counts"]["val"]
        assert validator.gt_sizes.sum(axis=1).tolist() == [prepared["objects"]["val"][name] for name in names]
        assert validator.gt_sizes.tolist() == [[native_stats["val"]["classes"][name][key]
            for key in ("small", "medium", "large")] for name in names]
        operating = {}
        for class_id in metrics.box.ap_class_index:
            i = list(metrics.box.ap_class_index).index(class_id)
            p, r = metrics.box.p_curve[i], metrics.box.r_curve[i]
            qualified = np.flatnonzero(p >= 0.75)
            best_i = int(qualified[np.argmax(r[qualified])]) if len(qualified) else None
            operating[names[int(class_id)]] = {
                "ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i]),
                "precision_f1_point": float(metrics.box.p[i]), "recall_f1_point": float(metrics.box.r[i]),
                "at_precision_at_least_0_75": None if best_i is None else {
                    "confidence": float(metrics.box.px[best_i]), "precision": float(p[best_i]), "recall": float(r[best_i])}}
        point_index = int(np.flatnonzero(thresholds == 0.25)[0])
        by_size = {}
        for i, name in enumerate(names):
            by_size[name] = {key: {"targets": int(validator.gt_sizes[i, k]),
                "matched": int(validator.found_sizes[point_index, i, k]),
                "recall": float(validator.found_sizes[point_index, i, k] / max(1, validator.gt_sizes[i, k]))}
                for k, key in enumerate(("small", "medium", "large"))}
        result = {"resolution": size, "images": validator.seen, "metrics": {key: float(value)
            for key, value in metrics.results_dict.items()}, "per_class": operating,
            "diagnostic_thresholds": thresholds.tolist(), "gt_class_size_counts": validator.gt_sizes.tolist(),
            "matched_class_size_counts_by_threshold": validator.found_sizes.tolist(),
            "predicted_class_counts_by_threshold": validator.pred_counts.tolist(),
            "size_recall_at_confidence_0_25_iou_0_5": by_size, "worst_person_images": validator.worst_people,
            "elapsed_seconds": round(time.monotonic() - started, 2)}
        results[str(size)] = result
        save_json(output / ("diagnostics_" + str(size) + ".json"), result)
        np.savez_compressed(output / ("confidence_curves_" + str(size) + ".npz"),
            confidence=metrics.box.px, precision=metrics.box.p_curve, recall=metrics.box.r_curve,
            classes=metrics.box.ap_class_index, ap=metrics.box.all_ap)
        save_json(output / "progress.json", {"completed_resolutions": list(results), "training_started": False})
        del model, validator, metrics
        torch.cuda.empty_cache()
    assert protection() == before and file_sha(checkpoint) == checkpoint_sha
    report = {"schema": "idd_detector_error_analysis_v1", "passed": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": checkpoint_sha,
        "counts": prepared["counts"], "resolutions": sizes,
        "aggregate": {key: value["metrics"] for key, value in results.items()},
        "protected_files_unchanged": True, "protected_sha256": before,
        "initial_detector_unchanged": True, "training_started": False, "optimizer_steps": 0,
        "official_test_used": False, "control_allowed": False,
        "scope": "same full project validation; resolution and confidence development analysis"}
    save_json(output / "analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 960])
    args = parser.parse_args()
    main(args.output, args.sizes)
