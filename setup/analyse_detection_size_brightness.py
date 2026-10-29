"""Review the current 1280 detector by class, object size and image brightness on project validation only.

Inference only: no training, no optimizer, no official test images, no default or checkpoint change.
Use --limit N for a small evenly spaced subset (smoke test); omit it for the full validation set.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json
from setup.verify_detection_runtime import protection

CHECKPOINT = ROOT / "runs/detection_train_20261025_02/weights/best.pt"
CHECKPOINT_SHA = "f7bb7e4c5959520d797dcde5547fdcded9ece1e9651a3e7bd6ffa2273749182c"
MATCHED_RESULT = ROOT / "runs/detection_pilot_recovery_20261025_01/matched_final_1280.json"
DATASET = ROOT / "dataset/idd117k_detection_v3"
THRESHOLDS = [0.10, 0.25, 0.50]
OPERATING = 1  # index of 0.25 in THRESHOLDS
SIZE_NAMES = ("small", "medium", "large")
SIZE_EDGES = [32, 96]  # sqrt(native box area) scaled to a 640 px longest side, same bins as analyse_detection_errors.py
BRIGHTNESS_SCHEMES = {  # quantile cut points over the validated images, and bucket names
    "terciles": ([1 / 3, 2 / 3], ("darker_third", "middle_third", "brighter_third")),
    "tails": ([0.10, 0.90], ("darkest_10pct", "middle_80pct", "brightest_10pct"))}


def make_subset(output, limit):
    """Copy an evenly spaced subset so the real label cache is never rewritten."""
    images = sorted((DATASET / "images/val").iterdir())
    step = max(1, len(images) // limit)
    chosen = images[::step][:limit]
    for folder in ("images/val", "labels/val"):
        (output / "subset" / folder).mkdir(parents=True)
    counts = {}
    for image in chosen:
        label = DATASET / "labels/val" / (image.stem + ".txt")
        shutil.copy2(image, output / "subset/images/val" / image.name)
        shutil.copy2(label, output / "subset/labels/val" / label.name)
        for line in label.read_text().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
    names = json.loads((DATASET / "preparation.json").read_text())["names"]
    yaml = output / "subset/data.yaml"
    yaml.write_text("path: " + (output / "subset").as_posix() + "\ntrain: images/val\nval: images/val\nnames:\n" +
                    "".join("  %d: %s\n" % (i, n) for i, n in enumerate(names)))
    return yaml, len(chosen), [counts.get(i, 0) for i in range(len(names))]


def main(output, limit, batch, workers, imgsz=1280, augment=False, nms_iou=0.7, checkpoint=None, checkpoint_sha=None, val_root=None):
    global CHECKPOINT, CHECKPOINT_SHA
    if checkpoint is not None:
        CHECKPOINT, CHECKPOINT_SHA = checkpoint.resolve(), checkpoint_sha
    output = output.resolve()
    if output.parent != ROOT / "runs":
        raise ValueError("Use a fresh run directory directly under runs/")
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
    prepared = json.loads((DATASET / "preparation.json").read_text())
    names = prepared["names"]
    nc, ns, nt = len(names), len(SIZE_NAMES), len(THRESHOLDS)
    before = protection()
    cache_before = file_sha(DATASET / "labels/val.cache")
    assert file_sha(CHECKPOINT) == CHECKPOINT_SHA
    thresholds = np.array(THRESHOLDS)

    if val_root is not None:
        # another validation folder with images/ and labels/ (e.g. v4 val_land or holdout_dats); counts come from its labels
        val_root = val_root.resolve()
        data_yaml = output / "data.yaml"
        data_yaml.write_text("path: " + val_root.as_posix() + "\ntrain: images\nval: images\nnames:\n" +
                             "".join("  %d: %s\n" % (i, n) for i, n in enumerate(names)))
        expected_images = sum(1 for _ in (val_root / "images").glob("*.jpg"))
        counts = [0] * len(names)
        for label in (val_root / "labels").glob("*.txt"):
            for line in label.read_text().split("\n"):
                if line.strip():
                    counts[int(line.split()[0])] += 1
        expected_objects = counts
    elif limit:
        data_yaml, expected_images, expected_objects = make_subset(output, limit)
    else:
        data_yaml, expected_images = DATASET / "data.yaml", prepared["counts"]["val"]
        expected_objects = [prepared["objects"]["val"][name] for name in names]

    def size_bins(boxes, truth):
        native = ops.scale_boxes(truth["imgsz"], boxes.clone(), truth["ori_shape"], ratio_pad=truth["ratio_pad"])
        wh = (native[:, 2:4] - native[:, 0:2]).clamp(min=0).cpu().numpy()
        return np.digitize(np.round(np.sqrt(wh.prod(axis=1)) * 640 / max(truth["ori_shape"]), 2), SIZE_EDGES)

    class ReviewValidator(DetectionValidator):
        latest = None

        def init_metrics(self, model):
            super().init_metrics(model)
            ReviewValidator.latest = self
            self.files, self.luma = [], []
            self.gt, self.found, self.pred_tp, self.pred_fp = [], [], [], []
            # rows: true class; columns: predicted class, then 'nothing overlapping', then 'same-class box already used'
            self.confusion = np.zeros((ns, nc, nc + 2), dtype=np.int64)
            self.last_progress = 0.0

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                truth = self._prepare_batch(si, batch)
                image = batch["img"][si]
                top, left = int(round(float(truth["ratio_pad"][1][1]) - 0.1)), int(round(float(truth["ratio_pad"][1][0]) - 0.1))
                crop = image[:, top:image.shape[1] - top, left:image.shape[2] - left]
                luma = float((0.299 * crop[0] + 0.587 * crop[1] + 0.114 * crop[2]).mean() * 255)
                self.files.append(Path(truth["im_file"]).name)
                self.luma.append(luma)
                true_cls = truth["cls"].cpu().numpy().astype(int)
                pred_cls = pred["cls"].cpu().numpy().astype(int)
                conf = pred["conf"].cpu().numpy()
                gt_bins = size_bins(truth["bboxes"], truth)
                pred_bins = size_bins(pred["bboxes"], truth)
                gt = np.zeros((nc, ns), dtype=np.int32)
                np.add.at(gt, (true_cls, gt_bins), 1)
                found = np.zeros((nt, nc, ns), dtype=np.int32)
                tp = np.zeros((nt, nc, ns), dtype=np.int32)
                fp = np.zeros((nt, nc, ns), dtype=np.int32)
                all_ious = box_iou(truth["bboxes"], pred["bboxes"]).cpu().numpy()
                same = all_ious * (true_cls[:, None] == pred_cls[None, :])
                for ti, threshold in enumerate(thresholds):
                    kept = np.flatnonzero(conf >= threshold)
                    matches = np.array(np.nonzero(same[:, kept] >= 0.5)).T
                    if len(matches) > 1:
                        scores = same[matches[:, 0], kept[matches[:, 1]]]
                        matches = matches[scores.argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    gt_hit = matches[:, 0] if len(matches) else np.zeros(0, dtype=int)
                    pred_hit = kept[matches[:, 1]] if len(matches) else np.zeros(0, dtype=int)
                    np.add.at(found[ti], (true_cls[gt_hit], gt_bins[gt_hit]), 1)
                    is_tp = np.zeros(len(conf), dtype=bool)
                    is_tp[pred_hit] = True
                    np.add.at(tp[ti], (pred_cls[kept][is_tp[kept]], pred_bins[kept][is_tp[kept]]), 1)
                    np.add.at(fp[ti], (pred_cls[kept][~is_tp[kept]], pred_bins[kept][~is_tp[kept]]), 1)
                    if ti == OPERATING:
                        hit = np.zeros(len(true_cls), dtype=bool)
                        hit[gt_hit] = True
                        for g in range(len(true_cls)):
                            if hit[g]:
                                self.confusion[gt_bins[g], true_cls[g], true_cls[g]] += 1
                                continue
                            candidates = kept[all_ious[g, kept] >= 0.5] if len(kept) else kept
                            if not len(candidates):
                                self.confusion[gt_bins[g], true_cls[g], nc] += 1
                                continue
                            best = candidates[np.argmax(all_ious[g, candidates])]
                            column = nc + 1 if pred_cls[best] == true_cls[g] else pred_cls[best]
                            self.confusion[gt_bins[g], true_cls[g], column] += 1
                self.gt.append(gt.astype(np.int16))
                self.found.append(found.astype(np.int16))
                self.pred_tp.append(tp.astype(np.int16))
                self.pred_fp.append(fp.astype(np.int16))
            if time.monotonic() - self.last_progress > 20:
                save_json(output / "progress.json", {"images_seen": int(self.seen), "images_expected": expected_images,
                    "checked_utc": datetime.now(timezone.utc).isoformat(), "training_started": False})
                self.last_progress = time.monotonic()

    started = time.monotonic()
    model = YOLO(str(CHECKPOINT), task="detect")
    metrics = model.val(data=str(data_yaml), validator=ReviewValidator, imgsz=imgsz, batch=batch, workers=workers,
        device=0, half=False, conf=0.001, iou=nms_iou, max_det=300, augment=augment, plots=False, save_json=False,
        project=str(output), name="validation_%d" % imgsz, verbose=False)
    matched_settings = imgsz == 1280 and not augment and nms_iou == 0.7 and checkpoint is None and val_root is None
    elapsed = time.monotonic() - started
    v = ReviewValidator.latest
    assert v.seen == expected_images == len(v.files) == len(set(v.files))
    gt = np.stack(v.gt).astype(np.int64)
    found = np.stack(v.found).astype(np.int64)
    tp = np.stack(v.pred_tp).astype(np.int64)
    fp = np.stack(v.pred_fp).astype(np.int64)
    luma = np.array(v.luma)
    assert gt.sum(axis=(0, 2)).tolist() == expected_objects, "ground-truth counts differ from the independent label count"
    assert int(v.confusion.sum()) == int(gt.sum())
    np.savez_compressed(output / "per_image_counts.npz", files=np.array(v.files), luma=luma, gt=gt.astype(np.int16),
        found=found.astype(np.int16), pred_tp=tp.astype(np.int16), pred_fp=fp.astype(np.int16), confusion=v.confusion)

    cuts = {scheme: np.quantile(luma, quantiles) for scheme, (quantiles, _) in BRIGHTNESS_SCHEMES.items()}
    bright = {scheme: np.digitize(luma, cuts[scheme]) for scheme in BRIGHTNESS_SCHEMES}

    def rate(num, den):
        return None if den == 0 else float(num / den)

    def cell(mask, ti, ci):
        g, f = gt[mask][:, ci].sum(axis=0), found[mask][:, ti, ci].sum(axis=0)
        p, q = tp[mask][:, ti, ci].sum(axis=0), fp[mask][:, ti, ci].sum(axis=0)
        return g, f, p, q

    everything = np.ones(len(luma), dtype=bool)
    result = {}
    for ti, threshold in enumerate(THRESHOLDS):
        by_class = {}
        for ci, name in enumerate(names):
            g, f, p, q = cell(everything, ti, ci)
            entry = {"targets": int(g.sum()), "recall": rate(f.sum(), g.sum()),
                     "predictions": int(p.sum() + q.sum()), "precision": rate(p.sum(), p.sum() + q.sum()), "by_size": {}, "by_brightness": {}}
            for si, size in enumerate(SIZE_NAMES):
                entry["by_size"][size] = {"targets": int(g[si]), "found": int(f[si]), "recall": rate(f[si], g[si]),
                    "predictions": int(p[si] + q[si]), "true_positive": int(p[si]), "precision": rate(p[si], p[si] + q[si])}
            for scheme, (_, labels) in BRIGHTNESS_SCHEMES.items():
                entry["by_brightness"][scheme] = {}
                for bi, label in enumerate(labels):
                    g, f, p, q = cell(bright[scheme] == bi, ti, ci)
                    entry["by_brightness"][scheme][label] = {"images": int((bright[scheme] == bi).sum()), "targets": int(g.sum()),
                        "found": int(f.sum()), "recall": rate(f.sum(), g.sum()), "predictions": int(p.sum() + q.sum()),
                        "precision": rate(p.sum(), p.sum() + q.sum()),
                        "recall_by_size": {size: rate(f[si], g[si]) for si, size in enumerate(SIZE_NAMES)},
                        "targets_by_size": {size: int(g[si]) for si, size in enumerate(SIZE_NAMES)}}
            by_class[name] = entry
        result["conf_%.2f" % threshold] = by_class

    confusion = {size: {names[ci]: {**{names[cj]: int(v.confusion[si, ci, cj]) for cj in range(nc)},
                                    "nothing_overlapping": int(v.confusion[si, ci, nc]),
                                    "same_class_box_already_used": int(v.confusion[si, ci, nc + 1])}
                        for ci in range(nc)} for si, size in enumerate(SIZE_NAMES)}
    framework = {str(k): float(val) for k, val in metrics.results_dict.items()}
    per_class_ap = {names[int(c)]: {"ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i])}
                    for i, c in enumerate(metrics.box.ap_class_index)}
    reference = json.loads(MATCHED_RESULT.read_text())["metrics"] if not limit and matched_settings else None
    differences = None if reference is None else {k: framework[k] - reference[k] for k in reference}
    report = {"schema": "idd_detector_size_brightness_review_v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ("evenly spaced validation subset (smoke test, not a result)" if limit else
                  "other evaluation folder: " + str(val_root) if val_root is not None else "full project validation"),
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
        "images": int(v.seen), "targets": int(gt.sum()), "imgsz": imgsz, "test_time_augmentation": augment, "nms_iou": nms_iou,
        "precision": "GPU FP32", "batch": batch, "workers": workers, "per_class_ap": per_class_ap,
        "operating_point": "confidence >= 0.25 (and 0.10 / 0.50), IoU >= 0.5, same class, one-to-one greedy matching; not AP",
        "size_bins": "sqrt(native box area) scaled to a 640 px longest side: small < 32, medium 32-96, large >= 96",
        "brightness": {"definition": "mean luma (0-255) of the letterbox-cropped model input; image-level proxy only, not time-of-day or weather",
            "luma_min": float(luma.min()), "luma_max": float(luma.max()), "luma_median": float(np.median(luma)),
            "schemes": {scheme: {"quantiles": quantiles, "cut_points": [float(c) for c in cuts[scheme]],
                "images_per_bucket": {label: int((bright[scheme] == i).sum()) for i, label in enumerate(labels)}}
                for scheme, (quantiles, labels) in BRIGHTNESS_SCHEMES.items()}},
        "framework_metrics": framework, "reference_metrics_matched_final_1280": reference,
        "framework_minus_reference": differences,
        "framework_matches_reference": None if differences is None else all(abs(d) < 5e-4 for d in differences.values()),
        "results": result, "confusion_conf_0_25": confusion,
        "elapsed_seconds": round(elapsed, 2), "seconds_per_image": round(elapsed / v.seen, 4),
        "speed_ms_per_image": metrics.speed,
        "training_started": False, "optimizer_steps": 0, "official_test_used": False, "control_allowed": False,
        "protected_files_unchanged": protection() == before, "checkpoint_unchanged": file_sha(CHECKPOINT) == CHECKPOINT_SHA,
        "val_cache_unchanged": file_sha(DATASET / "labels/val.cache") == cache_before}
    save_json(output / "analysis.json", report)
    print(json.dumps({k: report[k] for k in ("scope", "images", "targets", "elapsed_seconds", "seconds_per_image",
        "framework_matches_reference", "protected_files_unchanged", "checkpoint_unchanged", "val_cache_unchanged")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280, help="inference-only setting probe; 1280 is the matched default")
    parser.add_argument("--augment", action="store_true", help="Ultralytics test-time augmentation (flips/scales)")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=Path, help="evaluate another checkpoint (its SHA-256 must be given)")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--val-root", type=Path, help="score another folder with images/ and labels/ instead of the project validation")
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        parser.error("--checkpoint and --checkpoint-sha256 go together")
    if args.val_root is not None and args.limit:
        parser.error("--val-root and --limit cannot be combined")
    main(args.output, args.limit, args.batch, args.workers, args.imgsz, args.augment, args.nms_iou, args.checkpoint, args.checkpoint_sha256,
         args.val_root)
