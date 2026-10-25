"""Verify saved detector checkpoints without training or rescoring the dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(name, directory):
    if Path(name).name != name or not name.startswith("detection_train_"):
        raise ValueError("Expected a local detector run name")
    run = ROOT / "runs" / name
    if Path(directory).name != directory or not directory.startswith("report_verification"):
        raise ValueError("Expected a fresh verification directory inside this run")
    output = run / directory
    output.mkdir(exist_ok=False)
    os.environ.update({
        "YOLO_CONFIG_DIR": str(output / "settings"),
        "MPLCONFIGDIR": str(output / "matplotlib"),
        "MPLBACKEND": "Agg", "YOLO_AUTOINSTALL": "False", "WANDB_DISABLED": "true",
    })
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir()
    Path(os.environ["MPLCONFIGDIR"]).mkdir()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.utils.metrics import Metric

    settings.update({key: False for key in (
        "sync", "wandb", "comet", "clearml", "mlflow", "neptune", "raytune",
        "tensorboard", "dvc", "hub") if key in settings})
    assert Path(settings.file).resolve().is_relative_to(output.resolve())
    expected = json.loads((ROOT / "dataset/idd117k_detection_v2/preparation.json").read_text())["names"]
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    checkpoints = {}
    before = {}
    for label in ("best", "last"):
        path = run / "weights" / (label + ".pt")
        before[label] = sha(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        network = checkpoint["model"]
        names = [network.names[index] for index in range(len(network.names))]
        assert names == expected
        tensors = list(network.state_dict().values())
        assert all(bool(torch.isfinite(value).all()) for value in tensors)
        history = checkpoint.get("train_results", {})
        history_last = {key: value[-1] for key, value in history.items() if len(value)}
        checkpoints[label] = {
            "path": path.relative_to(ROOT).as_posix(), "sha256": before[label],
            "bytes": path.stat().st_size, "stored_epoch": checkpoint.get("epoch"),
            "checkpoint_date": checkpoint.get("date"), "version": checkpoint.get("version"),
            "names": names, "state_tensors": len(tensors), "all_tensors_finite": True,
            "optimizer_present": checkpoint.get("optimizer") is not None,
            "best_fitness": checkpoint.get("best_fitness"),
            "train_metrics": checkpoint.get("train_metrics"), "history_last_row": history_last,
            "history_rows": len(history.get("epoch", [])),
        }
        del checkpoint, network, tensors

    detector = YOLO(str(run / "weights/best.pt"), task="detect")
    detector.model.to("cuda:0").float().eval().requires_grad_(False)
    with torch.inference_mode():
        prediction = detector.model(torch.zeros((1, 3, 640, 640), device="cuda:0"))
        raw = prediction[0] if isinstance(prediction, tuple) else prediction
        assert tuple(raw.shape) == (1, 4 + len(expected), 8400)
        assert bool(torch.isfinite(raw).all())
        raw_shape = list(raw.shape)
    assert all(parameter.grad is None for parameter in detector.model.parameters())
    image = next(path for path in (ROOT / "dataset/idd117k_detection_v2/images/val").iterdir()
                 if path.is_file() and path.suffix.lower() in (".jpg", ".png", ".jpeg"))
    result = detector.predict(str(image), imgsz=640, device=0, conf=0.25,
                              verbose=False, save=False)[0]
    boxes = result.boxes.xyxy.cpu()
    assert bool(torch.isfinite(boxes).all())
    height, width = result.orig_shape
    assert bool((boxes[:, 0::2] >= 0).all() and (boxes[:, 0::2] <= width + 0.001).all())
    assert bool((boxes[:, 1::2] >= 0).all() and (boxes[:, 1::2] <= height + 0.001).all())
    assert all(parameter.grad is None for parameter in detector.model.parameters())
    assert before == {label: sha(run / "weights" / (label + ".pt")) for label in before}
    # Preserve the pinned local implementation used to interpret checkpoint selection.
    fitness_source = inspect.getsource(getattr(Metric.fitness, "fget", Metric.fitness))
    (output / "metric_fitness_source.txt").write_text(fitness_source, encoding="utf-8")
    report = {
        "schema": "idd_trained_checkpoint_verification_v1", "passed": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0), "checkpoints": checkpoints,
        "gpu_forward_shape": raw_shape, "gpu_forward_finite": True,
        "validation_image_smoke_check": {
            "image": image.relative_to(ROOT).as_posix(), "sha256": sha(image),
            "original_shape": list(result.orig_shape), "boxes": len(boxes),
            "confidence": 0.25, "finite_native_coordinate_boxes": True,
            "scope": "reload and inference smoke check; not an accuracy evaluation",
        },
        "checkpoint_files_unchanged": True, "additional_optimizer_steps": 0,
        "official_test_used": False, "model_control": False,
    }
    save(output / "checkpoint_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--directory", default="report_verification")
    args = parser.parse_args()
    main(args.run_name, args.directory)
