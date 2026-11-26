"""Camera-driving detector worker (WSL, idd-detect): deployed IDD YOLO11s, boxes only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.camera_driving_service import serve
from setup.prepare_simulation_dataset import sha

EXPECTED = {"schema": "carla_camera_driving_detector_v1", "scope": "native_carla_camera_driving",
            "training_allowed": False, "control_allowed": True}


def main(session_file):
    session = json.loads(Path(session_file).read_text())
    manifest = json.loads((ROOT / session["detector"]["manifest"]).read_text())
    for key, value in EXPECTED.items():
        if manifest.get(key) != value:
            raise ValueError("Detector driving manifest differs: " + key)
    checkpoint = ROOT / manifest["checkpoint"]
    if not checkpoint.resolve().is_relative_to((ROOT / "runs").resolve()) or sha(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Detector checkpoint path/hash differs")
    folder = ROOT / session["detector"]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    # Settings stay in this run folder; online features off (as the other detector scripts).
    os.environ.update(YOLO_CONFIG_DIR=str(folder / "settings"), MPLCONFIGDIR=str(folder / "matplotlib"),
                      MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true")
    (folder / "settings").mkdir(exist_ok=True)
    (folder / "matplotlib").mkdir(exist_ok=True)
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune", "raytune",
                                        "tensorboard", "dvc", "hub") if k in settings})
    options = manifest["inference"]

    def load(folder):
        model = YOLO(str(checkpoint), task="detect")
        names = [model.names[i] for i in range(len(model.names))]
        if names != manifest["classes"]:
            raise ValueError("Detector class order differs from the manifest")
        model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=options["imgsz"], conf=options["conf"], iou=options["iou"],
                      half=options["half"], device=0, max_det=options["max_det"], verbose=False)
        runtime = {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0),
                   "checkpoint_sha256": manifest["checkpoint_sha256"], **options}
        return model, runtime

    def infer(model, rgb):
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Ultralytics reads numpy images as BGR.
        result = model.predict(rgb[..., ::-1], imgsz=options["imgsz"], conf=options["conf"], iou=options["iou"],
                               half=options["half"], device=0, max_det=options["max_det"], verbose=False)[0].boxes
        boxes = np.concatenate([result.xyxy.cpu().numpy(), result.conf.cpu().numpy()[:, None],
                                result.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
        torch.cuda.synchronize()
        return {"inference_ms": (time.perf_counter() - started) * 1000, "boxes": int(len(boxes))}, boxes.tobytes()

    def fingerprint(model):
        digest = hashlib.sha256()
        for tensor in model.model.state_dict().values():
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    serve(session_file, "detector", load, infer, fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    main(parser.parse_args().session.resolve())
