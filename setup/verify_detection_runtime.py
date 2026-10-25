"""Forward-only GPU readiness check using official COCO weights, never trains."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.prepare_simulation_dataset import protected


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def protection():
    return {**protected(), "simulation_checkpoint": sha(ROOT / "runs/simulation_frozen_20261022_01/best.keras"),
            "rule_driver": sha(ROOT / "integration/india_behavior.py")}


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "runtime_verification.json"
    if report_path.exists():
        raise FileExistsError("Preserve previous runtime proof; use a fresh output")
    before = protection()
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MPLBACKEND"] = "Agg"
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO, settings
    assert Path(settings.file).resolve().is_relative_to(Path(os.environ["YOLO_CONFIG_DIR"]).resolve())
    settings.update({k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune", "raytune", "tensorboard", "dvc", "hub") if k in settings})
    weights = ROOT / "models/cache/detection/yolo11s.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    if not weights.exists():
        partial = weights.with_suffix(".pt.partial")
        urllib.request.urlretrieve(url, partial)
        partial.replace(weights)
    weight_sha = sha(weights)
    assert torch.cuda.is_available(), "CUDA GPU is required for this readiness check"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    model = YOLO(str(weights), task="detect")
    assert len(model.names) == 80 and model.task == "detect"
    model.model.to("cuda:0").eval()
    model.model.requires_grad_(False)
    batch = torch.zeros((4,3,640,640), device="cuda:0", dtype=torch.float32)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in range(23):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = model.model(batch)
            end.record()
            torch.cuda.synchronize()
            raw = prediction[0] if isinstance(prediction, tuple) else prediction
            assert raw.shape[:2] == (4,84) and torch.isfinite(raw).all()
            if index >= 3:
                timings.append(float(start.elapsed_time(end)))
    assert all(p.grad is None for p in model.model.parameters())
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    # Exercise native-size -> letterbox -> decoded boxes without evaluating a
    # real image or treating COCO output as an IDD fine-tuned result.
    probes = [np.zeros((360,640,3),np.uint8), np.full((720,1280,3),127,np.uint8)]
    results = model.predict(probes, imgsz=640, device=0, conf=.25, verbose=False, save=False)
    for image, result in zip(probes,results):
        assert result.orig_shape == image.shape[:2]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        assert np.isfinite(boxes).all()
        if len(boxes):
            assert np.all(boxes[:,0::2]>=0) and np.all(boxes[:,0::2]<=image.shape[1]+1e-3)
            assert np.all(boxes[:,1::2]>=0) and np.all(boxes[:,1::2]<=image.shape[0]+1e-3)
    assert sha(weights) == weight_sha and protection() == before
    source = {"source_url":url,"sha256":weight_sha,"model":"yolo11s","training_dataset":"COCO",
              "original_classes":80,"idd_finetuning_completed":False,"real_deployment_approved":False,
              "control_allowed":False,"official_checkpoint_file_unchanged":True}
    save(weights.with_name("yolo11s.provenance.json"),source)
    result = {"passed":True,"scope":"forward_only_runtime_readiness_not_model_accuracy_or_training_memory",
        "training_started":False,"backward_or_optimizer_steps":0,"real_images_evaluated":0,
        "torch":torch.__version__,"torchvision":torchvision.__version__,"ultralytics":ultralytics.__version__,
        "python":sys.version,"executable":sys.executable,"cuda_runtime":torch.version.cuda,
        "isolated_settings_file":str(settings.file),
        "gpu":torch.cuda.get_device_name(0),"total_gpu_bytes":torch.cuda.get_device_properties(0).total_memory,
        "batch":4,"imgsz":640,"warmup_batches":3,"measured_batches":20,
        "batch_forward_median_ms":statistics.median(timings),"batch_forward_max_ms":max(timings),
        "peak_allocated_bytes":peak,"peak_reserved_bytes":reserved,"native_size_postprocessing_checked":True,
        "pretrained_checkpoint":source,"checkpoint_sha256":weight_sha,"protected_sha256":before,
        "protected_files_unchanged":True,"distance_estimation_implemented":False,"model_control":False}
    save(report_path,result)
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"],check=True,capture_output=True,text=True)
    (output / "detector_environment.lock.txt").write_text(freeze.stdout,encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/detection_readiness_20261023_01")
    main(parser.parse_args().output)
