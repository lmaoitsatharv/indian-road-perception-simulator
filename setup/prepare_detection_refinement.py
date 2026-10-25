"""Create a verified higher-resolution fine-tuning recipe from the retained detector."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json
from setup.verify_detection_runtime import protection


def main(analysis):
    analysis = analysis.resolve()
    analysed = json.loads((analysis / "analysis.json").read_text())
    assert analysed["passed"] and analysed["resolutions"] == [640, 960]
    recipe_path = ROOT / "detection_refinement_recipe.json"
    if recipe_path.exists():
        raise FileExistsError("Preserve the saved refinement recipe")
    output = analysis / "refinement_runtime"
    output.mkdir(exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(output / "settings"), MPLCONFIGDIR=str(output / "matplotlib"),
                      MPLBACKEND="Agg", YOLO_AUTOINSTALL="False", WANDB_DISABLED="true")
    for folder in ("settings", "matplotlib"):
        (output / folder).mkdir()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    settings.update({key: False for key in ("sync", "wandb", "comet", "clearml", "mlflow", "neptune",
        "raytune", "tensorboard", "dvc", "hub") if key in settings})
    before = protection()
    source = ROOT / "runs/detection_train_20261024_02/weights/best.pt"
    verification = ROOT / "runs/detection_train_20261024_02/report_verification_02/checkpoint_verification.json"
    verified = json.loads(verification.read_text())
    assert verified["passed"] and file_sha(source) == verified["checkpoints"]["best"]["sha256"]
    assert file_sha(source) == analysed["checkpoint_sha256"]
    assert torch.__version__ == "2.9.1+cu128" and ultralytics.__version__ == "8.3.225" and torch.cuda.is_available()
    torch.set_num_threads(4)
    model = YOLO(str(source), task="detect")
    names = [model.names[i] for i in range(len(model.names))]
    assert names == verified["checkpoints"]["best"]["names"]
    model.model.to("cuda:0").float().eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(3):
            prediction = model.model(torch.zeros((4, 3, 960, 960), device="cuda:0"))
            raw = prediction[0] if isinstance(prediction, tuple) else prediction
            assert tuple(raw.shape) == (4, 16, 18900) and bool(torch.isfinite(raw).all())
    probes = [np.zeros((1080, 1920, 3), dtype=np.uint8), np.zeros((600, 1280, 3), dtype=np.uint8)]
    for original, prediction in zip(probes, model.predict(probes, imgsz=960, device=0, verbose=False, save=False)):
        assert prediction.orig_shape == original.shape[:2]
        assert bool(torch.isfinite(prediction.boxes.data).all())
    assert all(parameter.grad is None for parameter in model.model.parameters())
    assert protection() == before and file_sha(source) == analysed["checkpoint_sha256"]
    provenance = {"model": "yolo11s", "training_dataset": "IDD117K project filtered split",
        "idd_finetuning_completed": True, "control_allowed": False,
        "sha256": file_sha(source), "source_run": "runs/detection_train_20261024_02",
        "verification_file": verification.relative_to(ROOT).as_posix(), "verification_sha256": file_sha(verification)}
    save_json(output / "initial_checkpoint.json", provenance)
    runtime = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False, "protected_files_unchanged": True, "batch": 4, "imgsz": 960,
        "isolated_settings_file": str(settings.file), "checkpoint_sha256": file_sha(source),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__, "names": names,
        "gpu": torch.cuda.get_device_name(0), "gpu_peak_forward_allocated_bytes": torch.cuda.max_memory_allocated(),
        "forward_shape": [4, 16, 18900], "native_size_postprocessing_checked": True,
        "optimizer_steps": 0, "scope": "forward-only; verify actual first backward batches after launch"}
    save_json(output / "runtime_verification.json", runtime)
    baseline = json.loads((ROOT / "detection_training_recipe.json").read_text())
    recipe = {**baseline, "initialization": "verified_idd_detector",
        "initial_weights": source.relative_to(ROOT).as_posix(),
        "initial_weights_metadata": (output / "initial_checkpoint.json").relative_to(ROOT).as_posix(),
        "runtime_verification": (output / "runtime_verification.json").relative_to(ROOT).as_posix(),
        "imgsz": 960, "batch": 4, "epochs": 30, "patience": 8,
        "lr0": 0.0003, "lrf": 0.1, "cos_lr": True, "warmup_epochs": 1.0, "warmup_bias_lr": 0.0,
        "mosaic": 0.25, "scale": 0.3, "save_period": 5, "close_mosaic": 5,
        "sampling": {"type": "rare_images_with_pixel_group_correction", "class_image_boosts": {"4": 2, "8": 4, "9": 2},
            "validation_weighted": False, "seed": 42, "duplicate_pixel_base_probability": "divide each identity by group size"},
        "baseline_analysis": (analysis / "analysis.json").relative_to(ROOT).as_posix(),
        "baseline_analysis_sha256": file_sha(analysis / "analysis.json"),
        "matched_comparison_resolution": 960,
        "notes": ["New fine-tuning run from the verified epoch-20 detector, not exact optimizer resume.",
            "Higher resolution targets the measured small-object misses; full validation is unchanged.",
            "Gentler mosaic/scale augmentation and zero bias warmup avoid shrinking tiny targets as aggressively or jumping trained biases.",
            "Bounded rare-image sampling boosts bicycle, animal and traffic-light exposure; repeated-pixel identities share base weight.",
            "All 67,159 training identities stay eligible, with 67,159 draws per epoch. Labels and provider split membership are unchanged.",
            "Up to 30 additional epochs with patience 8; requested score targets remain unproven.",
            "Periodic full checkpoints preserve recovery state. Road checkpoints, original detector and original recipe remain separate."]}
    save_json(recipe_path, recipe)
    print(json.dumps({"passed": True, "recipe": str(recipe_path), "runtime": runtime}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    main(parser.parse_args().analysis)
