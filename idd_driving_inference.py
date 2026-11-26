"""Explicit camera-driving scope for the simulation road model (native CARLA only).

Separate from the observation-only live loader (idd_live_simulation_inference.py,
control_allowed false) and from recorded/real inference. This scope lets the
simulation road model's mask steer the camera car in the connected CARLA town.
It never trains and is not approved for real images.
"""
import json
from pathlib import Path

from idd_inference import ROOT, SHAPE, LockedRoadSegmenter, file_sha256, load_final_lock

EXPECTED = {"schema": "simulator_camera_driving_v1", "scope": "native_carla_camera_driving",
            "training_allowed": False, "control_allowed": True, "shared_real_deployment_approved": False,
            "input_shape": list(SHAPE), "threshold": .5, "normalization": "RGB /127.5 -1 exactly once"}


def load_driving_spec(path):
    path = Path(path).resolve(strict=True)
    runs = (ROOT / "runs").resolve()
    if not path.is_relative_to(runs) or path.stat().st_size > 100_000:
        raise ValueError("Driving model manifest must be a small local runs/ artifact")
    spec = json.loads(path.read_text())
    for key, value in EXPECTED.items():
        if type(spec.get(key)) is not type(value) or spec[key] != value:
            raise ValueError("Camera-driving road-model protocol differs: " + key)
    files = {}
    for key in ("checkpoint", "confirmation"):
        relative = Path(spec[key])
        target = (ROOT / relative).resolve(strict=True)
        if relative.is_absolute() or not target.is_relative_to(runs) or file_sha256(target) != spec[key + "_sha256"]:
            raise ValueError("Camera-driving file/hash differs: " + key)
        files[key] = target
    confirmed = json.loads(files["confirmation"].read_text())
    if confirmed["status"] != "completed" or confirmed["checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Driving checkpoint lacks its completed recorded confirmation")
    baseline, _ = load_final_lock()
    if spec["base_checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ValueError("Protected real-model anchor differs")
    return spec, files


class DrivingRoadSegmenter(LockedRoadSegmenter):
    def __init__(self, manifest, gpu_memory_mib=1024):
        spec, files = load_driving_spec(manifest)
        super().__init__(device="gpu", gpu_memory_mib=gpu_memory_mib)
        with self.tf.device(self.device):
            self.model = self.tf.keras.models.load_model(files["checkpoint"], compile=False, safe_mode=True)
        if tuple(self.model.input_shape[1:]) != SHAPE or tuple(self.model.output_shape[1:]) != (512, 512, 1):
            raise ValueError("Driving model input/output shape differs")
        self.model.trainable = False
        self.checkpoint = files["checkpoint"]
        self.lock = {**self.lock, "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"], "scope": spec["scope"]}
        self.runtime.update(checkpoint_sha256=spec["checkpoint_sha256"], scope=spec["scope"], training_allowed=False,
                            control_allowed=True, model_manifest_sha256=file_sha256(manifest))
