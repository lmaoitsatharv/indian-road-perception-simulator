"""Explicit simulator-only inference; real-image and real-video tools do not import this."""
import json
from pathlib import Path

from idd_inference import ROOT, SHAPE, LockedRoadSegmenter, file_sha256, load_final_lock

NORMALIZATION = "RGB /127.5 -1 exactly once"


def load_simulation_model_spec(path):
    path = Path(path).resolve(strict=True)
    runs = (ROOT / "runs").resolve()
    if not path.is_relative_to(runs) or path.stat().st_size > 100_000:
        raise ValueError("Choose a small simulator-model manifest inside runs/.")
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema": "simulator_road_model_v1", "scope": "recorded_carla_only",
                "training_allowed": False, "input_shape": list(SHAPE),
                "normalization": NORMALIZATION, "threshold": .5,
                "shared_real_deployment_approved": False}
    for key, value in expected.items():
        if spec.get(key) != value or type(spec.get(key)) is not type(value):
            raise ValueError(f"Simulator-only model protocol differs: {key}")
    baseline, _ = load_final_lock()
    if spec.get("base_checkpoint_sha256") != baseline["checkpoint_sha256"]:
        raise ValueError("Simulator model was not compared with the locked real-road baseline")
    resolved = {}
    for key in ("checkpoint", "selection_artifact"):
        relative = Path(spec[key])
        source = (ROOT / relative).resolve(strict=True)
        if (relative.is_absolute() or not source.is_relative_to(runs)
                or file_sha256(source) != spec[key + "_sha256"]):
            raise ValueError(f"Simulator model path/hash differs: {key}")
        resolved[key] = source
    if resolved["checkpoint"].suffix != ".keras":
        raise ValueError("Expected a saved Keras simulator checkpoint")
    return spec, resolved


class SimulationRoadSegmenter(LockedRoadSegmenter):
    """Opt-in model for verified recorded CARLA frames, outside the real-image path."""

    def __init__(self, manifest, *, device="gpu", gpu_memory_mib=3072):
        spec, files = load_simulation_model_spec(manifest)
        # Reuse the unchanged runtime setup, deterministic settings and input contract.
        # Its inference function is first traced after the selected model is loaded.
        super().__init__(device=device, gpu_memory_mib=gpu_memory_mib)
        baseline_lock = self.lock
        with self.tf.device(self.device):
            self.model = self.tf.keras.models.load_model(files["checkpoint"], compile=False, safe_mode=True)
        if (tuple(self.model.input_shape[1:]) != SHAPE
                or tuple(self.model.output_shape[1:]) != (512, 512, 1)
                or file_sha256(files["checkpoint"]) != spec["checkpoint_sha256"]):
            raise ValueError("Simulator checkpoint changed or has incompatible input/output")
        self.model.trainable = False
        self.checkpoint = files["checkpoint"]
        self.lock = {**baseline_lock, "model": "separate_simulator_candidate",
                     "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"],
                     "scope": spec["scope"]}
        self.runtime.update(checkpoint_sha256=spec["checkpoint_sha256"],
                            base_checkpoint_sha256=baseline_lock["checkpoint_sha256"],
                            model_manifest=str(Path(manifest).resolve()),
                            model_manifest_sha256=file_sha256(manifest),
                            scope=spec["scope"], shared_real_deployment_approved=False,
                            selection_artifact=spec["selection_artifact"],
                            real_image_and_video_default="Unchanged LockedRoadSegmenter")
