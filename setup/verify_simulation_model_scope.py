"""Check that experimental simulator selection cannot replace the real default."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_inference import file_sha256, load_final_lock
from idd_simulation_inference import NORMALIZATION, load_simulation_model_spec


def main():
    output = ROOT / "runs/model_retention_20261021_01"
    protected = json.loads((output / "protection.json").read_text(encoding="utf-8"))["protected_files"]
    before = {name: file_sha256(ROOT / name) for name in protected}
    assert before == protected
    lock, checkpoint = load_final_lock()
    assert lock["checkpoint"] == "runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr/best.keras"
    assert file_sha256(checkpoint) == "93a8faf3d43b77f0ce74d5cdbaf94124728d371c3b68b2a3386f9f72fc106725"
    candidate = ROOT / "runs/road_boundary_adaptation_20261020_03/candidate.keras"
    selection = ROOT / "runs/road_boundary_adaptation_20261020_03/selection.json"
    spec = {"schema": "simulator_road_model_v1", "scope": "recorded_carla_only", "training_allowed": False,
            "input_shape": [512, 512, 3], "normalization": NORMALIZATION, "threshold": .5,
            "shared_real_deployment_approved": False, "base_checkpoint_sha256": lock["checkpoint_sha256"],
            "checkpoint": str(candidate.relative_to(ROOT)), "checkpoint_sha256": file_sha256(candidate),
            "selection_artifact": str(selection.relative_to(ROOT)), "selection_artifact_sha256": file_sha256(selection)}
    rejected = []
    with tempfile.TemporaryDirectory(dir=output, prefix="scope-check-") as temporary:
        manifest = Path(temporary) / "model.json"

        def publish(value):
            manifest.write_text(json.dumps(value), encoding="utf-8")

        publish(spec)
        loaded, files = load_simulation_model_spec(manifest)
        assert loaded == spec and files["checkpoint"] == candidate
        mutations = {"real_scope": {"scope": "real_camera"}, "shared_approval": {"shared_real_deployment_approved": True},
                     "threshold": {"threshold": .4}, "hash": {"checkpoint_sha256": "0" * 64},
                     "path_escape": {"checkpoint": "../outside.keras"},
                     "normalization": {"normalization": "divide twice"}, "training": {"training_allowed": True}}
        for name, mutation in mutations.items():
            publish({**spec, **mutation})
            try:
                load_simulation_model_spec(manifest)
            except (ValueError, FileNotFoundError):
                rejected.append(name)
            else:
                raise AssertionError("Accepted changed simulator protocol: " + name)
    # Real inference uses only the original loader. The simulator path is opt-in.
    for name in ("predict_road.py", "predict_road_video.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "LockedRoadSegmenter(" in source and "SimulationRoadSegmenter" not in source
    after = {name: file_sha256(ROOT / name) for name in protected}
    assert after == before
    result = {"passed": True, "rejected_protocol_changes": rejected,
              "real_inference_source_and_checkpoint_unchanged": True,
              "protected_files": after, "reserved_test_images_read": False,
              "tensorflow_loaded": "tensorflow" in sys.modules}
    assert not result["tensorflow_loaded"]
    (output / "scope_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "rejected": rejected, "real_default_unchanged": True}))


if __name__ == "__main__":
    main()
