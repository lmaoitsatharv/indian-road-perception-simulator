"""Synthetic deployment checks; never train or read reserved test image contents."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import idd_inference as inference
import predict_road


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inference_contract_", dir=ROOT / "runs/review_20261013")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.rgb = np.broadcast_to(np.array([0, 127, 255], np.uint8), inference.SHAPE).copy()
        self.probability = np.full((512, 512), .5, np.float32)
        self.prediction = inference.RoadPrediction(self.probability, np.ones((512, 512), np.uint8), 1.0, "/device:CPU:0")
        self.lock = json.loads((ROOT / "models/segmentation_final_lock.json").read_text(encoding="utf-8"))

    def export(self, *, prediction=None, stem="frame", timestamp=None, metadata=None, lock=None):
        return inference.write_prediction(self.folder / "output", stem, self.rgb,
                                          prediction or self.prediction, metadata or {},
                                          timestamp_s=timestamp, model_lock=lock)

    def test_normalize_once_and_preserve_input(self):
        original = self.rgb.copy()
        normalized = inference.normalize_raw_rgb(self.rgb)
        np.testing.assert_array_equal(self.rgb, original)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(float(normalized.min()), -1)
        self.assertEqual(float(normalized.max()), 1)
        self.assertIs(inference.require_normalized_rgb(normalized), normalized)
        with self.assertRaises(ValueError):
            inference.normalize_raw_rgb(normalized)

    def test_invalid_raw_and_normalized_inputs(self):
        for value in (np.zeros(inference.SHAPE, np.float32), np.zeros((256, 512, 3), np.uint8)):
            with self.subTest(dtype=str(value.dtype), shape=value.shape), self.assertRaises(ValueError):
                inference.require_raw_rgb(value)
        for value in (np.zeros(inference.SHAPE, np.float64), np.ones((512, 512), np.float32),
                      np.full(inference.SHAPE, np.nan, np.float32), np.full(inference.SHAPE, np.inf, np.float32),
                      np.full(inference.SHAPE, 1.01, np.float32), np.full(inference.SHAPE, -1.01, np.float32)):
            with self.subTest(dtype=str(value.dtype), shape=value.shape), self.assertRaises(ValueError):
                inference.require_normalized_rgb(value)

    def test_decode_hash_matches_decoded_bytes(self):
        path = self.folder / "rgb.png"
        Image.fromarray(self.rgb).save(path)
        decoded, metadata = inference.read_rgb(path)
        np.testing.assert_array_equal(decoded, self.rgb)
        self.assertEqual(metadata["input_sha256"], inference.file_sha256(path))
        self.assertFalse(metadata["resize_applied"])
        self.assertFalse(metadata["exif_orientation_applied"])

    def test_explicit_resize_and_exif_transform(self):
        path = self.folder / "oriented.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (1024, 256), (10, 20, 30)).save(path, exif=exif)
        with self.assertRaises(ValueError):
            inference.read_rgb(path)
        rgb, metadata = inference.read_rgb(path, resize=True)
        self.assertEqual(rgb.shape, inference.SHAPE)
        self.assertEqual(metadata["oriented_source_size_wh"], [256, 1024])
        self.assertTrue(metadata["exif_orientation_applied"])
        self.assertEqual(metadata["source_to_model_pixel_centers"], [[2., 0, .5], [0, .5, -.25], [0, 0, 1]])

    def test_export_exact_roundtrip_and_threshold_tie(self):
        folder = self.export(timestamp=np.float64(0), lock=self.lock)
        np.testing.assert_array_equal(np.array(Image.open(folder / "road_mask_01.png")), self.prediction.mask)
        np.testing.assert_array_equal(np.load(folder / "road_probability.npy", allow_pickle=False), self.probability)
        self.assertEqual(set(np.array(Image.open(folder / "mask_display.png")).ravel()), {255})
        metadata = json.loads((folder / "frame.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["timestamp_s"], 0.)
        self.assertEqual(metadata["threshold_comparator"], ">=")
        self.assertEqual(metadata["checkpoint_sha256"], self.lock["checkpoint_sha256"])
        self.assertTrue(metadata["exported_at_utc"].endswith("+00:00"))
        self.assertFalse((folder / "frame.json").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertFalse(metadata["planning_use_valid"])

    def test_export_does_not_clobber_existing_frame(self):
        folder = self.export()
        before = inference.file_sha256(folder / "frame.json")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(inference.file_sha256(folder / "frame.json"), before)

    def test_export_rejects_escape_and_windows_invalid_names(self):
        for name in ("../escape", "..\\escape", "/absolute", "E:\\escape", ".", "..", "", "a/b", "a\\b", "NUL", "a.", "a ", "a:b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.export(stem=name)
        self.assertFalse((self.folder / "output").exists())

    def test_export_rejects_bad_values_before_writing(self):
        for probability, mask, duration in (
                (np.full((512, 512), np.nan, np.float32), self.prediction.mask, 1.),
                (np.full((512, 512), 1.1, np.float32), self.prediction.mask, 1.),
                (self.probability.astype(np.float64), self.prediction.mask, 1.),
                (self.probability, self.prediction.mask * 255, 1.),
                (self.probability, np.zeros((512, 512), np.uint8), 1.),
                (self.probability, self.prediction.mask, float("inf")),
                (self.probability, self.prediction.mask, -1.)):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.export(prediction=inference.RoadPrediction(probability, mask, duration, "CPU:0"))
        self.assertFalse((self.folder / "output").exists())

    def test_export_rejects_bad_timestamps_and_metadata(self):
        for timestamp in (-1, np.nan, np.inf, True, "1"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.export(timestamp=timestamp)
        with self.assertRaises(ValueError):
            self.export(metadata={"nonfinite": float("nan")})
        self.assertFalse((self.folder / "output").exists())

    def test_atomic_json_preserves_previous_on_invalid_data(self):
        path = self.folder / "status.json"
        inference.write_json_atomic(path, {"status": "valid"})
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            inference.write_json_atomic(path, {"invalid": float("nan")})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.folder.glob("*.tmp")), [])

    def fixture_lock(self):
        (self.folder / "models").mkdir()
        (self.folder / "dataset/splits").mkdir(parents=True)
        retained = json.loads((ROOT / "models/retained_models.json").read_text(encoding="utf-8"))
        checkpoint = self.folder / self.lock["checkpoint"]
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint: never loaded")
        self.lock["checkpoint_sha256"] = inference.file_sha256(checkpoint)
        selected = next(m for m in retained["models"] if m["name"] == self.lock["model"])
        selected["checkpoint_sha256"] = self.lock["checkpoint_sha256"]
        (self.folder / "dataset/splits/pair_splits.csv").write_bytes((ROOT / "dataset/splits/pair_splits.csv").read_bytes())
        self.retained = retained
        self.save_lock()
        return checkpoint

    def save_lock(self):
        inference.write_json_atomic(self.folder / "models/segmentation_final_lock.json", self.lock)
        inference.write_json_atomic(self.folder / "models/retained_models.json", self.retained)

    def test_lock_rejects_semantic_drift(self):
        self.fixture_lock()
        original = copy.deepcopy(self.lock)
        with patch.object(inference, "ROOT", self.folder):
            lock, checkpoint = inference.load_final_lock()
            self.assertEqual(lock["model"], "morning_best")
            self.assertTrue(checkpoint.is_relative_to(self.folder))
            for key, value in (("threshold_comparator", ">"), ("mask_labels", [0, 255]),
                               ("augmentation", True), ("inference_training_flag", True),
                               ("input_preprocessing", "raw /255"), ("seed", 99),
                               ("deterministic_operations", False), ("threshold", True)):
                self.lock = copy.deepcopy(original)
                self.lock[key] = value
                self.save_lock()
                with self.subTest(key=key), self.assertRaises(ValueError):
                    inference.load_final_lock()

    def test_lock_rejects_checkpoint_manifest_and_selection_mismatch(self):
        checkpoint = self.fixture_lock()
        with patch.object(inference, "ROOT", self.folder):
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
                inference.load_final_lock()
            checkpoint.write_bytes(b"synthetic checkpoint: never loaded")
            manifest = self.folder / "dataset/splits/pair_splits.csv"
            original = manifest.read_bytes()
            manifest.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                inference.load_final_lock()
            manifest.write_bytes(original)
            self.retained["recommended_current_model"] = "boundary_best"
            self.save_lock()
            with self.assertRaisesRegex(ValueError, "selection disagree"):
                inference.load_final_lock()

    def test_lock_rejects_escape_and_duplicate_manifest_ids(self):
        self.fixture_lock()
        with patch.object(inference, "ROOT", self.folder):
            original = self.lock["checkpoint"]
            self.lock["checkpoint"] = "../outside.keras"
            next(m for m in self.retained["models"] if m["name"] == self.lock["model"])["checkpoint"] = self.lock["checkpoint"]
            self.save_lock()
            with self.assertRaisesRegex(ValueError, "relative path"):
                inference.load_final_lock()
            self.lock["checkpoint"] = original
            next(m for m in self.retained["models"] if m["name"] == self.lock["model"])["checkpoint"] = original
            manifest = self.folder / "dataset/splits/pair_splits.csv"
            contents = manifest.read_text(encoding="utf-8").splitlines()
            manifest.write_text("\n".join(contents + [contents[1]]) + "\n", encoding="utf-8")
            self.lock["manifest_sha256"] = self.retained["manifest_sha256"] = inference.file_sha256(manifest)
            self.save_lock()
            with self.assertRaisesRegex(ValueError, "repeated ID"):
                inference.load_final_lock()

    def mocked_segmenter(self, values=None, output_device="/job:localhost/device:GPU:0"):
        segmenter = object.__new__(inference.LockedRoadSegmenter)
        segmenter.tf = types.SimpleNamespace(convert_to_tensor=lambda value, dtype: value, float32=np.float32)
        segmenter.device, segmenter.threshold = "/GPU:0", .5
        segmenter.inputs = []
        def infer(value):
            segmenter.inputs.append(value.copy())
            return types.SimpleNamespace(numpy=lambda: values if values is not None else self.probability[None, ..., None], device=output_device)
        segmenter._infer = infer
        return segmenter

    def test_inference_entry_points_use_exactly_one_normalization(self):
        segmenter = self.mocked_segmenter()
        raw_prediction = segmenter.predict_rgb(self.rgb)
        normalized = inference.normalize_raw_rgb(self.rgb)
        normalized_prediction = segmenter.predict_normalized(normalized)
        np.testing.assert_array_equal(segmenter.inputs[0], normalized[None])
        np.testing.assert_array_equal(segmenter.inputs[0], segmenter.inputs[1])
        np.testing.assert_array_equal(raw_prediction.mask, normalized_prediction.mask)

    def test_inference_rejects_bad_output_shape_nan_and_device(self):
        for values, device in ((np.zeros((1, 256, 512, 1), np.float32), "GPU:0"),
                               (np.full((1, 512, 512, 1), np.nan, np.float32), "GPU:0"),
                               (None, "/job:localhost/device:CPU:0")):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                self.mocked_segmenter(values, device).predict_rgb(self.rgb)

    def test_cli_duplicate_and_timestamp_preflight_does_not_load_model(self):
        for extra in (("--image", "a.png", "--image", "a.png"),
                      ("--image", "a.png", "--image", "b.png", "--timestamp-s", "1")):
            with patch.object(sys, "argv", ["predict_road.py", "--output-dir", str(self.folder / "cli"), *extra]), \
                    patch.object(predict_road, "LockedRoadSegmenter") as constructor, self.assertRaises(SystemExit):
                predict_road.main()
            constructor.assert_not_called()
            self.assertFalse((self.folder / "cli").exists())

    def test_cli_input_changed_since_preflight_fails_truthfully(self):
        segmenter = self.mocked_cli_segmenter()
        output = self.folder / "changed_input"
        with patch.object(sys, "argv", ["predict_road.py", "--image", "synthetic.png", "--output-dir", str(output)]), \
                patch.object(predict_road, "LockedRoadSegmenter", return_value=segmenter), \
                patch.object(predict_road, "read_rgb", side_effect=[(self.rgb, {"input_sha256": "a"}), (self.rgb, {"input_sha256": "b", "input_file": "synthetic.png"})]):
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                predict_road.main()
        status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["checkpoint_unchanged"])
        self.assertEqual(status["images"], [])

    def mocked_cli_segmenter(self):
        checkpoint = self.folder / "checkpoint"
        checkpoint.write_bytes(b"unchanged")
        lock = {**self.lock, "checkpoint_sha256": inference.file_sha256(checkpoint)}
        return types.SimpleNamespace(checkpoint=checkpoint, lock=lock, runtime={}, warmup=lambda: None,
                                     predict_rgb=lambda rgb: self.prediction)

    def test_cli_keyboard_interrupt_and_checkpoint_change_are_recorded(self):
        for changed in (False, True):
            segmenter = self.mocked_cli_segmenter()
            output = self.folder / f"failed_{changed}"
            def interrupt():
                if changed:
                    segmenter.checkpoint.write_bytes(b"altered")
                    raise RuntimeError("inference failure")
                raise KeyboardInterrupt("user interruption")
            segmenter.warmup = interrupt
            with patch.object(sys, "argv", ["predict_road.py", "--image", "synthetic.png", "--output-dir", str(output)]), \
                    patch.object(predict_road, "LockedRoadSegmenter", return_value=segmenter), \
                    patch.object(predict_road, "read_rgb", return_value=(self.rgb, {"input_sha256": "a"})):
                with self.assertRaises(RuntimeError if changed else KeyboardInterrupt):
                    predict_road.main()
            status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed" if changed else "interrupted")
            self.assertEqual(status["checkpoint_unchanged"], not changed)
            self.assertIn("finished_at_utc", status)


class Maintenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "setup"))
        cls.modules = {}
        for name, relative in (("overnight", "setup/summarize_overnight_results.py"),
                               ("boundary", "setup/build_boundary_report.py"),
                               ("analysis", "setup/summarize_training_analysis.py"),
                               ("validation", "review_idd_validation.py"),
                               ("preprocessing", "check_preprocessing.py")):
            spec = importlib.util.spec_from_file_location("review_maintenance_" + name, ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[name] = module

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maintenance_contract_", dir=ROOT / "runs/review_20261013")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_historical_generators_refuse_final_lock_before_reads_or_writes(self):
        (self.folder / "models").mkdir()
        (self.folder / "models/segmentation_final_lock.json").write_text("{}", encoding="utf-8")
        for name in ("overnight", "boundary"):
            with self.subTest(generator=name), patch.object(self.modules[name], "ROOT", self.folder), \
                    patch.object(Path, "open", side_effect=AssertionError("Historical file access after final lock")), \
                    self.assertRaisesRegex(RuntimeError, "locked"):
                self.modules[name].main()

    def test_training_analysis_import_performs_no_run_io(self):
        source_path = ROOT / "setup/summarize_training_analysis.py"
        source = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
        namespace = {"__name__": "read_only_import_probe", "__file__": str(source_path)}
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected run read at import")), \
                patch.object(Path, "read_text", side_effect=AssertionError("Unexpected run read at import")), \
                patch.object(Path, "write_text", side_effect=AssertionError("Unexpected write at import")):
            exec(source, namespace)
        self.assertTrue(callable(namespace["main"]))

    def test_training_analysis_default_preserves_saved_evidence(self):
        module = self.modules["analysis"]
        with patch.object(module, "summarize", side_effect=AssertionError("Unexpected regeneration")), \
                patch.object(Path, "open", side_effect=AssertionError("Unexpected source/output access")):
            module.main([])
        with self.assertRaises(FileExistsError):
            module.summarize(self.folder)

    def test_archived_console_log_is_discovered_without_regeneration(self):
        module = self.modules["analysis"]
        archived = self.folder / "archive/completed_runtime_20261003/setup/frozen_01_console.log"
        archived.parent.mkdir(parents=True)
        archived.write_text("Epoch 1 completed in 1.0 seconds", encoding="utf-8")
        self.assertEqual(module.find_console_log(self.folder), archived)
        current = self.folder / "setup/frozen_01_console.log"
        current.parent.mkdir()
        current.write_text("Epoch 1 completed in 2.0 seconds", encoding="utf-8")
        self.assertEqual(module.find_console_log(self.folder), current)

    def test_preprocessing_defaults_and_existing_output_guard(self):
        module = self.modules["preprocessing"]
        self.assertEqual(module.PREVIEW_SPLITS, ("train", "val"))
        with patch.object(module, "read_pairs", side_effect=AssertionError("Unexpected dataset access")), \
                self.assertRaisesRegex(FileExistsError, "preserved"):
            module.main(self.folder)

    def test_validation_existing_output_refused_before_gpu_or_source_access(self):
        module = self.modules["validation"]
        run = self.folder / "run"
        (run / "validation_review").mkdir(parents=True)
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected run read")), \
                self.assertRaisesRegex(FileExistsError, "preserved"):
            module.review(run)

    def test_null_or_missing_initial_does_not_claim_equivalence(self):
        module = self.modules["validation"]
        for recorded in (None, "/mnt/e/old/Tesla/models/removed_initial.keras"):
            with self.subTest(recorded=recorded), patch.object(module.tf.keras.models, "load_model", side_effect=AssertionError("Missing initial model must not load")):
                result = module.verify_initial_comparison({"initial_checkpoint": recorded, "phase": "frozen"}, object(), self.folder)
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["encoder_weights_unchanged"])
            self.assertIsNone(result["decoder_weights_changed"])
            self.assertIsNone(result["initial_checkpoint_sha256"])

    def test_recorded_windows_wsl_and_relative_checkpoint_paths_resolve(self):
        module = self.modules["validation"]
        checkpoint = self.folder / "runs/reference/best.keras"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic lineage fixture")
        for recorded in ("/mnt/e/old/Tesla/runs/reference/best.keras", "E:/old/Tesla/runs/reference/best.keras", "runs/reference/best.keras"):
            with self.subTest(recorded=recorded):
                self.assertEqual(module.resolve_recorded_checkpoint(recorded, self.folder), checkpoint)
        with patch.object(module.tf.keras.models, "load_model", side_effect=AssertionError("Hash mismatch must precede loading")), \
                self.assertRaisesRegex(ValueError, "recorded hash"):
            module.verify_initial_comparison({"initial_checkpoint": "runs/reference/best.keras", "initial_checkpoint_sha256": "incorrect", "phase": "frozen"}, object(), self.folder)


def preserved_artifacts():
    baseline = json.loads((ROOT / "setup/review_20261013/model_pipeline_before.json").read_text(encoding="utf-8"))
    after = {path: inference.file_sha256(ROOT / path) for path in baseline["sha256"]}
    if after != baseline["sha256"]:
        raise RuntimeError("A protected checkpoint, lock, manifest or test evidence file changed.")
    return {"sha256_before": baseline["sha256"], "sha256_after": after,
            "all_protected_artifacts_unchanged": True, "manifest_counts": baseline["manifest_counts"]}


def tensorflow_synthetic_checks(cpu_model_smoke=False):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    from idd_data import decode_pair, normalize_pair, read_pairs
    rgb = np.broadcast_to(np.array([0, 127, 255], np.uint8), inference.SHAPE).copy()
    with tempfile.TemporaryDirectory(prefix="tf_contract_", dir=ROOT / "runs/review_20261013") as temporary:
        folder = Path(temporary)
        Image.fromarray(rgb).save(folder / "image.png")
        mask = np.zeros((512, 512), np.uint8)
        mask[256:] = 1
        Image.fromarray(mask).save(folder / "mask.png")
        image, labels = decode_pair(str(folder / "image.png"), str(folder / "mask.png"))
        normalized, normalized_labels = normalize_pair(image, labels)
        np.testing.assert_array_equal(normalized.numpy(), inference.normalize_raw_rgb(rgb))
        np.testing.assert_array_equal(normalized_labels.numpy()[..., 0], mask)
        Image.fromarray(mask * 255).save(folder / "invalid_mask.png")
        try:
            decode_pair(str(folder / "image.png"), str(folder / "invalid_mask.png"))
        except tf.errors.InvalidArgumentError:
            pass
        else:
            raise RuntimeError("Nonbinary mask was accepted.")
    # read_pairs inspects the CSV and filenames only; no real PNG is decoded here.
    pairs = read_pairs(ROOT / "dataset/splits")
    counts = {split: len(records) for split, records in pairs.items()}
    if counts != inference.SPLIT_COUNTS:
        raise RuntimeError("Live split file structure changed.")
    result = {"tensorflow": tf.__version__, "normalization_matches_original_exactly": True,
              "binary_masks_preserved": True, "nonbinary_masks_rejected": True,
              "manifest_and_file_structure_counts": counts, "gpu_used": False}
    if cpu_model_smoke:
        # Use a fresh interpreter for model smoke, because set_visible_devices in
        # this preprocessing check deliberately creates no reusable GPU context.
        segmenter = inference.LockedRoadSegmenter(device="cpu")
        fingerprint = lambda: hashlib.sha256(b"".join(v.tobytes() for v in segmenter.model.get_weights())).hexdigest()
        before = fingerprint()
        prediction = segmenter.predict_rgb(rgb)
        if before != fingerprint():
            raise RuntimeError("Inference updated model or BatchNorm weights.")
        result["actual_locked_cpu_inference"] = {"status": "passed", "output_shape": list(prediction.mask.shape),
                                                "output_device": prediction.output_device,
                                                "mask_values": list(map(int, np.unique(prediction.mask))),
                                                "model_and_bn_unchanged": True, "synthetic_input_only": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / "runs/review_20261013/model_pipeline_verification.json")
    parser.add_argument("--cpu-model-smoke", action="store_true")
    parser.add_argument("--maintenance-only", action="store_true", help="Pure/mock historical-output guards; no inference or real dataset iteration.")
    args = parser.parse_args()
    (ROOT / "runs/review_20261013").mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "training_started": False, "test_inference_rerun": False, "test_image_contents_read": False,
              "gpu_or_wsl_used": False, "synthetic_tests_do_not_establish_accuracy_or_latency": True}
    baseline_reproductions = ROOT / "runs/review_20261013/model_pipeline_baseline_reproductions.json"
    if baseline_reproductions.exists():
        report["reproduced_baseline_failures"] = json.loads(baseline_reproductions.read_text(encoding="utf-8"))
    report["preservation_before_tests"] = preserved_artifacts()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Maintenance if args.maintenance_only else Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report["contracts"] = {"tests_run": result.testsRun, "passed": result.wasSuccessful(),
                            "failures": [str(test) for test, _ in result.failures],
                            "errors": [str(test) for test, _ in result.errors]}
    try:
        if args.maintenance_only:
            report["verification_scope"] = "Pure/mock maintenance guards and import/lineage behavior; no inference or dataset iteration."
        else:
            report["tensorflow_synthetic"] = tensorflow_synthetic_checks(args.cpu_model_smoke)
        report["preservation_after_tests"] = preserved_artifacts()
        report["status"] = "passed" if result.wasSuccessful() else "failed"
    except Exception as error:
        report.update(status="failed", error=str(error), error_type=type(error).__name__)
        raise
    finally:
        inference.write_json_atomic(args.output_json, report)
    print(json.dumps({"status": report["status"], "tests_run": result.testsRun,
                      "output_json": str(args.output_json)}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
