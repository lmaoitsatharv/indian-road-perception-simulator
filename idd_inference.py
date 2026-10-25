"""Reusable inference for the final locked road model. No training or model writes.

Raw uint8 RGB and already normalized float32 input have separate entry points.
Outputs stay in the 512x512 camera image frame; they are not metric planning maps.
"""
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import numbers
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SHAPE = (512, 512, 3)
SPLIT_COUNTS = {"train": 5507, "val": 687, "test": 689}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json_atomic(path, value):
    """Publish complete UTF-8 JSON; interrupted writes preserve the previous file."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_final_lock():
    """Check deployment semantics and manifest metadata before loading TensorFlow."""
    lock = json.loads((ROOT / "models/segmentation_final_lock.json").read_text(encoding="utf-8"))
    expected = {"threshold": .5, "threshold_comparator": ">=", "input_shape": list(SHAPE),
                "mask_labels": [0, 1], "inference_training_flag": False, "augmentation": False,
                "test_time_augmentation": False, "deterministic_operations": True, "seed": 42,
                "training_prohibited": True, "parameter_or_bn_updates_prohibited": True,
                "postprocessing": "threshold only; no morphology or component filtering",
                "input_preprocessing": "Existing idd_data.make_dataset(training=False): raw RGB normalized once to [-1,1].",
                "split_counts": SPLIT_COUNTS}
    for key, value in expected.items():
        if lock.get(key) != value or type(lock.get(key)) is not type(value):
            raise ValueError(f"The final inference protocol has changed: {key}.")
    retained = json.loads((ROOT / "models/retained_models.json").read_text(encoding="utf-8"))
    selected = [model for model in retained["models"] if model["name"] == lock.get("model")]
    if (len(selected) != 1 or retained.get("recommended_current_model") != lock.get("model")
            or selected[0]["checkpoint"] != lock.get("checkpoint")
            or selected[0]["checkpoint_sha256"] != lock.get("checkpoint_sha256")):
        raise ValueError("The final lock and retained model selection disagree.")
    checkpoint_relative = Path(lock["checkpoint"])
    checkpoint = (ROOT / checkpoint_relative).resolve()
    if checkpoint_relative.is_absolute() or not checkpoint.is_relative_to(ROOT.resolve()):
        raise ValueError("Locked checkpoint must be a relative path inside this project.")
    if file_sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match the final model lock.")
    manifest = ROOT / "dataset/splits/pair_splits.csv"
    if (retained.get("manifest_sha256") != lock.get("manifest_sha256")
            or file_sha256(manifest) != lock["manifest_sha256"]):
        raise ValueError("Split manifest hash does not match the final model lock.")
    counts, seen = {split: 0 for split in SPLIT_COUNTS}, set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            pair_id, split = int(row["id"]), row["split"]
            if pair_id < 0 or pair_id in seen or split not in counts:
                raise ValueError("Invalid or repeated ID in the locked split manifest.")
            seen.add(pair_id)
            counts[split] += 1
    if counts != SPLIT_COUNTS:
        raise ValueError("Split counts do not match the final model lock.")
    return lock, checkpoint


def require_raw_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != SHAPE:
        raise ValueError("Raw input must be uint8 RGB with shape (512,512,3). Use predict_normalized for existing float32 [-1,1] tensors.")
    return image


def normalize_raw_rgb(image):
    """The existing MobileNetV2 transform, once, on a fresh float32 array."""
    image = require_raw_rgb(image)
    return image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)


def require_normalized_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.float32 or image.shape != SHAPE:
        raise ValueError("Normalized input must be float32 with shape (512,512,3).")
    if not np.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("Normalized input must contain finite values in [-1,1].")
    return image


def read_rgb(path, *, resize=False):
    """Read one image, applying EXIF orientation; resizing requires an explicit opt-in."""
    path = Path(path).expanduser().resolve(strict=True)
    # Decode and fingerprint the same bytes, even if the caller replaces a file
    # while prediction runs. Only this one encoded image is held in memory.
    encoded = path.read_bytes()
    with Image.open(io.BytesIO(encoded)) as opened:
        orientation = opened.getexif().get(274)
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        if original_size != (512, 512):
            if not resize:
                raise ValueError(f"{path.name} is {original_size}, expected (512,512). Pass --resize explicitly for a new non-square image.")
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
        rgb = np.array(image, dtype=np.uint8)
    sx, sy = 512/original_size[0], 512/original_size[1]
    metadata = {"input_file": str(path), "input_sha256": hashlib.sha256(encoded).hexdigest(),
                "oriented_source_size_wh": list(original_size), "output_size_wh": [512, 512],
                "resize_applied": original_size != (512, 512),
                "exif_orientation_applied": orientation in range(2, 9) if isinstance(orientation, int) else False,
                "resize_policy": "explicit direct bilinear resize to the trained square input" if original_size != (512, 512) else "none",
                "source_to_model_pixel_centers": [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]],
                "calibration_note": "Transform is from the oriented source image. Camera intrinsics/projection are not supplied or inferred."}
    return rgb, metadata


@dataclass(frozen=True)
class RoadPrediction:
    probability: np.ndarray
    mask: np.ndarray
    inference_ms: float
    output_device: str


class LockedRoadSegmenter:
    """Load the selected checkpoint once and reuse its fixed inference graph."""

    def __init__(self, *, device="gpu", gpu_memory_mib=3072):
        if device not in ("gpu", "cpu"):
            raise ValueError("device must be 'gpu' or explicitly 'cpu'.")
        if device == "gpu" and (isinstance(gpu_memory_mib, bool)
                or not isinstance(gpu_memory_mib, numbers.Integral) or gpu_memory_mib < 1024):
            raise ValueError("GPU memory limit must be an integer of at least 1024 MiB.")
        self.lock, self.checkpoint = load_final_lock()

        import tensorflow as tf
        import idd_model  # Register the saved custom objects.
        import idd_boundary
        self.tf = tf
        devices = tf.config.list_physical_devices("GPU")
        if device == "gpu":
            if not devices:
                raise RuntimeError("No TensorFlow GPU is available. Use the verified WSL launcher, or explicitly select CPU.")
            try:
                tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_mib)])
            except RuntimeError as error:
                raise RuntimeError("GPU context was already initialized. Start a fresh Python process for this bounded inference runtime.") from error
            self.device = "/GPU:0"
            details = tf.config.experimental.get_device_details(devices[0])
        else:
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as error:
                raise RuntimeError("Start a fresh Python process to select CPU explicitly.") from error
            self.device, details = "/CPU:0", {}
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(self.checkpoint, compile=False, safe_mode=True)
        if file_sha256(self.checkpoint) != self.lock["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during model loading.")
        if tuple(self.model.input_shape[1:]) != SHAPE or tuple(self.model.output_shape[1:]) != (512, 512, 1):
            raise ValueError("The locked model input/output shapes are inconsistent.")
        self.threshold = float(self.lock["threshold"])

        @tf.function(input_signature=[tf.TensorSpec([1, *SHAPE], tf.float32)], jit_compile=False)
        def infer(normalized):
            with tf.device(self.device):
                return self.model(normalized, training=False)

        self._infer = infer
        self.runtime = {"device_requested": device, "device": self.device,
                        "gpu_details": details, "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                        "gpu_memory_limit_mib": gpu_memory_mib if device == "gpu" else None,
                        "model_loaded_once": True, "optimizer_loaded": False,
                        "normalization": "raw RGB /127.5 -1 once; normalized entry point does not rescale",
                        "threshold": self.threshold, "checkpoint_sha256": self.lock["checkpoint_sha256"]}

    def predict_rgb(self, image):
        """Predict from raw uint8 RGB; float inputs are rejected to prevent double scaling."""
        return self.predict_normalized(normalize_raw_rgb(image))

    def predict_normalized(self, image):
        """Predict from an existing normalized tensor; no second normalization occurs."""
        image = require_normalized_rgb(image)
        started = time.perf_counter()
        tensor = self._infer(self.tf.convert_to_tensor(image[None], dtype=self.tf.float32))
        values = tensor.numpy()
        if values.shape != (1, 512, 512, 1):
            raise RuntimeError("Road probability output has an unexpected shape.")
        if not tensor.device.upper().endswith(self.device.lstrip("/")):
            raise RuntimeError(f"Inference output was on {tensor.device}, expected {self.device}.")
        probability = values[0, ..., 0].astype(np.float32, copy=False)
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise RuntimeError("Invalid road probability output.")
        mask = (probability >= self.threshold).astype(np.uint8)
        elapsed = (time.perf_counter()-started)*1000
        return RoadPrediction(probability, mask, elapsed, tensor.device)

    def warmup(self):
        """Compile the inference graph with a synthetic raw image; no dataset access."""
        return self.predict_rgb(np.zeros(SHAPE, dtype=np.uint8))


def write_prediction(output_dir, stem, rgb, prediction, source_metadata, *, timestamp_s=None, model_lock=None):
    """Save binary data and display previews separately; refuse to overwrite a frame."""
    if timestamp_s is not None and (isinstance(timestamp_s, bool) or not isinstance(timestamp_s, numbers.Real)
                                    or not np.isfinite(timestamp_s) or timestamp_s < 0):
        raise ValueError("timestamp_s must be finite and nonnegative, or None for an offline image.")
    if (not isinstance(stem, str) or not stem or stem in (".", "..")
            or any(c in stem for c in '/\\:<>"|?*') or any(ord(c) < 32 for c in stem)
            or stem.endswith((" ", "."))):
        raise ValueError("stem must be a single valid filename component.")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if stem.split(".")[0].upper() in reserved:
        raise ValueError("stem must not use a reserved Windows filename.")
    rgb = require_raw_rgb(rgb)
    probability, mask = np.asarray(prediction.probability), np.asarray(prediction.mask)
    if probability.dtype != np.float32 or probability.shape != SHAPE[:2] or not np.isfinite(probability).all():
        raise ValueError("Export probability must be finite float32 with shape (512,512).")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Export probability must be in [0,1].")
    if (mask.dtype != np.uint8 or mask.shape != SHAPE[:2] or not np.isin(mask, [0, 1]).all()
            or not np.array_equal(mask, (probability >= .5).astype(np.uint8))):
        raise ValueError("Export mask must be uint8 0/1 and match the locked >=0.5 probability threshold.")
    if not np.isfinite(prediction.inference_ms) or prediction.inference_ms < 0:
        raise ValueError("Inference duration must be finite and nonnegative.")
    if not isinstance(prediction.output_device, str) or not prediction.output_device:
        raise ValueError("Inference output device must be recorded.")
    if model_lock is not None and (model_lock.get("threshold") != .5
            or model_lock.get("threshold_comparator") != ">=" or model_lock.get("mask_labels") != [0, 1]):
        raise ValueError("Export model lock does not match the fixed output protocol.")
    base = Path(output_dir).expanduser().resolve()
    folder = base / stem
    metadata = {"schema_version": "0.1-draft", "kind": "road_segmentation_camera_frame",
                "frame": "model_image_pixels", "units": "pixels", "timestamp_s": float(timestamp_s) if timestamp_s is not None else None,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "timestamp_source": "provided_by_caller" if timestamp_s is not None else "offline_capture_time_unknown",
                "image_shape_hwc": [512, 512, 3], "mask_values": [0, 1],
                "mask_file": "road_mask_01.png", "probability_file": "road_probability.npy",
                "probability_dtype": "float32", "probabilities_are_calibrated_confidence": False,
                "threshold": .5, "threshold_comparator": ">=", "road_fraction": float(prediction.mask.mean()),
                "source": source_metadata, "checkpoint_sha256": model_lock["checkpoint_sha256"] if model_lock else None,
                "inference_ms_excludes_decode_and_export": prediction.inference_ms,
                "output_device": prediction.output_device,
                "metric_projection_available": False, "actor_detection_available": False,
                "planning_map": None, "planning_use_valid": False,
                "note": "A camera road-surface mask is not a metric map or an obstacle/safety decision."}
    # Validate metadata before creating any output. frame.json is published last;
    # its absence identifies an interrupted frame export without hiding evidence.
    json.dumps(metadata, allow_nan=False)
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(mask).save(folder / "road_mask_01.png")
    np.save(folder / "road_probability.npy", probability, allow_pickle=False)
    Image.fromarray(rgb).save(folder / "model_input_rgb.png")
    Image.fromarray(mask*255).save(folder / "mask_display.png")
    overlay = rgb.copy()
    road = mask == 1
    overlay[road] = (.6*rgb[road]+.4*np.array([30, 220, 110])).astype(np.uint8)
    Image.fromarray(overlay).save(folder / "road_overlay.png")
    write_json_atomic(folder / "frame.json", metadata)
    return folder
