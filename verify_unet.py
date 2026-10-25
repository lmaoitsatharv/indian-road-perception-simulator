"""Verify a real U-Net forward/backward pass without applying optimizer updates."""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from idd_data import build_datasets
from idd_model import BCEDiceLoss, DECODER_FILTERS, RoadDice, RoadIoU, SKIP_NAMES, WEIGHTS_URL, build_unet, compile_model, get_encoder_weights


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_fingerprint(model):
    digest = hashlib.sha256()
    for value in model.get_weights():
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_metrics_and_loss():
    # Hand-computed fixture: TP=1, FP=1, FN=1 => IoU=1/3; Dice=1/2.
    truth = tf.constant([[[[1.0], [1.0], [0.0], [0.0]]]])
    prediction = tf.constant([[[[0.8], [0.2], [0.7], [0.1]]]])
    for cls, expected in ((RoadIoU, 1 / 3), (RoadDice, 0.5)):
        metric = cls()
        # Updating in two pieces also checks aggregation across batches.
        metric.update_state(truth[..., :2, :], prediction[..., :2, :])
        metric.update_state(truth[..., 2:, :], prediction[..., 2:, :])
        np.testing.assert_allclose(metric.result().numpy(), expected)
        metric.reset_state()
        metric.update_state(tf.zeros_like(truth), tf.zeros_like(truth))
        assert float(metric.result().numpy()) == 1.0
        restored = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
        assert isinstance(restored, cls)
    loss_fn = BCEDiceLoss()
    expected_bce = -np.log([0.8, 0.2, 1 - 0.7, 1 - 0.1]).mean()
    expected_dice = (2 * (0.8 + 0.2) + 1e-6) / (2 + 0.8 + 0.2 + 0.7 + 0.1 + 1e-6)
    np.testing.assert_allclose(loss_fn(truth, prediction).numpy(), expected_bce + 1 - expected_dice, rtol=1e-6)
    restored_loss = tf.keras.losses.deserialize(tf.keras.losses.serialize(loss_fn))
    assert isinstance(restored_loss, BCEDiceLoss)
    return True


def verify_model(model, project_root, weights_path):
    root = Path(project_root).resolve()
    out = root / "models"
    if (out / "retained_models.json").exists():
        raise RuntimeError(
            "Construction was already verified before training. See models/model_verification.json "
            "and models/retained_models.json. Use an isolated project copy for a new setup verification "
            "so it does not replace completed-project metadata or recreate the removed setup binary."
        )
    out.mkdir(exist_ok=True)
    checkpoint = out / "unet_initial_UNTRAINED.keras"
    if checkpoint.exists():
        raise FileExistsError(f"Preserving the previous setup checkpoint: {checkpoint}")
    manifest = root / "dataset" / "splits" / "pair_splits.csv"
    manifest_hash = sha256(manifest)
    datasets, counts = build_datasets(manifest.parent, batch_size=4, seed=42)
    assert counts == {"train": 5507, "val": 687, "test": 689}
    # Use one training batch. Do not consume test images or report test metrics.
    images, masks = next(iter(datasets["train"]))
    assert images.shape == (4, 512, 512, 3)
    assert masks.shape == (4, 512, 512, 1)
    assert float(tf.reduce_min(images)) >= -1 and float(tf.reduce_max(images)) <= 1
    assert set(np.unique(masks.numpy())) == {0.0, 1.0}
    encoder = model.get_layer("mobilenetv2_encoder")
    assert not encoder.trainable and not encoder.trainable_weights
    assert tuple(model.input_shape) == (None, 512, 512, 3)
    assert tuple(model.output_shape) == (None, 512, 512, 1)
    assert not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers + encoder.layers)
    encoder_before = weight_fingerprint(encoder)
    compile_model(model)

    start = time.perf_counter()
    probabilities = model(images, training=False).numpy()
    forward_seconds = time.perf_counter() - start
    assert probabilities.shape == (4, 512, 512, 1)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    verify_metrics_and_loss()
    print(f"Real batch forward pass passed in {forward_seconds:.2f} seconds", flush=True)

    # Save the fresh decoder BEFORE the gradient probe changes its BatchNorm state.
    # This is an initial setup artifact, not a trained segmentation checkpoint.
    model.save(checkpoint)
    with (out / "model_summary.txt").open("w", encoding="utf-8") as stream:
        model.summary(print_fn=lambda text: stream.write(text + "\n"))
    start = time.perf_counter()
    with tf.GradientTape() as tape:
        one_prediction = model(images[:1], training=True)
        loss = BCEDiceLoss()(masks[:1], one_prediction)
    gradients = tape.gradient(loss, model.trainable_weights)
    assert np.isfinite(float(loss))
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients)
    norm = float(tf.linalg.global_norm(gradients))
    assert np.isfinite(norm) and norm > 0
    gradient_seconds = time.perf_counter() - start
    assert int(model.optimizer.iterations.numpy()) == 0
    assert encoder_before == weight_fingerprint(encoder)
    print(f"Finite, nonzero decoder gradients passed in {gradient_seconds:.2f} seconds; no optimizer updates applied", flush=True)

    restored = tf.keras.models.load_model(checkpoint, compile=False)
    restored_prediction = restored(images[:1], training=False).numpy()
    np.testing.assert_allclose(restored_prediction, probabilities[:1], atol=1e-6, rtol=1e-5)
    assert sha256(manifest) == manifest_hash
    report = {
        "status": "passed", "stage": "model_construction_and_verification",
        "segmentation_training_started": False, "optimizer_updates": 0,
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "tensorflow_devices": [str(device) for device in tf.config.list_physical_devices()],
        "tensorflow_gpu_devices": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "split_counts": counts, "manifest_sha256": manifest_hash,
        "encoder_weights_source": WEIGHTS_URL, "encoder_weights_file": str(weights_path),
        "encoder_weights_sha256": sha256(weights_path),
        "encoder_feature_layers": list(SKIP_NAMES),
        "encoder_feature_shapes": [list(tensor.shape) for tensor in encoder.outputs],
        "decoder_filters": list(DECODER_FILTERS),
        "total_parameters": model.count_params(),
        "trainable_parameters": int(sum(np.prod(weight.shape) for weight in model.trainable_weights)),
        "input_shape": list(images.shape), "output_shape": list(probabilities.shape),
        "output_range": [float(probabilities.min()), float(probabilities.max())],
        "gradient_probe_loss": float(loss), "gradient_global_norm": norm,
        "forward_batch4_seconds": forward_seconds,
        "forward_and_gradient_batch1_seconds": gradient_seconds,
        "timing_scope": "Single local CPU verification passes, not a training throughput or final latency benchmark.",
        "checks": {
            "pretrained_encoder_loaded": True, "encoder_frozen_and_unchanged": True,
            "single_existing_normalization": True, "output_shape_and_probability_range": True,
            "finite_nonzero_decoder_gradients": True, "loss_and_metric_hand_calculations": True,
            "custom_loss_and_metrics_serialize": True, "checkpoint_roundtrip_matches": True,
            "manifest_unchanged": True, "no_test_images_consumed": True,
        },
        "initial_model_path": str(checkpoint), "initial_model_sha256": sha256(checkpoint),
        "interpretation": "ImageNet encoder is pretrained; road decoder is untrained. Shape/gradient checks establish that the model can learn, not its road segmentation performance.",
    }
    (out / "model_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    config = {
        "architecture": "U-Net with MobileNetV2 alpha=1.0 ImageNet encoder",
        "input_shape": [512, 512, 3], "output_shape": [512, 512, 1], "output_activation": "sigmoid",
        "preprocessing": "Existing idd_data.py normalizes RGB once to [-1,1]; masks stay 0/1.",
        "seed": 42, "batch_size": 4, "manifest_sha256": manifest_hash,
        "decoder_filters": list(DECODER_FILTERS), "skip_layers": list(SKIP_NAMES),
        "new_implementation_choices": {
            "initial_encoder_frozen": True, "encoder_batchnorm_mode": "inference",
            "loss": "mean per-image BCE + (1 - per-image soft Dice), equal weights, smooth=1e-6",
            "optimizer": "Adam, learning_rate=0.001, clipnorm=1.0",
            "maximum_initial_epochs": 20, "threshold": 0.5,
            "checkpoint_monitor": "val_road_iou, maximize", "early_stopping_patience": 5,
            "metric_aggregation": "TP/FP/FN summed across all pixels/batches in a split; class-1 road only",
            "empty_union_rule": "Global empty/empty overlap is 1. No per-image IoU/Dice averaging.",
            "optional_fine_tuning": "After validation review, unfreeze blocks 13-16 except BN; Adam lr=1e-5; new run and recompile.",
        },
        "training_status": "not_started",
    }
    (out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Model saved and reload verified: {checkpoint}", flush=True)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    tf.keras.utils.set_random_seed(42)
    weights = get_encoder_weights(root / "models" / "cache")
    report = verify_model(build_unet(weights), root, weights)
    print(json.dumps(report, indent=2))
