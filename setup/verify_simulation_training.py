"""Check masked math, global metrics, GPU learning, frozen encoder and reload.

Uses training/validation samples only. The discarded smoke checkpoint cannot be
used to initialize the real simulation training run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from idd_model import BCEDiceLoss, RoadIoU, RoadDice, build_unet, WEIGHTS_NAME
from simulation_data import make_dataset
from simulation_model import MaskedBCEDiceLoss, MaskedRoadIoU, MaskedRoadDice, compile_simulation_model, pack_targets
from train_simulation import configure_gpu, dataset_checks, source_hashes, MANIFEST_SHA, utc
from setup.prepare_simulation_dataset import save, protected


def weight_hash(weights):
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.asarray(weight).tobytes())
    return digest.hexdigest()


def math_checks():
    truth = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], np.float32).reshape(2, 2, 3, 1)
    valid = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], np.float32).reshape(truth.shape)
    prediction = np.array([.9, .8, .1, .1, .9, .2, .6, .2, .8, .7, .2, .9], np.float32).reshape(truth.shape)
    packed = np.concatenate([truth, valid], axis=-1)
    objective = MaskedBCEDiceLoss()
    actual = objective.call(packed, prediction).numpy()
    bce = -(truth * np.log(prediction) + (1-truth) * np.log(1-prediction))
    axes = (1, 2, 3)
    expected = (bce * valid).sum(axes) / valid.sum(axes) + 1 - (
        2 * (truth * prediction * valid).sum(axes) + 1e-6
    ) / (((truth + prediction) * valid).sum(axes) + 1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    variable = tf.Variable(prediction)
    with tf.GradientTape() as tape:
        loss = objective(packed, variable)
    gradient = tape.gradient(loss, variable).numpy()
    assert np.isfinite(gradient).all() and np.any(gradient[valid == 1] != 0)
    np.testing.assert_array_equal(gradient[valid == 0], 0)
    altered_truth = np.where(valid, truth, 1-truth)
    altered_prediction = np.where(valid, prediction, 1-prediction)
    altered = np.concatenate([altered_truth, valid], -1)
    np.testing.assert_allclose(objective.call(altered, altered_prediction), actual, atol=1e-7)
    full = np.concatenate([truth, np.ones_like(valid)], -1)
    np.testing.assert_allclose(objective.call(full, prediction), BCEDiceLoss().call(truth, prediction), atol=1e-6)
    tp = int(np.count_nonzero((truth >= .5) & (prediction >= .5) & (valid == 1)))
    fp = int(np.count_nonzero((truth < .5) & (prediction >= .5) & (valid == 1)))
    fn = int(np.count_nonzero((truth >= .5) & (prediction < .5) & (valid == 1)))
    assert (tp, fp, fn) == (3, 2, 1)
    for masked_cls, original_cls, expected_metric in (
        (MaskedRoadIoU, RoadIoU, tp/(tp+fp+fn)),
        (MaskedRoadDice, RoadDice, 2*tp/(2*tp+fp+fn)),
    ):
        metric = masked_cls()
        # Independent updates prove accumulation across uneven batches.
        metric.update_state(packed[:1], prediction[:1])
        metric.update_state(packed[1:], prediction[1:])
        np.testing.assert_allclose(metric.result(), expected_metric)
        np.testing.assert_array_equal([metric.tp.numpy(), metric.fp.numpy(), metric.fn.numpy()], [tp, fp, fn])
        metric.reset_state()
        metric.update_state(altered, altered_prediction)
        np.testing.assert_allclose(metric.result(), expected_metric)
        metric.reset_state()
        metric.update_state(full, prediction)
        original = original_cls()
        original.update_state(truth, prediction)
        np.testing.assert_allclose(metric.result(), original.result())
        metric.reset_state()
        metric.update_state(np.concatenate([np.zeros_like(truth), valid], -1), np.zeros_like(prediction))
        np.testing.assert_allclose(metric.result(), 1.)
    return {"manual_bce_dice": True, "ignored_pixel_gradient_zero": True,
            "ignored_prediction_and_label_invariant": True, "fully_valid_matches_real_loss_and_metrics": True,
            "global_counts_across_batches": {"tp": tp, "fp": fp, "fn": fn}, "empty_road_overlap": 1.0}


def verify(dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_utc": utc(), "pid": os.getpid(),
              "test_images_read": False, "manifest_sha256": MANIFEST_SHA,
              "source_sha256": source_hashes()}
    save(output / "verification.json", report)
    before = None
    try:
        report["gpu"] = configure_gpu()
        pairs, before = dataset_checks(dataset)
        report["math"] = math_checks()
        # Exercise the final smaller batch without discarding a real sample.
        batches = list(make_dataset(pairs["train"][:5], batch_size=4))
        assert [int(b[0].shape[0]) for b in batches] == [4, 1]
        image, truth, valid = batches[0]
        assert image.dtype == tf.float32 and float(tf.reduce_min(image)) >= -1 and float(tf.reduce_max(image)) <= 1
        augmented = next(iter(make_dataset(pairs["train"][:4], training=True, batch_size=4)))
        for target in augmented[1:]:
            assert set(np.unique(target.numpy())).issubset({0., 1.})
        assert np.count_nonzero(augmented[1].numpy() * (1-augmented[2].numpy())) == 0
        model = compile_simulation_model(build_unet(ROOT / "models/cache/weights" / WEIGHTS_NAME))
        encoder = model.get_layer("mobilenetv2_encoder")
        assert not encoder.trainable and not encoder.trainable_weights
        encoder_before = weight_hash(encoder.weights)
        decoder_before = weight_hash(model.trainable_weights)
        packed = pack_targets(*augmented)[1]
        learned = model.train_on_batch(augmented[0], packed, return_dict=True)
        assert all(np.isfinite(float(x)) for x in learned.values()), learned
        assert int(model.optimizer.iterations) == 1
        assert weight_hash(encoder.weights) == encoder_before
        assert weight_hash(model.trainable_weights) != decoder_before
        validation = next(iter(make_dataset(pairs["val"][:4], batch_size=4)))
        val_image, val_packed = pack_targets(*validation)
        prediction = model(val_image, training=False)
        assert "/device:GPU:" in prediction.device, prediction.device
        assert prediction.shape == (4, 512, 512, 1) and prediction.dtype == tf.float32
        assert np.isfinite(prediction.numpy()).all()
        assert float(tf.reduce_min(prediction)) >= 0 and float(tf.reduce_max(prediction)) <= 1
        model.reset_metrics()
        reference_metrics = model.test_on_batch(val_image, val_packed, return_dict=True)
        checkpoint = output / "discarded_smoke.keras"
        model.save(checkpoint)
        reloaded = tf.keras.models.load_model(checkpoint)
        np.testing.assert_allclose(reloaded(val_image, training=False), prediction, rtol=1e-6, atol=1e-6)
        assert int(reloaded.optimizer.iterations) == 1
        reloaded.reset_metrics()
        replay = reloaded.test_on_batch(val_image, val_packed, return_dict=True)
        for key, value in reference_metrics.items():
            np.testing.assert_allclose(replay[key], value, rtol=1e-6, atol=1e-6)
        # Inference consumers can load an ordinary 1-channel probability model.
        inference = tf.keras.models.load_model(checkpoint, compile=False)
        np.testing.assert_allclose(inference(val_image, training=False), prediction, rtol=1e-6, atol=1e-6)
        report.update(status="passed", completed_utc=utc(), gpu_batch_size=4,
                      gpu_prediction_device=prediction.device, train_step={k: float(v) for k,v in learned.items()},
                      validation_smoke={k: float(v) for k,v in reference_metrics.items()},
                      frozen_encoder_and_batchnorm_unchanged=True, decoder_updated=True,
                      smaller_batch_retained=True, compiled_and_inference_reload_match=True,
                      optimizer_state_restored=True, training_initialization="fresh; smoke model discarded")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if before is not None:
            report["protected_real_unchanged"] = protected() == before
            if not report["protected_real_unchanged"]:
                report.update(status="failed", error="Protected real model files changed")
        save(output / "verification.json", report)
        print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/simulation_india_v5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset.resolve(), args.output.resolve())
