"""Verify WSL GPU computation and one disposable project training batch.

Uses training images only. No checkpoint writes or full training run.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from idd_data import make_dataset, read_pairs
from idd_model import compile_model

ROOT = Path(__file__).resolve().parent
EXPECTED_MANIFEST = '512cc054e2291ec5e197b0dca913ba94698cf055062b54511977816872005e90'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    report = {'status': 'running', 'python': platform.python_version(),
              'tensorflow': tf.__version__, 'keras': tf.keras.__version__,
              'full_training_started': False, 'test_images_consumed': False}
    destination = ROOT / 'setup' / 'wsl_gpu_verification.json'
    destination.parent.mkdir(exist_ok=True)
    checkpoint = ROOT / 'runs' / 'frozen_01' / 'best.keras'
    manifest = ROOT / 'dataset' / 'splits' / 'pair_splits.csv'
    checkpoint_before, manifest_before = digest(checkpoint), digest(manifest)
    try:
        if manifest_before != EXPECTED_MANIFEST:
            raise RuntimeError('Split manifest changed; stop before GPU verification.')
        physical = tf.config.list_physical_devices('GPU')
        if not physical:
            raise RuntimeError('TensorFlow sees no GPU; GPU setup is incomplete.')
        for device in physical:
            tf.config.experimental.set_memory_growth(device, True)
        report['gpu_devices'] = [tf.config.experimental.get_device_details(d) for d in physical]
        tf.keras.utils.set_random_seed(42)
        # This small operation must execute on the GPU, with no CPU fallback.
        tf.config.set_soft_device_placement(False)
        with tf.device('/GPU:0'):
            result = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
        assert 'GPU:0' in result.device
        np.testing.assert_allclose(result.numpy(), 64.0)
        report['matrix_operation_device'] = result.device
        tf.config.set_soft_device_placement(True)
        pairs = read_pairs(ROOT / 'dataset' / 'splits')
        counts = {key: len(value) for key, value in pairs.items()}
        assert counts == {'train': 5507, 'val': 687, 'test': 689}, counts
        report['split_counts'] = counts
        images, masks = next(iter(make_dataset(pairs['train'], training=False, batch_size=4)))
        assert tuple(images.shape) == (4, 512, 512, 3)
        assert tuple(masks.shape) == (4, 512, 512, 1)
        assert float(tf.reduce_min(images)) >= -1 and float(tf.reduce_max(images)) <= 1
        assert set(np.unique(masks.numpy())).issubset({0, 1})
        with tf.device('/GPU:0'):
            model = tf.keras.models.load_model(checkpoint, compile=False)
            compile_model(model)
            predictions = model(images, training=False)
            assert 'GPU:0' in predictions.device, predictions.device
            started = time.perf_counter()
            # One optimizer update in RAM exercises CUDA/cuDNN backpropagation.
            # This disposable model is never saved and is discarded at exit.
            metrics = model.train_on_batch(images, masks, return_dict=True)
            elapsed = time.perf_counter() - started
        values = {key: float(value) for key, value in metrics.items()}
        assert all(np.isfinite(value) for value in values.values()), values
        report.update(status='passed', batch_size=4, output_shape=list(predictions.shape),
                      prediction_device=predictions.device, disposable_batch_metrics=values,
                      disposable_batch_seconds=elapsed,
                      timing_note='Includes first-call compilation; not an epoch benchmark.',
                      checkpoint_sha256=checkpoint_before, manifest_sha256=manifest_before)
        print('PASS: real IDD batch forward/backward computation works on the GPU.')
        print('Split: 5507 train / 687 validation / 689 test. Checkpoint preserved.')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['checkpoint_unchanged'] = digest(checkpoint) == checkpoint_before
        report['manifest_unchanged'] = digest(manifest) == manifest_before
        destination.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
