"""Bounded GPU-only demonstration. Does not read data or train/save a model."""
import json
import time
from pathlib import Path

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError('No TensorFlow GPU found')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)


@tf.function
def multiply(a, b):
    with tf.device('/GPU:0'):
        return tf.matmul(a, b)


with tf.device('/GPU:0'):
    a = tf.ones((4096, 4096), dtype=tf.float32)
    b = tf.ones((4096, 4096), dtype=tf.float32)
result = multiply(a, b)
assert 'GPU:0' in result.device, result.device
assert float(result[0, 0].numpy()) == 4096.0
started = time.monotonic()
iterations = 0
while time.monotonic() - started < 25:
    result = multiply(a, b)
    result[0, 0].numpy()  # Synchronize each operation before continuing.
    iterations += 1
report = {'status': 'passed', 'operation_device': result.device,
          'elapsed_seconds': time.monotonic() - started, 'operations': iterations,
          'full_training_started': False, 'dataset_accessed': False,
          'note': 'GPU activity demonstration, not a model speed benchmark.'}
path = Path(__file__).resolve().parent / 'setup' / 'gpu_activity_demo.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
