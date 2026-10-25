"""Inference-only export of baseline/trial-13 masks for the validation gallery."""
import argparse
import csv
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import tensorflow as tf
from idd_data import make_dataset, read_pairs
import idd_model  # Saved custom classes; no training is called.

MODELS = {
    'baseline': ROOT / 'runs/frozen_01',
    'trial13': ROOT / 'runs/overnight_20261003/trial_13_shadow_fixed_bn_low_lr',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(status='preparing', split='val', test_images_opened=False,
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    atomic(output / 'inference_status.json', status)
    try:
        manifest = ROOT / 'dataset/splits/pair_splits.csv'
        before_hashes = {'manifest': sha(manifest)}
        records_by_split = read_pairs(ROOT / 'dataset/splits')
        assert {k: len(v) for k, v in records_by_split.items()} == dict(train=5507, val=687, test=689)
        records = records_by_split['val']
        gpu_devices = tf.config.list_physical_devices('GPU')
        if not gpu_devices:
            raise RuntimeError('The verified WSL GPU environment is required.')
        tf.config.set_logical_device_configuration(gpu_devices[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=4096)
        ])
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        results = {}
        for name, run in MODELS.items():
            checkpoint = run / 'best.keras'
            config = json.loads((run / 'run_config.json').read_text())
            verified = json.loads((run / 'validation_review/validation_metrics.json').read_text())
            checkpoint_hash = sha(checkpoint)
            assert before_hashes['manifest'] == config['manifest_sha256'] == verified['manifest_sha256']
            assert checkpoint_hash == verified['checkpoint_sha256']
            before_hashes[name] = checkpoint_hash
            mask_dir = output / 'predicted_masks' / name
            mask_dir.mkdir(parents=True)
            status.update(status='exporting', model=name, completed=0, total=len(records))
            atomic(output / 'inference_status.json', status)
            print(f'Exporting {name}: {len(records)} validation images, batch {config["batch_size"]}', flush=True)
            model = tf.keras.models.load_model(checkpoint, compile=False)

            @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
            def predict_graph(images):
                return model(images, training=False)

            dataset = make_dataset(records, training=False, batch_size=config['batch_size'])
            rows, totals, offset = [], np.zeros(3, dtype=np.int64), 0
            prediction_device = None
            for images, masks in dataset:
                probabilities = predict_graph(images)
                prediction_device = probabilities.device
                values = probabilities.numpy()[..., 0]
                assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
                targets = masks.numpy()[..., 0]
                assert np.isin(targets, [0, 1]).all()
                predictions = values >= 0.5
                for index, prediction in enumerate(predictions):
                    key = records[offset + index]['id']
                    truth = targets[index].astype(bool)
                    tp = int(np.count_nonzero(truth & prediction))
                    fp = int(np.count_nonzero(~truth & prediction))
                    fn = int(np.count_nonzero(truth & ~prediction))
                    totals += [tp, fp, fn]
                    denominator = tp + fp + fn
                    rows.append(dict(id=key, road_iou=tp / denominator if denominator else 1.0,
                                     road_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0))
                    Image.fromarray(prediction.astype(np.uint8)).save(mask_dir / f'Prediction_{key}.png')
                offset += len(predictions)
                if offset == len(records) or offset % 80 < config['batch_size']:
                    status['completed'] = offset
                    atomic(output / 'inference_status.json', status)
                    print(f'{name}: {offset}/{len(records)} saved', flush=True)
            assert offset == 687 and len(list(mask_dir.glob('Prediction_*.png'))) == 687
            tp, fp, fn = map(int, totals)
            global_iou = tp / (tp + fp + fn)
            with (run / 'validation_review/per_image_metrics.csv').open(newline='', encoding='utf-8') as stream:
                reference = {int(row['id']): float(row['road_iou']) for row in csv.DictReader(stream)}
            assert set(reference) == {row['id'] for row in rows}
            difference = abs(global_iou - verified['global_road_iou'])
            assert difference <= 1e-6, f'{name}: inference differs from the reviewed checkpoint: {difference}'
            subset_mismatches = []
            for existing in (run / 'validation_review/predicted_masks').glob('Prediction_*.png'):
                if not np.array_equal(np.asarray(Image.open(existing)), np.asarray(Image.open(mask_dir / existing.name))):
                    subset_mismatches.append(existing.stem)
            results[name] = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=checkpoint_hash,
                                 batch_size=config['batch_size'], images=687, global_road_iou=global_iou,
                                 global_road_dice=2 * tp / (2 * tp + fp + fn), tp=tp, fp=fp, fn=fn,
                                 prediction_device=prediction_device, reference_iou_difference=difference,
                                 maximum_per_image_reference_difference=max(abs(row['road_iou'] - reference[row['id']]) for row in rows),
                                 earlier_example_mask_mismatches=subset_mismatches)
            with (output / f'{name}_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'road_iou', 'road_dice'])
                writer.writeheader()
                writer.writerows(rows)
            del predict_graph, model, dataset
            tf.keras.backend.clear_session()
            gc.collect()
        assert sha(manifest) == before_hashes['manifest']
        assert all(sha(run / 'best.keras') == before_hashes[name] for name, run in MODELS.items())
        status.update(status='completed', model=None, completed=687, total=687, threshold=0.5,
                      models=results, gpu=details, manifest_sha256=before_hashes['manifest'],
                      original_checkpoints_and_manifest_unchanged=True,
                      mask_values=[0, 1], normalization='Existing loader, once to [-1,1], no augmentation',
                      elapsed_seconds=time.perf_counter() - started,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        atomic(output / 'inference_status.json', status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        status.update(status='failed', error=repr(error))
        atomic(output / 'inference_status.json', status)
        raise


if __name__ == '__main__':
    main()
