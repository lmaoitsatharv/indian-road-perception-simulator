"""Bounded validation-only diagnosis of batch shape and graph placement."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from idd_inference import LockedRoadSegmenter, read_rgb, normalize_raw_rgb

OUT = ROOT / "runs/inference_diagnosis_20261003"


def fingerprint(model):
    result = hashlib.sha256()
    for weight in model.get_weights():
        result.update(weight.tobytes())
    return result.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with (ROOT / "dataset/splits/pair_splits.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = sorted(int(row["id"]) for row in csv.DictReader(stream) if row["split"] == "val")[:16]
    assert ids[0] == 9
    segmenter = LockedRoadSegmenter(device="gpu", gpu_memory_mib=3072)
    tf = segmenter.tf
    before = fingerprint(segmenter.model)
    images = [read_rgb(ROOT / f"dataset/splits/val/images/Image_{i}.png")[0] for i in ids]
    inputs = np.stack([normalize_raw_rgb(image) for image in images])
    # The same dynamic signature and automatic placement as the earlier export.
    @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
    def export_graph(tensor):
        return segmenter.model(tensor, training=False)
    @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
    def gpu_graph(tensor):
        with tf.device('/GPU:0'):
            return segmenter.model(tensor, training=False)

    probability = segmenter.predict_rgb(images[0]).probability
    dynamic_one = export_graph(inputs[:1]).numpy()[0, ..., 0]
    dynamic_batch = export_graph(inputs).numpy()[0, ..., 0]
    placed_one = gpu_graph(inputs[:1]).numpy()[0, ..., 0]
    placed_batch = gpu_graph(inputs).numpy()[0, ..., 0]
    reference = np.array(Image.open(ROOT / "runs/model_comparisons/baseline_vs_trial13_validation/predicted_masks/trial13/Prediction_9.png"))
    results = {}
    for name, values in [('fixed_batch1', probability), ('dynamic_batch1', dynamic_one),
                         ('dynamic_batch16', dynamic_batch), ('placed_dynamic_batch1', placed_one),
                         ('placed_dynamic_batch16', placed_batch)]:
        disagreement = (values >= .5) != reference.astype(bool)
        results[name] = {
            'disagreement_pixels_vs_saved_batch16': int(disagreement.sum()),
            'maximum_distance_from_threshold_at_disagreement': float(np.max(np.abs(values[disagreement]-.5))) if disagreement.any() else None,
            'maximum_probability_difference_vs_fixed_batch1': float(np.max(np.abs(values-probability))),
            'mean_probability_difference_vs_fixed_batch1': float(np.mean(np.abs(values-probability))),
        }
        np.save(OUT / f'{name}_Image_9_probability.npy', values, allow_pickle=False)
        Image.fromarray((values >= .5).astype(np.uint8)).save(OUT / f'{name}_Image_9_mask01.png')
    assert fingerprint(segmenter.model) == before
    report = {'status':'completed', 'diagnostic_validation_ids':ids, 'compared_id':9,
              'checkpoint_sha256':segmenter.lock['checkpoint_sha256'], 'weight_fingerprint':before,
              'in_memory_weights_unchanged':True, 'test_image_contents_read':False,
              'dtype_policies':{layer.name:layer.dtype_policy.name for layer in segmenter.model.layers},
              'runtime':segmenter.runtime, 'comparisons':results}
    (OUT / 'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
