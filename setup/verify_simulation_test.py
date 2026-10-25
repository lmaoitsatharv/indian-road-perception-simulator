"""Independently recount all saved test predictions; no TensorFlow/model calls."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.prepare_simulation_dataset import protected, save, sha


def verify(output):
    protocol = json.loads((output / "protocol.json").read_text())
    result = json.loads((output / "test_results.json").read_text())
    assert result["status"] == "completed"
    assert protocol["threshold"] == result["threshold"] == .5
    assert protocol["checkpoint_sha256"] == sha(ROOT / protocol["checkpoint"])
    dataset = ROOT / "dataset/simulation_india_v5"
    assert sha(dataset / "splits/pair_splits.csv") == protocol["manifest_sha256"]
    with (dataset / "splits/pair_splits.csv").open(newline="") as stream:
        metadata = {int(r["id"]): r for r in csv.DictReader(stream) if r["split"] == "test"}
    with (output / "per_image.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(r["id"]) for r in rows]
    assert len(ids) == len(set(ids)) == len(metadata) == 500
    assert set(ids) == set(metadata)
    assert {p.name for p in (output / "predictions").iterdir()} == {f"Prediction_{i}.png" for i in ids}
    counts = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    groups = {k: {} for k in ("episode_id", "layout", "weather", "case")}
    verified_rows = []
    for row in rows:
        identifier = int(row["id"])
        meta = metadata[identifier]
        prediction_path = output / "predictions" / f"Prediction_{identifier}.png"
        assert sha(prediction_path) == row["prediction_sha256"]
        prediction = np.array(Image.open(prediction_path))
        truth = np.array(Image.open(dataset / f"splits/test/masks/Mask_{identifier}.png"))
        validity = np.array(Image.open(dataset / f"splits/test/valid/Valid_{identifier}.png"))
        assert prediction.shape == truth.shape == validity.shape == (512,512)
        assert set(np.unique(prediction)).issubset({0,1})
        assert set(np.unique(truth)).issubset({0,1}) and set(np.unique(validity)).issubset({0,1})
        v = validity.astype(bool)
        # Direct indexed truth/prediction combinations, independent of TF masks.
        t, p = truth[v], prediction[v]
        tally = {"tp": int(np.count_nonzero((t == 1) & (p == 1))),
                 "fp": int(np.count_nonzero((t == 0) & (p == 1))),
                 "fn": int(np.count_nonzero((t == 1) & (p == 0))),
                 "tn": int(np.count_nonzero((t == 0) & (p == 0)))}
        for key, value in tally.items():
            assert value == int(row[key]), (identifier, key)
            counts[key] += value
        assert len(t) == int(row["valid_pixels"]) == sum(tally.values())
        assert 512*512-len(t) == int(row["ignored_pixels"])
        union = tally["tp"] + tally["fp"] + tally["fn"]
        overlap = tally["tp"]/union if union else 1.
        assert abs(overlap-float(row["road_iou"])) < 1e-12
        for key in groups:
            assert row[key] == meta[key]
            group = groups[key].setdefault(meta[key], {"images": 0, "valid_pixels": 0,
                                                       **dict.fromkeys(counts, 0)})
            group["images"] += 1
            group["valid_pixels"] += len(t)
            for field,value in tally.items():
                group[field] += value
        verified_rows.append({"id": identifier, "episode_id": meta["episode_id"],
                              "layout": meta["layout"], "road_iou": overlap})
    assert counts == result["counts"]
    for key, by_name in groups.items():
        for name, group in by_name.items():
            for field, value in group.items():
                assert value == result["by_group"][key][name][field]
    tp, fp, fn, tn = (counts[k] for k in ("tp","fp","fn","tn"))
    expected = {"road_iou": tp/(tp+fp+fn), "road_dice": 2*tp/(2*tp+fp+fn),
                "precision": tp/(tp+fp), "recall": tp/(tp+fn),
                "pixel_accuracy": (tp+tn)/(tp+fp+fn+tn)}
    for key,value in expected.items():
        assert abs(value-result["metrics"][key]) < 1e-12
    assert sum(counts.values()) == result["valid_pixels"]
    assert result["valid_pixels"] + result["ignored_pixels"] == 500*512*512
    assert abs(result["valid_coverage"] - result["valid_pixels"]/(500*512*512)) < 1e-12
    assert protected() == protocol["protected_real_sha256"]
    # Evidence-only sheet: worst saved image from each of four test episodes.
    worst = {}
    for row in verified_rows:
        episode = row["episode_id"]
        if episode not in worst or row["road_iou"] < worst[episode]["road_iou"]:
            worst[episode] = row
    chosen = sorted(worst.values(), key=lambda r: r["layout"])
    sheet = Image.new("RGB", (1024, 50 + len(chosen)*290), "#151920")
    draw = ImageDraw.Draw(sheet)
    for i, label in enumerate(("RGB", "Reference (ignored: magenta)", "Saved prediction", "Errors: FP red / FN blue")):
        draw.text((i*256+8, 12), label, fill="white")
    for index, row in enumerate(chosen):
        identifier = row["id"]
        image = np.array(Image.open(dataset / f"splits/test/images/Image_{identifier}.png").convert("RGB"))
        truth = np.array(Image.open(dataset / f"splits/test/masks/Mask_{identifier}.png")) == 1
        valid = np.array(Image.open(dataset / f"splits/test/valid/Valid_{identifier}.png")) == 1
        prediction = np.array(Image.open(output / f"predictions/Prediction_{identifier}.png")) == 1
        reference = np.repeat((truth.astype(np.uint8)*255)[...,None], 3, axis=-1)
        reference[~valid] = (220,0,220)
        binary = np.repeat((prediction.astype(np.uint8)*255)[...,None], 3, axis=-1)
        errors = image.copy()
        errors[valid & prediction & ~truth] = (255,25,25)
        errors[valid & ~prediction & truth] = (35,120,255)
        errors[~valid] = (errors[~valid].astype(np.float32)*.5 + np.array([110,0,110])).astype(np.uint8)
        top = 50 + index*290
        draw.text((8,top), f"{row['layout']} | image {identifier} | IoU {row['road_iou']*100:.3f}% | worst image in this episode", fill="white")
        for column, panel in enumerate((image, reference, binary, errors)):
            sheet.paste(Image.fromarray(panel).resize((256,256)), (column*256,top+24))
    sheet.save(output / "test_examples.png")
    evidence = {"status": "passed", "images_recounted": 500, "counts": counts,
                "global_metrics": expected, "all_per_image_and_group_counts_match": True,
                "saved_prediction_hashes_match": True, "checkpoint_and_manifest_unchanged": True,
                "protected_real_unchanged": True, "model_inference_repeated": False,
                "sheet": "test_examples.png", "sheet_selection": "worst image in each test episode",
                "source_sha256": sha(Path(__file__))}
    save(output / "independent_verification.json", evidence)
    result["independent_verification"] = "passed"
    result["independent_verification_sha256"] = sha(output / "independent_verification.json")
    save(output / "test_results.json", result)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output.resolve())
