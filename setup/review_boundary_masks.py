"""Review previously exported validation masks without re-running either model."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from boundary_metrics import counts, scores, summarize

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def evaluate(mask_dir, output, reference_csv=None):
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / "dataset/splits/pair_splits.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = sorted(int(r["id"]) for r in csv.DictReader(stream) if r["split"] == "val")
    assert len(ids) == 687
    assert {p.name for p in mask_dir.glob("Prediction_*.png")} == {f"Prediction_{i}.png" for i in ids}
    refs = None
    if reference_csv:
        with reference_csv.open(newline="") as stream:
            refs = {int(r["id"]): float(r["road_iou"]) for r in csv.DictReader(stream)}
    rows = []
    target_hash = hashlib.sha256()
    for index, i in enumerate(ids):
        path = ROOT / f"dataset/splits/val/masks/Mask_{i}.png"
        with Image.open(path) as im:
            truth = np.asarray(im)
        with Image.open(mask_dir / f"Prediction_{i}.png") as im:
            pred = np.asarray(im)
        assert truth.shape == pred.shape == (512, 512)
        assert np.isin(truth, [0, 1]).all() and np.isin(pred, [0, 1]).all()
        c = counts(truth, pred)
        row = {"id": i, **c, **scores(c)}
        if refs is not None:
            assert abs(row["road_iou"] - refs[i]) < 1e-12
        rows.append(row)
        target_hash.update(str(i).encode() + bytes.fromhex(sha(path)))
        if (index + 1) % 100 == 0:
            print(f"Boundary review: {index + 1}/687", flush=True)
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update(manifest_sha256=sha(ROOT / "dataset/splits/pair_splits.csv"),
                   validation_mask_digest=target_hash.hexdigest(), threshold=.5, test_images_used=False,
                   metric_settings={"boundary_iou": "5px and 14px inner bands, square erosion, frame included",
                                    "visible_boundary_f1": "3px square tolerance, approximate many-to-one matches, frame excluded",
                                    "global": "counts summed across all 687 images; means separately labelled"},
                   prediction_directory=str(mask_dir), finished_at_utc=datetime.now(timezone.utc).isoformat())
    atomic(output / "metrics.json", summary)
    print(json.dumps(summary["global"]), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    args = parser.parse_args()
    evaluate(args.masks, args.output, args.reference_csv)
