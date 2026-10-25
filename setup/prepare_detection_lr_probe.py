"""Copy a small, labelled development cohort for a paired learning-rate experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idd_detection_data import file_sha, save_json
from project_runtime import linux_path


def main(output):
    import numpy as np
    assert output.parent == ROOT / "runs" and output.name.startswith("detection_lr_probe_")
    output.mkdir(exist_ok=False)
    dataset = ROOT / "dataset/idd117k_detection_v3"
    prepared = json.loads((dataset / "preparation.json").read_text())
    cache = np.load(dataset / "labels/train.cache", allow_pickle=True).item()["labels"]
    assert len(cache) == prepared["counts"]["train"] == 56729
    names = prepared["names"]
    rng = np.random.default_rng(20261025)
    selected = set()
    weights = []
    for class_id in range(12):
        eligible = [i for i, row in enumerate(cache) if class_id in row["cls"].reshape(-1)]
        selected.update(int(i) for i in rng.choice(eligible, 2, replace=False))
    for row in cache:
        classes = row["cls"].reshape(-1).astype(int)
        weights.append(max([1] + [int({4: 2, 8: 4, 9: 2}.get(int(c), 1)) for c in classes]))
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    draws = sorted(selected) + rng.choice(len(cache), size=256-len(selected), p=weights, replace=True).tolist()
    rng.shuffle(draws)
    probe = json.loads((ROOT / "runs/detection_fast_probe_20261025_01/selection.json").read_text())
    records = []
    counts = {split: np.zeros(12, dtype=int) for split in ("train", "val")}
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            (output / "dataset" / folder / split).mkdir(parents=True)
        chosen = [(i, Path(cache[index]["im_file"]).name) for i, index in enumerate(draws)] if split == "train" else list(enumerate(row["image"] for row in probe["records"]))
        for index, basename in chosen:
            image = dataset / "images" / split / basename
            label = dataset / "labels" / split / (Path(basename).stem + ".txt")
            target = (f"{index:04d}_" if split == "train" else "") + basename
            dst_image = output / "dataset/images" / split / target
            dst_label = output / "dataset/labels" / split / (Path(target).stem + ".txt")
            shutil.copy2(image, dst_image)
            shutil.copy2(label, dst_label)
            image_sha, label_sha = file_sha(image), file_sha(label)
            assert file_sha(dst_image) == image_sha and file_sha(dst_label) == label_sha
            if split == "val":
                expected = probe["records"][index]
                assert image_sha == expected["image_sha256"] and label_sha == expected["label_sha256"]
            for line in label.read_text().splitlines():
                if line.strip():
                    counts[split][int(line.split()[0])] += 1
            records.append({"split": split, "source_image": basename, "prepared_image": target,
                "image_sha256": image_sha, "label_sha256": label_sha})
    assert all(counts["train"] > 0) and counts["val"].tolist() == [probe["objects"][name] for name in names]
    linux_root = linux_path(output)
    save_json(output / "dataset/data.yaml", {"path": linux_root + "/dataset", "train": "images/train",
        "val": "images/val", "names": dict(enumerate(names))})
    parent = ROOT / "runs/detection_train_20261024_04"
    verified = json.loads((parent / "refinement_result.json").read_text())
    source = parent / "weights/best.pt"
    assert verified["epochs_completed"] == 30 and verified["checkpoint_checks"]["best"]["all_tensors_finite"]
    assert file_sha(source) == verified["checkpoint_checks"]["best"]["sha256"]
    shutil.copy2(source, output / "starting_checkpoint.pt")
    assert file_sha(output / "starting_checkpoint.pt") == file_sha(source)
    save_json(output / "selection.json", {"passed": True, "seed": 20261025, "names": names,
        "train_draws": 256, "unique_source_train_images": len(set(draws)), "val_images": 192,
        "objects": {split: dict(zip(names, value.tolist())) for split, value in counts.items()},
        "training_selection": "At least two source images per class, then fixed weighted draws using existing rare-class boosts; duplicates intentional",
        "validation_selection": "Same fixed stratified 192-image development sample used by earlier screening; not full validation",
        "starting_checkpoint_sha256": file_sha(source), "starting_parent_epoch": 30,
        "records": records, "base_learning_rate": .00003, "candidate_learning_rate": .0000375,
        "increase_percent": 25, "resolution": 1280, "probe_batch": 1, "probe_nominal_batch": 16,
        "epochs_per_arm": 1, "cpu_only": True, "official_test_used": False,
        "note": "Paired small-cohort screening differs from full GPU training in batch/accumulation and coverage. A win requires larger confirmation.",
        "created_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"passed": True, "train_draws": 256, "validation_images": 192,
        "output": str(output), "increase_percent": 25}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
