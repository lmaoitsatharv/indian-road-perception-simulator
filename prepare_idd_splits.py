"""Copy the numbered IDD image/mask pairs into reproducible 80/10/10 splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


DATASET = Path(__file__).resolve().parent / "dataset"
SOURCE = DATASET / "IDD_RESIZED"
DESTINATION = DATASET / "splits"
STAGING = DATASET / ".splits_staging_idd"
SEED = 42


def inventory(folder: Path, prefix: str) -> dict[int, Path]:
    if not folder.is_dir():
        raise RuntimeError(f"Missing source directory: {folder}")
    pattern = re.compile(rf"{prefix}_(\d+)\.png", re.IGNORECASE)
    found: dict[int, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or not (match := pattern.fullmatch(path.name)):
            raise RuntimeError(f"Unexpected source entry: {path}")
        image_id = int(match.group(1))
        if image_id in found:
            raise RuntimeError(f"Duplicate {prefix} ID {image_id}: {found[image_id]}, {path}")
        found[image_id] = path
    return found


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def duplicate_stats(hashes: dict[int, str]) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, digest in hashes.items():
        groups[digest].append(image_id)
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    return {
        "groups": len(duplicates),
        "extra_files": sum(len(ids) - 1 for ids in duplicates),
        "example_id_groups": duplicates[:10],
    }


def main() -> None:
    if DESTINATION.exists() or STAGING.exists():
        raise RuntimeError("A split or staging directory already exists; refusing to overwrite it")

    images = inventory(SOURCE / "image_archive", "Image")
    masks = inventory(SOURCE / "mask_archive", "Mask")
    image_ids, mask_ids = set(images), set(masks)
    if image_ids != mask_ids:
        raise RuntimeError(
            f"Unmatched images: {sorted(image_ids - mask_ids)[:20]}; "
            f"unmatched masks: {sorted(mask_ids - image_ids)[:20]}"
        )
    ids = sorted(image_ids)
    if not ids:
        raise RuntimeError("No image/mask pairs found")

    source_bytes = sum(path.stat().st_size for path in (*images.values(), *masks.values()))
    if shutil.disk_usage(DATASET).free < source_bytes + 1_000_000_000:
        raise RuntimeError("Insufficient free disk space for copies and a safety margin")

    print(f"Auditing {len(ids)} source pairs and checking exact duplicates...", flush=True)
    image_hash = {image_id: sha256(images[image_id]) for image_id in ids}
    mask_hash = {image_id: sha256(masks[image_id]) for image_id in ids}
    image_duplicates = duplicate_stats(image_hash)
    mask_duplicates = duplicate_stats(mask_hash)
    pair_hash = {image_id: image_hash[image_id] + mask_hash[image_id] for image_id in ids}
    pair_duplicates = duplicate_stats(pair_hash)
    print(
        f"Duplicate images: {image_duplicates['extra_files']}; "
        f"duplicate complete pairs: {pair_duplicates['extra_files']}",
        flush=True,
    )

    # Exact duplicate images stay in one split, avoiding direct image leakage.
    image_groups: dict[str, list[int]] = defaultdict(list)
    for image_id in ids:
        image_groups[image_hash[image_id]].append(image_id)
    groups = list(image_groups.values())
    random.Random(SEED).shuffle(groups)

    n = len(ids)
    val_count = round(n * 0.1)
    targets = {"train": n - 2 * val_count, "val": val_count, "test": val_count}
    counts = {split: 0 for split in targets}
    assignment: dict[int, str] = {}
    for group in groups:
        split = max(targets, key=lambda name: targets[name] - counts[name])
        for image_id in group:
            assignment[image_id] = split
        counts[split] += len(group)

    for split in targets:
        (STAGING / split / "images").mkdir(parents=True, exist_ok=False)
        (STAGING / split / "masks").mkdir(parents=True, exist_ok=False)

    rows = []
    for index, image_id in enumerate(ids, 1):
        split = assignment[image_id]
        image_copy = STAGING / split / "images" / images[image_id].name
        mask_copy = STAGING / split / "masks" / masks[image_id].name
        shutil.copy2(images[image_id], image_copy)
        shutil.copy2(masks[image_id], mask_copy)
        if sha256(image_copy) != image_hash[image_id]:
            raise RuntimeError(f"Image copy failed verification: {image_copy}")
        if sha256(mask_copy) != mask_hash[image_id]:
            raise RuntimeError(f"Mask copy failed verification: {mask_copy}")
        rows.append(
            {
                "id": image_id,
                "split": split,
                "source_image": str(images[image_id]),
                "source_mask": str(masks[image_id]),
                "split_image": str(DESTINATION / split / "images" / images[image_id].name),
                "split_mask": str(DESTINATION / split / "masks" / masks[image_id].name),
                "image_sha256": image_hash[image_id],
                "mask_sha256": mask_hash[image_id],
            }
        )
        if index % 1000 == 0 or index == n:
            print(f"Copied and verified {index}/{n} pairs", flush=True)

    with (STAGING / "pair_splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    actual = {}
    for split in targets:
        image_copies = list((STAGING / split / "images").iterdir())
        mask_copies = list((STAGING / split / "masks").iterdir())
        if len(image_copies) != counts[split] or len(mask_copies) != counts[split]:
            raise RuntimeError(f"Count verification failed for {split}")
        actual[split] = {"images": len(image_copies), "masks": len(mask_copies)}

    report = {
        "seed": SEED,
        "requested_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "source_pairs": n,
        "source_image_files": len(images),
        "source_mask_files": len(masks),
        "missing_pairs": 0,
        "duplicate_ids": 0,
        "exact_duplicate_images": image_duplicates,
        "exact_duplicate_masks": mask_duplicates,
        "exact_duplicate_complete_pairs": pair_duplicates,
        "counts": actual,
        "sequence_handling": "No sequence IDs or grouping metadata found in the flat numbered source folders; split by pair ID, grouping only exact duplicate images.",
    }
    (STAGING / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    STAGING.rename(DESTINATION)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Output: {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
