"""Check train/validation preprocessing in a new folder; test files stay unread."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from idd_data import BATCH_SIZE, IMAGE_SIZE, SEED, augment_pair, make_dataset, read_pairs

ROOT = Path(__file__).resolve().parent
PREVIEW_SPLITS = ("train", "val")


def save_preview(batches, path):
    tile, header, footer = 256, 38, 24
    panel_w, panel_h = tile * 3, tile + header + footer
    canvas = Image.new("RGB", (panel_w * 2, panel_h * len(batches)), "#111827")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for row, split in enumerate(batches):
        images, masks = batches[split]
        for column in range(2):
            x, y = column * panel_w, row * panel_h
            rgb = np.rint((images[column] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            road = masks[column, ..., 0] == 1
            overlay = rgb.copy()
            overlay[road] = (rgb[road] * 0.55 + np.array([40, 240, 110]) * 0.45).astype(np.uint8)
            title = f"{split.upper()} | batch item {column + 1} | " + ("train flips enabled" if split == "train" else "no augmentation")
            draw.text((x + 8, y + 10), title, fill="white", font=font)
            views = [Image.fromarray(rgb), Image.fromarray(road.astype(np.uint8) * 255).convert("RGB"), Image.fromarray(overlay)]
            for i, (view, label) in enumerate(zip(views, ["Image: normalization reversed for display", "Mask: white = 1", "Road overlay"])):
                resampling = Image.Resampling.NEAREST if i == 1 else Image.Resampling.BILINEAR
                canvas.paste(view.resize((tile, tile), resampling), (x + i * tile, y + header))
                # Short labels keep the preview readable at 256 pixels per panel.
                label = ["Image (display values)", "Mask: white = 1", "Road overlay"][i]
                draw.text((x + i * tile + 8, y + header + tile + 3), label, fill="#cbd5e1", font=font)
    canvas.save(path)


def main(output_dir=None):
    root = ROOT
    output = (Path(output_dir).expanduser().resolve() if output_dir is not None else
              root / "runs" / ("preprocessing_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    if output.exists():
        raise FileExistsError("Choose a new preprocessing output folder; previous reports and previews are preserved.")
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(SEED)
    pairs = read_pairs(root / "dataset" / "splits")
    counts = {split: len(records) for split, records in pairs.items()}
    # Construct and iterate train/validation pipelines only. Test counts above
    # come from manifest/file names, without reading test image or mask bytes.
    datasets = {split: make_dataset(pairs[split], training=split == "train", batch_size=BATCH_SIZE, seed=SEED)
                for split in PREVIEW_SPLITS}
    assert counts == {"train": 5507, "val": 687, "test": 689}, counts
    summaries, batches = {}, {}
    for split, dataset in datasets.items():
        images, masks = next(iter(dataset))
        image_array, mask_array = images.numpy(), masks.numpy()
        assert images.shape == (BATCH_SIZE, *IMAGE_SIZE, 3)
        assert masks.shape == (BATCH_SIZE, *IMAGE_SIZE, 1)
        assert images.dtype == masks.dtype == tf.float32
        assert np.isfinite(image_array).all() and np.isfinite(mask_array).all()
        assert image_array.min() >= -1 and image_array.max() <= 1
        assert set(np.unique(mask_array).tolist()) == {0.0, 1.0}
        assert np.all(mask_array.sum(axis=(1, 2, 3)) > 0)
        cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
        assert cardinality == math.ceil(counts[split] / BATCH_SIZE)
        if split != "train":
            # Independent Pillow read verifies TensorFlow's raw labels and scaling.
            for index, record in enumerate(pairs[split][:BATCH_SIZE]):
                with Image.open(record["image"]) as source:
                    expected_image = np.asarray(source, dtype=np.float32) / 127.5 - 1
                with Image.open(record["mask"]) as source:
                    expected_mask = np.asarray(source, dtype=np.float32)[..., None]
                np.testing.assert_allclose(image_array[index], expected_image, atol=1e-6)
                np.testing.assert_array_equal(mask_array[index], expected_mask)
            images_again, masks_again = next(iter(dataset))
            np.testing.assert_array_equal(image_array, images_again.numpy())
            np.testing.assert_array_equal(mask_array, masks_again.numpy())
        batches[split] = (image_array, mask_array)
        summaries[split] = {
            "pairs": counts[split], "batches": cardinality,
            "image_shape": list(images.shape), "mask_shape": list(masks.shape),
            "dtype": "float32", "image_min": float(image_array.min()),
            "image_max": float(image_array.max()), "mask_values": np.unique(mask_array).tolist(),
        }
        print(f"{split}: {counts[split]} pairs; {cardinality} batches; tensor checks passed", flush=True)

    # New loaders initialized with the same seed produce the same first batch.
    train_again = next(iter(make_dataset(pairs["train"], training=True)))
    np.testing.assert_array_equal(batches["train"][0], train_again[0].numpy())
    np.testing.assert_array_equal(batches["train"][1], train_again[1].numpy())
    train_next_pass = next(iter(datasets["train"]))
    assert not np.array_equal(batches["train"][0], train_next_pass[0].numpy())

    # A deliberately asymmetric synthetic pair verifies registration through flips.
    marker = np.zeros((*IMAGE_SIZE, 1), dtype=np.float32)
    marker[17:223, 11:109, :] = 1
    marker_image = np.repeat(marker * 255, 3, axis=-1)
    observed = set()
    for index in range(32):
        aug_image, aug_mask = augment_pair(tf.constant(marker_image), tf.constant(marker), [SEED, index])
        np.testing.assert_array_equal(aug_image.numpy()[..., 0:1] / 255, aug_mask.numpy())
        if np.array_equal(aug_mask.numpy(), marker):
            observed.add("unchanged")
        else:
            np.testing.assert_array_equal(aug_mask.numpy(), marker[:, ::-1, :])
            observed.add("flipped")
        if len(observed) == 2:
            break
    assert observed == {"unchanged", "flipped"}

    # A tiny slice checks that batching doesn't silently drop the final sample.
    tail = list(make_dataset(pairs["val"][:5]).as_numpy_iterator())
    assert [len(images) for images, _ in tail] == [4, 1]
    save_preview(batches, output / "batch_preview.png")
    report = {
        "status": "passed", "tensorflow_version": tf.__version__, "seed": SEED,
        "batch_size": BATCH_SIZE, "splits": summaries,
        "checks": {
            "manifest_matches_all_folders": True,
            "one_real_batch_per_preview_split": True,
            "validation_matches_original_pixels": True,
            "validation_repeats_identically": True,
            "train_first_batch_reproducible_from_seed": True,
            "train_reshuffles_on_next_pass": True,
            "paired_horizontal_flip_alignment": True,
            "final_partial_batch_retained": True,
        },
        "scope": "Actual train/validation preprocessing and batching; test metadata/file names only; no model created or trained.",
        "split_counts": counts,
        "test_image_contents_read": False,
        "preview_splits": list(PREVIEW_SPLITS),
        "preprocessing": "Image float32 scaled to [-1,1] with MobileNetV2 preprocess_input. Mask float32 remains 0/1. No resizing; original 512x512 resolution.",
        "augmentation": "Train-only shared horizontal flip with probability 0.5. No augmentation in val/test.",
        "data_limitations": "Original sequence IDs are unavailable. The 110 all-zero-mask pairs were removed at the user's request; results describe the retained dataset.",
    }
    (output / "batch_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"All preprocessing checks passed. Report and preview: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New folder; default is a unique runs/preprocessing_check timestamp.")
    main(parser.parse_args().output_dir)
