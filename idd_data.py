"""TensorFlow input pipeline for the cleaned binary IDD road dataset.

Images: float32 [512, 512, 3], MobileNetV2 range [-1, 1].
Masks:  float32 [512, 512, 1], labels stay exactly 0 or 1.
Preprocessing happens in memory; PNG files are never rewritten.
"""

import csv
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (512, 512)
SEED = 42
BATCH_SIZE = 4
SPLITS = ("train", "val", "test")


def read_pairs(splits_dir):
    """Use CSV IDs to pair files; reconstruct paths so the dataset is portable."""
    root = Path(splits_dir).expanduser().resolve()
    pairs = {split: [] for split in SPLITS}
    seen = set()
    with (root / "pair_splits.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("pair_splits.csv must contain id and split columns")
        for row in reader:
            pair_id, split = int(row["id"]), row["split"]
            if pair_id < 0 or pair_id in seen or split not in pairs:
                raise ValueError(f"Invalid or repeated pair: {row['id']}, {split}")
            seen.add(pair_id)
            image_path = root / split / "images" / f"Image_{pair_id}.png"
            mask_path = root / split / "masks" / f"Mask_{pair_id}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing image or mask for pair {pair_id}")
            pairs[split].append({"id": pair_id, "image": str(image_path), "mask": str(mask_path)})

    for split, records in pairs.items():
        records.sort(key=lambda row: row["id"])
        if not records:
            raise ValueError(f"The {split} split is empty")
        for folder, prefix in (("images", "Image"), ("masks", "Mask")):
            actual = {p.name for p in (root / split / folder).iterdir()}
            expected = {f"{prefix}_{row['id']}.png" for row in records}
            if actual != expected:
                raise ValueError(f"{split}/{folder} does not match the CSV manifest")
    return pairs


def decode_pair(image_path, mask_path):
    """Read an RGB image and the original single-channel 0/1 labels."""
    image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    mask = tf.ensure_shape(mask, [*IMAGE_SIZE, 1])
    # Conversion to float preserves 1 as 1; convert_image_dtype would rescale it.
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    checks = [
        tf.debugging.assert_less_equal(tf.reduce_max(mask), 1.0, message="Mask contains labels above 1"),
        tf.debugging.assert_equal(tf.reduce_max(mask), 1.0, message="An empty mask entered the cleaned dataset"),
    ]
    with tf.control_dependencies(checks):
        return tf.identity(image), tf.identity(mask)


def augment_pair(image, mask, seed):
    """One shared random decision flips both tensors horizontally together."""
    flip = tf.random.stateless_uniform([], seed=seed) < 0.5
    return tf.cond(
        flip,
        lambda: (tf.image.flip_left_right(image), tf.image.flip_left_right(mask)),
        lambda: (image, mask),
    )


def normalize_pair(image, mask):
    """MobileNetV2 preprocessing changes images only; labels remain 0/1."""
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), mask


def make_dataset(records, *, training=False, batch_size=BATCH_SIZE, seed=SEED, augment=True):
    """Build finite batches; preserve the final partial batch.

    Training shuffles file paths and uses fresh, reproducible random seeds on
    each pass. Validation/test retain their numeric ID order without augmentation.
    There is no full-dataset image cache, keeping memory usage bounded.
    """
    if batch_size < 1 or not records:
        raise ValueError("batch_size must be positive and records must not be empty")
    paths = tf.data.Dataset.from_tensor_slices((
        [row["image"] for row in records], [row["mask"] for row in records]
    ))
    if training:
        paths = paths.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    if training and augment:
        random_seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True)
        paths = tf.data.Dataset.zip((paths, random_seeds))

        def load_training(pair_paths, random_seed):
            image, mask = decode_pair(*pair_paths)
            flip_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(random_seed, tf.int32)])
            image, mask = augment_pair(image, mask, flip_seed)
            return normalize_pair(image, mask)

        dataset = paths.map(load_training, num_parallel_calls=4, deterministic=True)
    else:
        dataset = paths.map(
            lambda image_path, mask_path: normalize_pair(*decode_pair(image_path, mask_path)),
            num_parallel_calls=4,
            deterministic=True,
        )

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options).batch(batch_size, drop_remainder=False).prefetch(1)


def build_datasets(splits_dir, batch_size=BATCH_SIZE, seed=SEED):
    pairs = read_pairs(splits_dir)
    datasets = {
        split: make_dataset(records, training=split == "train", batch_size=batch_size, seed=seed)
        for split, records in pairs.items()
    }
    return datasets, {split: len(records) for split, records in pairs.items()}
