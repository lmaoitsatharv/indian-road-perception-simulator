"""Separate simulation input pipeline: RGB, binary road target, valid-pixel weights.

The three outputs are (image, mask, valid). Training must use an explicitly
masked BCE + soft Dice objective and masked global metrics. Ordinary unmasked
IDD training is deliberately not exposed here. No training occurs on import.
"""
import csv
from pathlib import Path
import tensorflow as tf

from idd_data import normalize_pair
from idd_experiments import light_augment

IMAGE_SIZE = (512,512)
SEED = 42
BATCH_SIZE = 4


def read_pairs(splits_dir):
    root = Path(splits_dir).expanduser().resolve()
    pairs, ids, groups = {s: [] for s in ("train","val","test")}, set(), {}
    with (root / "pair_splits.csv").open(newline="",encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            pair_id, split, episode = int(row["id"]), row["split"], row["episode_id"]
            if pair_id in ids or pair_id < 0 or split not in pairs:
                raise ValueError("Invalid/repeated simulation pair ID or split")
            if episode in groups and groups[episode] != split:
                raise ValueError("A simulation episode leaked across splits")
            groups[episode] = split
            ids.add(pair_id)
            record = {"id": pair_id, "episode_id": episode}
            for key,folder,prefix in (("image","images","Image"),("mask","masks","Mask"),("valid","valid","Valid")):
                path = root / split / folder / f"{prefix}_{pair_id}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                record[key] = str(path)
            pairs[split].append(record)
    for split, records in pairs.items():
        if not records:
            raise ValueError(f"Empty split: {split}")
        records.sort(key=lambda r:r["id"])
        for key,folder in (("image","images"),("mask","masks"),("valid","valid")):
            if {p.name for p in (root/split/folder).iterdir()} != {Path(r[key]).name for r in records}:
                raise ValueError(f"Extra/missing files in {split}/{folder}")
    return pairs


def decode_pair(image_path, mask_path, valid_path):
    image = tf.ensure_shape(tf.io.decode_png(tf.io.read_file(image_path),channels=3), [512,512,3])
    mask = tf.ensure_shape(tf.io.decode_png(tf.io.read_file(mask_path),channels=1), [512,512,1])
    valid = tf.ensure_shape(tf.io.decode_png(tf.io.read_file(valid_path),channels=1), [512,512,1])
    checks = [tf.debugging.assert_less_equal(tf.reduce_max(mask),tf.constant(1,tf.uint8)),
              tf.debugging.assert_less_equal(tf.reduce_max(valid),tf.constant(1,tf.uint8)),
              tf.debugging.assert_positive(tf.reduce_sum(tf.cast(valid,tf.int32))),
              tf.debugging.assert_equal(mask*(1-valid),tf.zeros_like(mask))]
    with tf.control_dependencies(checks):
        return tf.cast(image,tf.float32),tf.cast(mask,tf.float32),tf.cast(valid,tf.float32)


def make_dataset(records, *, training=False, batch_size=BATCH_SIZE, seed=SEED, profile="none"):
    if not records or batch_size < 1 or profile not in ("none","shadow"):
        raise ValueError("Invalid records, batch size or augmentation profile")
    paths = tf.data.Dataset.from_tensor_slices(tuple([r[k] for r in records] for k in ("image","mask","valid")))
    if training:
        paths = paths.shuffle(len(records),seed=seed,reshuffle_each_iteration=True)
        seeds = tf.data.Dataset.random(seed=seed,rerandomize_each_iteration=True)
        def training_load(pair, random_seed):
            image,mask,valid = decode_pair(*pair)
            paired = tf.concat([mask,valid],axis=-1)
            raw_seed = tf.stack([tf.cast(seed,tf.int32),tf.cast(random_seed,tf.int32)])
            image,paired = light_augment(image,paired,raw_seed,profile)
            image,mask = normalize_pair(image,paired[...,:1])
            return image,mask,paired[...,1:]
        dataset = tf.data.Dataset.zip((paths,seeds)).map(training_load,num_parallel_calls=4,deterministic=True)
    else:
        def evaluation_load(*pair):
            image,mask,valid = decode_pair(*pair)
            image,mask = normalize_pair(image,mask)
            return image,mask,valid
        dataset = paths.map(evaluation_load,num_parallel_calls=4,deterministic=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options).batch(batch_size,drop_remainder=False).prefetch(1)
