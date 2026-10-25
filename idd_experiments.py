"""Optional overnight experiment helpers; originals and the established loader stay intact."""
import tensorflow as tf

from idd_data import decode_pair, augment_pair, normalize_pair
from idd_model import build_unet


def light_augment(image, mask, seed, profile="shadow"):
    """Paired flip plus training RGB changes on raw 0..255 floats, before normalization."""
    # AdjustContrast has no deterministic CUDA implementation in this runtime.
    # Dataset preparation belongs on the CPU; model convolutions stay on the GPU.
    with tf.device("/CPU:0"):
        return _light_augment_cpu(image, mask, seed, profile)


def _light_augment_cpu(image, mask, seed, profile):
    seeds = tf.random.experimental.stateless_split(seed, 7)
    image, mask = augment_pair(image, mask, seeds[0])
    if profile == "none":
        return image, mask
    image = tf.image.stateless_random_brightness(image, 255 * .08, seeds[1])
    image = tf.image.stateless_random_contrast(image, .85, 1.15, seeds[2])
    image = tf.clip_by_value(image, 0., 255.)
    if profile == "shadow":
        # A smooth illumination boundary across the lower image. Labels are unchanged.
        values = tf.random.stateless_uniform([4], seeds[3])
        yy, xx = tf.meshgrid(tf.linspace(0., 1., 512), tf.linspace(0., 1., 512), indexing="ij")
        boundary = .2 + .6 * values[0] + (values[1] - .5) * yy
        shade = tf.sigmoid((boundary - xx) / .035) * tf.sigmoid((yy - .25) / .035)
        strength = .15 + .3 * values[2]
        image = tf.cond(values[3] < .35,
                        lambda: image * (1. - strength * shade[..., None]),
                        lambda: image)
    return tf.clip_by_value(image, 0., 255.), mask


def experiment_dataset(records, *, batch_size, seed, profile):
    paths = tf.data.Dataset.from_tensor_slices(([r["image"] for r in records], [r["mask"] for r in records]))
    paths = paths.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True)
    def load(pair, random_seed):
        image, mask = decode_pair(*pair)
        raw_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(random_seed, tf.int32)])
        image, mask = light_augment(image, mask, raw_seed, profile)
        return normalize_pair(image, mask)
    dataset = tf.data.Dataset.zip((paths, seeds)).map(load, num_parallel_calls=4, deterministic=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options).batch(batch_size, drop_remainder=False).prefetch(2)


def mixed_model_from_checkpoint(root, checkpoint, first_block=13, freeze_decoder_batchnorm=False):
    with tf.device("/CPU:0"):
        saved = tf.keras.models.load_model(checkpoint, compile=False)
        weights = saved.get_weights()
    del saved
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_unet(root / "models/cache/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    config = model.layers[-1].get_config()
    config["dtype"] = "float32"
    head = tf.keras.layers.Conv2D.from_config(config)
    model = tf.keras.Model(model.input, head(model.layers[-2].output), name=model.name)
    model.set_weights(weights)
    encoder = model.get_layer("mobilenetv2_encoder")
    encoder.trainable = first_block is not None
    if first_block is not None:
        for layer in encoder.layers:
            layer.trainable = (layer.name.startswith(tuple(f"block_{i}_" for i in range(first_block, 17)))
                               and not isinstance(layer, tf.keras.layers.BatchNormalization))
    if freeze_decoder_batchnorm:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model
