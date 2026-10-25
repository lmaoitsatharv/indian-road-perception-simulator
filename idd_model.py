"""MobileNetV2 U-Net, probability loss, and globally accumulated road metrics.

Input is already normalized to [-1, 1] by idd_data.py. This network performs
no rescaling, augmentation, or image resizing.
"""

from pathlib import Path

import tensorflow as tf

WEIGHTS_NAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
WEIGHTS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/" + WEIGHTS_NAME
SKIP_NAMES = (
    "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu",
    "block_13_expand_relu", "block_16_project",
)
DECODER_FILTERS = (256, 128, 64, 32)


def get_encoder_weights(cache_dir):
    """Fetch the official ImageNet no-top weights into the project directory."""
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(tf.keras.utils.get_file(
        fname=WEIGHTS_NAME, origin=WEIGHTS_URL,
        cache_dir=str(cache_dir), cache_subdir="weights",
    ))


def build_unet(weights_path):
    """A frozen pretrained encoder and a newly initialized trainable decoder."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained encoder weights not found: {weights_path}")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(512, 512, 3), include_top=False,
        alpha=1.0, weights=str(weights_path),
    )
    encoder = tf.keras.Model(
        base.input, [base.get_layer(name).output for name in SKIP_NAMES],
        name="mobilenetv2_encoder",
    )
    encoder.trainable = False
    inputs = tf.keras.Input((512, 512, 3), name="normalized_rgb")
    # Preserve pretrained BatchNorm moving statistics, including in fine-tuning.
    features = encoder(inputs, training=False)
    x = features[-1]
    for index, (filters, skip) in enumerate(zip(DECODER_FILTERS, reversed(features[:-1])), 1):
        x = tf.keras.layers.Conv2DTranspose(
            filters, 3, strides=2, padding="same", use_bias=False,
            kernel_initializer="he_normal", name=f"decoder_{index}_upsample",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"decoder_{index}_up_bn")(x)
        x = tf.keras.layers.ReLU(name=f"decoder_{index}_up_relu")(x)
        x = tf.keras.layers.Concatenate(name=f"decoder_{index}_skip")([x, skip])
        for conv in (1, 2):
            x = tf.keras.layers.Conv2D(
                filters, 3, padding="same", use_bias=False,
                kernel_initializer="he_normal", name=f"decoder_{index}_conv_{conv}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"decoder_{index}_bn_{conv}")(x)
            x = tf.keras.layers.ReLU(name=f"decoder_{index}_relu_{conv}")(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="full_resolution")(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="full_resolution_refine")(x)
    probabilities = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="road_probability")(x)
    return tf.keras.Model(inputs, probabilities, name="idd_mobilenetv2_unet")


@tf.keras.utils.register_keras_serializable(package="IDD")
class BCEDiceLoss(tf.keras.losses.Loss):
    """Mean per-image BCE plus mean per-image soft Dice loss (equal weights)."""

    def __init__(self, smooth=1e-6, name="bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        bce_per_image = tf.reduce_mean(bce, axis=(1, 2))
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return bce_per_image + 1.0 - dice

    def get_config(self):
        return {**super().get_config(), "smooth": self.smooth}


class _RoadOverlap(tf.keras.metrics.Metric):
    """Accumulate TP/FP/FN over all pixels and batches, then compute overlap."""

    def __init__(self, threshold=0.5, name="road_overlap", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype="float64")
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype="float64")
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype="float64")

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth, prediction = y_true >= 0.5, y_pred >= self.threshold
        weight = tf.constant(1.0, tf.float64)
        if sample_weight is not None:
            weight = tf.cast(sample_weight, tf.float64)
            extra_axes = tf.ones([tf.rank(y_pred) - tf.rank(weight)], dtype=tf.int32)
            weight = tf.reshape(weight, tf.concat([tf.shape(weight), extra_axes], axis=0))
        for variable, pixels in (
            (self.tp, tf.logical_and(truth, prediction)),
            (self.fp, tf.logical_and(tf.logical_not(truth), prediction)),
            (self.fn, tf.logical_and(truth, tf.logical_not(prediction))),
        ):
            variable.assign_add(tf.reduce_sum(tf.cast(pixels, tf.float64) * weight))

    def reset_state(self):
        for variable in (self.tp, self.fp, self.fn):
            variable.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


@tf.keras.utils.register_keras_serializable(package="IDD")
class RoadIoU(_RoadOverlap):
    def __init__(self, threshold=0.5, name="road_iou", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        union = self.tp + self.fp + self.fn
        # If the complete evaluated set has no foreground in truth or prediction,
        # define the perfect empty/empty overlap as 1. No per-image averaging.
        return tf.where(union > 0, tf.math.divide_no_nan(self.tp, union), tf.constant(1.0, tf.float64))


@tf.keras.utils.register_keras_serializable(package="IDD")
class RoadDice(_RoadOverlap):
    def __init__(self, threshold=0.5, name="road_dice", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        denominator = 2.0 * self.tp + self.fp + self.fn
        return tf.where(
            denominator > 0, tf.math.divide_no_nan(2.0 * self.tp, denominator),
            tf.constant(1.0, tf.float64),
        )


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=BCEDiceLoss(), metrics=[RoadIoU(), RoadDice()],
        jit_compile=False,
    )
    return model


def enable_last_encoder_blocks(model):
    """Optional phase two: unfreeze blocks 13-16 while all encoder BN stays frozen."""
    encoder = model.get_layer("mobilenetv2_encoder")
    encoder.trainable = True
    for layer in encoder.layers:
        layer.trainable = (
            layer.name.startswith(tuple(f"block_{i}_" for i in range(13, 17)))
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
    return encoder
