"""Simulation-only objective: packed targets contain [road, valid].

The network retains one probability output. Validity is training metadata, never
an input feature. Both BCE and soft Dice exclude every uncertain pixel.
"""
import tensorflow as tf

from idd_model import RoadDice, RoadIoU


@tf.keras.utils.register_keras_serializable(package="Simulation")
class MaskedBCEDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="masked_bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)

    def call(self, y_true, y_pred):
        truth = tf.cast(y_true[..., :1], tf.float32)
        valid = tf.cast(y_true[..., 1:2], tf.float32)
        probability = tf.cast(y_pred, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(truth, probability, from_logits=False)
        count = tf.reduce_sum(valid, axis=(1, 2, 3))
        bce = tf.math.divide_no_nan(tf.reduce_sum(bce * valid[..., 0], axis=(1, 2)), count)
        intersection = tf.reduce_sum(valid * truth * probability, axis=(1, 2, 3))
        denominator = tf.reduce_sum(valid * (truth + probability), axis=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return bce + 1.0 - dice

    def get_config(self):
        return {**super().get_config(), "smooth": self.smooth}


def _weights(target, prediction, sample_weight):
    valid = tf.cast(target[..., 1:2], tf.float64)
    if sample_weight is not None:
        extra = tf.ones([tf.rank(prediction) - tf.rank(sample_weight)], tf.int32)
        weight = tf.reshape(sample_weight, tf.concat([tf.shape(sample_weight), extra], axis=0))
        valid *= tf.cast(weight, tf.float64)
    return valid


@tf.keras.utils.register_keras_serializable(package="Simulation")
class MaskedRoadIoU(RoadIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[..., :1], y_pred, _weights(y_true, y_pred, sample_weight))


@tf.keras.utils.register_keras_serializable(package="Simulation")
class MaskedRoadDice(RoadDice):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[..., :1], y_pred, _weights(y_true, y_pred, sample_weight))


def pack_targets(image, road, valid):
    """Adapt the audited triple loader to ordinary serializable Keras training."""
    return image, tf.concat([road, valid], axis=-1)


def compile_simulation_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=MaskedBCEDiceLoss(), metrics=[MaskedRoadIoU(), MaskedRoadDice()],
        jit_compile=False,
    )
    return model
