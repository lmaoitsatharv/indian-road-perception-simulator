"""Training-only boundary emphasis and an explicitly defined outline metric.

The forward network and input loader are unchanged. These helpers derive
boundaries from supplied 0/1 masks; they do not detect painted road lines.
"""
import tensorflow as tf


def inner_boundary(mask, radius=5):
    mask = tf.cast(mask, tf.float32)
    padded = tf.pad(mask, [[0, 0], [radius, radius], [radius, radius], [0, 0]])
    eroded = -tf.nn.max_pool2d(-padded, 2 * radius + 1, 1, "VALID")
    return mask - eroded


@tf.keras.utils.register_keras_serializable(package="IDD")
class BoundaryWeightedBCEDice(tf.keras.losses.Loss):
    """Weighted BCE / sum(weights), plus the existing per-image soft Dice.

    A 7x7 mask morphological gradient gives a 3-pixel band on both sides of
    observed label transitions. Band weight=4; other pixels=1. SAME pooling
    ignores pixels outside the frame, so an artificial frame edge gets no
    extra training weight. alpha=0 is the matched ordinary BCE+Dice control.
    This is a custom weighted loss, not Kervadec's signed-distance loss.
    """
    def __init__(self, alpha=3.0, radius=3, smooth=1e-6,
                 name="boundary_weighted_bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha, self.radius, self.smooth = float(alpha), int(radius), float(smooth)

    def call(self, y_true, y_pred):
        truth, pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        k = 2 * self.radius + 1
        dilated = tf.nn.max_pool2d(truth, k, 1, "SAME")
        eroded = -tf.nn.max_pool2d(-truth, k, 1, "SAME")
        weights = tf.stop_gradient(1.0 + self.alpha * (dilated - eroded))
        bce = tf.keras.losses.binary_crossentropy(truth, pred, from_logits=False)
        weighted_bce = tf.reduce_sum(bce * weights[..., 0], (1, 2)) / tf.reduce_sum(weights, (1, 2, 3))
        intersection = tf.reduce_sum(truth * pred, (1, 2, 3))
        denominator = tf.reduce_sum(truth + pred, (1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return weighted_bce + 1.0 - dice

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "radius": self.radius, "smooth": self.smooth}


@tf.keras.utils.register_keras_serializable(package="IDD")
class BoundaryIoU(tf.keras.metrics.Metric):
    """Global IoU of inner mask boundary bands, default radius=5 pixels.

    Square morphology matches repeated 3x3 erosion in the authors' Boundary
    IoU implementation. The frame is background padded for evaluation.
    This strict 5px radius is predeclared; 14px (2% diagonal) is also reported.
    """
    def __init__(self, radius=5, threshold=0.5, name="boundary_iou_5px", **kwargs):
        super().__init__(name=name, **kwargs)
        self.radius, self.threshold = int(radius), float(threshold)
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype="float64")
        self.union = self.add_weight(name="union", initializer="zeros", dtype="float64")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("This global metric expects unweighted validation pixels")
        truth = inner_boundary(tf.cast(y_true >= .5, tf.float32), self.radius) > .5
        pred = inner_boundary(tf.cast(y_pred >= self.threshold, tf.float32), self.radius) > .5
        self.intersection.assign_add(tf.reduce_sum(tf.cast(truth & pred, tf.float64)))
        self.union.assign_add(tf.reduce_sum(tf.cast(truth | pred, tf.float64)))

    def result(self):
        return tf.where(self.union > 0, tf.math.divide_no_nan(self.intersection, self.union), tf.constant(1., tf.float64))

    def reset_state(self):
        self.intersection.assign(0.)
        self.union.assign(0.)

    def get_config(self):
        return {**super().get_config(), "radius": self.radius, "threshold": self.threshold}
