import tensorflow as tf

from typing import Dict, Callable

from .constants import ConfigKeys as ck


def tversky_loss(beta):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = y_true * y_pred
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

    return loss


def comb_loss():
    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + 1 - numerator / denominator
        return tf.reduce_mean(o)

    return dice_loss


LOSSES = {
    "tversky_loss": tversky_loss,
    "comb_loss": comb_loss,
}


def get_loss(cnf: Dict) -> Callable:
    return LOSSES[cnf[ck.LOSS_FUNCTION]](**cnf[ck.LOSS_PARAMETERS])
