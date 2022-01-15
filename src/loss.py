import tensorflow as tf
from typing import Dict, Callable

# START PROJECT IMPORTS
from .constants import ConfigKeys as ck
# END PROJECT_IMPORTS

def tversky_loss(beta):
    def loss(y_true, y_pred):
        tot = 0
        # print(y_pred.shape)
        for i in range(y_pred.shape[3]):
            y_true_a = tf.cast(y_true[:, :, i], tf.float32)
            y_pred_a = tf.cast(y_pred[:, :, i], tf.float32)

            y_pred_a = tf.math.sigmoid(y_pred_a)
            numerator = y_true_a * y_pred_a
            denominator = y_true_a * y_pred_a + beta * (1 - y_true_a) * y_pred_a + (1 - beta) * y_true_a * (1 - y_pred_a)
            rez = 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
            tot += rez
        return tot

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


def jaccard_loss(smooth=100):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd = (1 - jac) * smooth
        return tf.reduce_mean(jd)
    return loss


LOSSES = {
    "tversky_loss": tversky_loss,
    "comb_loss": comb_loss,
    "jaccard_loss": jaccard_loss,
}


def get_loss(cnf: Dict) -> Callable:
    return LOSSES[cnf[ck.LOSS_FUNCTION]](**cnf[ck.LOSS_PARAMETERS])
