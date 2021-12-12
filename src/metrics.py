import numpy as np
import tensorflow as tf

from typing import List, Dict

from .constants import ConfigKeys as ck


class MyIoU(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None):
        super(MyIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = 2

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(2, 2),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
        Returns:
          Update op.
        """

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)
        # val = tf.math.divide_no_nan(tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

        # tps, fps, fns = 0, 0, 0
        upp = 0
        downn = 0
        for threshold in np.arange(0.5, 1.0, 0.05):
            cur_iou = tf.cast(tf.greater(iou, threshold), dtype=tf.int16)
            upp += tf.reduce_sum(cur_iou, name='mean_iou')
            downn += len(cur_iou)
        # tps, fps, fns = 0, 0, 0
        # for threshold in np.arange(0.5, 1.0, 0.05):
        #     matches = tf.dtypes.cast(self.total_cm > threshold, dtype=tf.int16)
        #     true_positives = tf.math.reduce_sum(tf.dtypes.cast(matches >= 1, dtype=tf.int16), axis=1)  # Correct objects
        #     false_negatives = tf.math.reduce_sum(tf.dtypes.cast(matches == 0, dtype=tf.int16), axis=1)  # Missed objects
        #     false_positives = tf.math.reduce_sum(tf.dtypes.cast(matches == 0, dtype=tf.int16), axis=0)  # Extra objects
        #     tp, fp, fn = (
        #         tf.math.reduce_sum(true_positives),
        #         tf.math.reduce_sum(false_positives),
        #         tf.math.reduce_sum(false_negatives),
        #     )
        #     tps += tp
        #     fps += fp
        #     fns += fn
        # p = tps / (tps + fps + fns)
        # return tf.math.reduce_mean(p)
        return upp / downn

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MyIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


METRICS = {
    "MyIoU": MyIoU,
    "MeanIoU": tf.keras.metrics.MeanIoU(2)
}


def get_metrics(cnf: Dict) -> List:
    return [METRICS[metric] for metric in cnf[ck.METRICS]]
