import tensorflow as tf


def trloss(y_true, y_pred):
    w = 0.8
    epsilon = 1.0e-6
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    const_beta = tf.constant(w)
    const_1 = tf.constant(1.0)
    zero = tf.constant([0.0, 1.0])
    exp1_1 = tf.subtract(y_true, const_beta)
    exp1 = tf.multiply(exp1_1, zero)
    exp2 = tf.log(tf.subtract(const_1, y_pred))
    exp3 = tf.multiply(exp1, exp2)
    exp4 = tf.reduce_sum(exp3, axis=-1)
    return exp4