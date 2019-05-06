import tensorflow as tf
import numpy

target = numpy.array(
    [[1, 0], [0, 1], [0, 1]]
)
zero = numpy.array([0, 1])
output = numpy.array(
    [[0.99, 0.99], [0.99, 0.2], [0.1, 0.1]]
)

exp1 = (0.5 - target) * zero
exp2 = numpy.log(1 - output)
exp3 = exp1 * exp2
exp4 = exp3.sum().sum()
print(exp3)
print(exp4)

#assert False
const_05 = tf.constant(0.5)
const_1 = tf.constant(1.0)
target = tf.constant(
    [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
)
zero = tf.constant([0.0, 1.0])
output = tf.constant(
    [[0.99, 0.99], [0.99, 0.2], [0.1, 0.1]]
)
exp1_1 = tf.subtract(const_05, target)
exp1 = tf.multiply(exp1_1, zero)
exp2_1 = tf.subtract(const_1, output)
exp2 = tf.log(exp2_1)
exp3 = tf.multiply(exp1, exp2)
exp4 = tf.reduce_sum(exp3, axis=-1)
with tf.Session() as sess:
    r1 = sess.run(exp1)
    print(r1)
    r2 = sess.run(exp2)
    print(r2)
    r3 = sess.run(exp3)
    print(r3)
    r4 = sess.run(exp4)
    print(r4)
