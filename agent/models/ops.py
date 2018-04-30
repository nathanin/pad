import tensorflow as tf
import numpy as np


def weight_variable(shape, name='weight',
    initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
    return tf.get_variable(name, shape=shape, initializer=initializer)

def bias_variable(shape, name='bias'):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.0))

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv(features, kernel, ksize=4, stride=2, pad='SAME', var_scope='conv', dilation=None):
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight_shape = [ksize, ksize, dim_in, kernel]
        weight = weight_variable(weight_shape, name='w')
        ## WHYY
        if dilation is not None:
            assert stride==1
            dilation = [dilation, dilation]

        out = tf.nn.convolution(features, weight, strides=[stride, stride],
            padding=pad, dilation_rate=dilation)

        oH, oW, oC = out.get_shape().as_list()[1:]
        bias = bias_variable([kernel], name='b')
        out = tf.reshape(tf.nn.bias_add(out, bias), [-1, oH, oW, oC])
        return out

LINEAR_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.01)
def linear(features, n_output, var_scope='linear', initializer=LINEAR_INIT):
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight_shape = [dim_in, n_output]
        weight = weight_variable(weight_shape, name='w', initializer=initializer)
        bias = bias_variable(n_output, name='b')
        return bias + tf.matmul(features, weight)


## https://gist.github.com/awjuliani/fffe41519166ee41a6bd5f5ce8ae2630
def updateTargetGraph(tfVars, tau):
    for idx,var in enumerate(tfVars):
        print(idx,var)

    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
