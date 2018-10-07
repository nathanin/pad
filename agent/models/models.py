import tensorflow as tf
from .ops import *

"""
input observations from environment --> learn a function that outputs
a decision --> updates from reward signal

splits action and value branches at the end. (?? ref ??)

"""
class DuelingQ(object):
    def __init__(self, board, action_dim, learning_rate, name='DuelingQ'):
        board_row, board_col = board.shape
        nonlin = tf.nn.relu
        kernels = [128, 256, 256, 512]

        with tf.variable_scope(name) as scope:
            ## input is number of orb types + 1 for selected
            self.state = tf.placeholder('float', [None, board_row, board_col, 7])
            print('state: ', self.state.get_shape())
            self.keep_prob = tf.placeholder_with_default(0.75, None, name='keep_prob')
            print('keep_prob: ', self.keep_prob.get_shape())

            ## Net definition
            net = nonlin(conv(self.state, kernels[0], ksize=(board_row, board_col), stride=1, var_scope='h1'))
            net = nonlin(conv(net, kernels[0], ksize=(board_row, board_col), stride=1, var_scope='h1_1'))
            print('h1: ', net.get_shape())
            net = nonlin(conv(net, kernels[1], ksize=4, stride=2, var_scope='h2'))
            print('h2: ', net.get_shape())
            net = nonlin(conv(net, kernels[2], ksize=2, stride=2, var_scope='h3'))
            print('h3: ', net.get_shape())
            net = nonlin(conv(net, kernels[3], ksize=2, stride=1, var_scope='h4'))
            print('h4: ', net.get_shape())

            net = tf.layers.flatten(net, name='flatten')
            print('flatten: ', net.get_shape())
            # net = tf.nn.dropout(net, self.keep_prob)

            ## Split advantage and value functions:
            net = nonlin(linear(net, 2048, var_scope='fork'))
            net = nonlin(linear(net, 1024, var_scope='fork_1'))
            net = nonlin(linear(net, 512, var_scope='fork_2'))
            adv_split, val_split = tf.split(net, 2, 1)
            # adv_split = tf.identity(net)
            # val_split = tf.identity(net)

            print('adv_split flat: ', adv_split.get_shape())
            print('val_split flat: ', val_split.get_shape())

            adv_split = nonlin(linear(adv_split, 512, var_scope='adv'))
            adv_split = nonlin(linear(adv_split, action_dim, var_scope='adv_action'))
            print('adv: ', adv_split.get_shape())

            val_split = nonlin(linear(val_split, 512, var_scope='val'))
            val_split = nonlin(linear(val_split, action_dim, var_scope='val_action'))
            print('value: ', val_split.get_shape())

            adv_mean = tf.reduce_mean(adv_split, 1, keepdims=True)
            print('adv_mean: ', adv_mean.get_shape())

            ## Double stream -- put them back together
            self.Qpred = val_split + (adv_split - adv_mean)
            print('net output: ', self.Qpred.get_shape())


            ## For predicting
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            print('actions: ', self.actions.get_shape())
            self.actions_onehot = tf.one_hot(self.actions, action_dim, dtype=tf.float32)

            ## Predicted Q, with r_t added
            # self.nextQ = tf.placeholder('float', [None,action_dim], name='nextQ')
            self.nextQ = tf.placeholder('float', [None], name='nextQ')
            print('nextQ: ', self.nextQ.get_shape())

            ## Zero the others
            self.Q = tf.reduce_sum(self.Qpred * self.actions_onehot, 1)
            print('Q: ', self.Q.get_shape())
            self.delta = tf.square(self.nextQ - self.Q)
            print('delta: ', self.delta.get_shape())

            ## endpoint operations
            self.action_op = tf.argmax(self.Qpred, axis=1)
            print('action: ', self.action_op.get_shape())
            self.loss_op = tf.reduce_sum(clipped_error(self.delta))
            print('loss: ', self.loss_op.get_shape())

            self.optimize_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
            self.init_op = tf.global_variables_initializer()
