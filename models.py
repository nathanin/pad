import tensorflow as tf


def weight_variable(shape, name=None):
    fill = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(fill, name=name)

def bias_variable(shape, name=None):
    fill = tf.constant(0., shape=shape)
    return tf.Variable(fill, name=name)

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

## https://gist.github.com/awjuliani/fffe41519166ee41a6bd5f5ce8ae2630
def updateTargetGraph(tfVars,tau):
    for idx,var in enumerate(tfVars):
        print idx,var

    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

"""
"""
class DuelingQ(object):
    def __init__(self, board, action_dim, learning_rate):
        board_row, board_col = board.shape

        self.weights = {'input': weight_variable([3, 3, 2, 32]),
                   'hidden1': weight_variable([1, 1, 32, 32]),
                   'hidden2': weight_variable([2, 2, 32, 32]),
                   'advantage': weight_variable([32, action_dim]),
                #    'advantage2': weight_variable([128, action_dim]),
                   'value': weight_variable([32, 1]),}
                #    'value2': weight_variable([128, 1]),}
                #    'output': weight_variable([512, action_dim])}

        self.biases = {'input': bias_variable([32]),
                       'hidden1': bias_variable([32]),
                       'hidden2': bias_variable([32]),
                       'advantage': bias_variable([action_dim]),
                       'value': bias_variable([1]), }

        self.state = tf.placeholder('float', [None, board_row, board_col, 2])
        print 'state: ', self.state.get_shape()
        self.keep_prob = tf.placeholder('float', None, name='keep_prob')
        print 'keep_prob: ', self.keep_prob.get_shape()

        ## Net definition
        net = tf.nn.conv2d(
            self.state, self.weights['input'],
            strides=[1, 2, 2, 1],
            padding='VALID')
        # net = tf.nn.bias_add(net, self.biases['input'])
        net = tf.nn.relu(net)
        print 'net input: ', net.get_shape()

        net = tf.nn.conv2d(
            net, self.weights['hidden1'],
            strides=[1, 1, 1, 1],
            padding='VALID')
        # net = tf.nn.bias_add(net, self.biases['hidden1'])
        net = tf.nn.relu(net)
        print 'net hidden1: ', net.get_shape()

        net = tf.nn.conv2d(
            net, self.weights['hidden2'],
            strides=[1, 1, 1, 1],
            padding='VALID')
        # net = tf.nn.bias_add(net, self.biases['hidden2'])
        net = tf.nn.relu(net)
        print 'net hidden2: ', net.get_shape()

        net = tf.contrib.layers.flatten(net)
        net = tf.nn.dropout(net, self.keep_prob)

        ## Split advantage and value functions:
        # adv_split, val_split = tf.split(net, 2, 1)
        adv_split = net
        val_split = net

        # adv_split = tf.contrib.layers.flatten(adv_split)
        # val_split = tf.contrib.layers.flatten(val_split)
        print 'adv_split flat: ', adv_split.get_shape()
        print 'val_split flat: ', val_split.get_shape()

        ## Dropout to simulate a Bayesian process
        # adv_split = tf.nn.dropout(adv_split, self.keep_prob)
        # val_split = tf.nn.dropout(val_split, self.keep_prob)

        adv_split = tf.matmul(adv_split, self.weights['advantage'])
        print 'adv: ', adv_split.get_shape()

        val_split = tf.matmul(val_split, self.weights['value'])
        print 'value: ', val_split.get_shape()

        adv_mean = tf.reduce_mean(adv_split, 1, keep_dims=True)
        print 'adv_mean: ', adv_mean.get_shape()

        ## Single stream
        # self.Qpred = tf.matmul(net, weights['output'])# + biases['output']
        ## Double stream
        self.Qpred = val_split + (adv_split - adv_mean)
        print 'net output: ', self.Qpred.get_shape()


        ## For predicting
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        print 'actions: ', self.actions.get_shape()
        self.actions_onehot = tf.one_hot(self.actions,action_dim,dtype=tf.float32)
        # print 'actions_onehot: ', self.actions_onehot.get_shape()

        ## Predicted Q, with r_t added
        # self.nextQ = tf.placeholder('float', [None,action_dim], name='nextQ')
        self.nextQ = tf.placeholder('float', [None], name='nextQ')
        print 'nextQ: ', self.nextQ.get_shape()

        ## Zero the others
        self.Q = tf.reduce_sum(self.Qpred * self.actions_onehot, 1)
        print 'Q: ', self.Q.get_shape()
        self.delta = tf.square(self.nextQ - self.Q)
        print 'delta: ', self.delta.get_shape()

        ## endpoint operations
        self.action_op = tf.argmax(self.Qpred, axis=1)
        print 'action: ', self.action_op.get_shape()
        # self.loss_op = tf.reduce_sum(self.delta)
        # self.loss_op = tf.reduce_mean(clipped_error(self.delta))
        self.loss_op = tf.reduce_sum(clipped_error(self.delta))
        print 'loss: ', self.loss_op.get_shape()

        self.optimize_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
        # self.optimize_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()
