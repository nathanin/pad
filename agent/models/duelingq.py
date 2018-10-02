import tensorflow as tf
from .ops import *

class DuelingQ(tf.keras.Model):
    """ DuelingQ Agent

    input observations from environment --> learn a function that outputs
    a decision --> updates from reward signal

    DuelingQ splits action and value branches at the end. (?? ref ??)

    """
    def __init__(self, action_dim, name='DuelingQ'):
        super(DuelingQ, self).__init__()

        nonlin = tf.nn.relu
        kernels = [32, 64, 128]
        adv_hidden = 256
        val_hidden = 256
        self.action_dim = action_dim
        self.gamma = 0.8

        conv_args = {
            'kernel_size': 2,
            'strides': 1, 
            'activation': nonlin,
            'padding': 'valid',
        }
        self.conv0 = tf.layers.Conv2D(kernels[0], **conv_args)
        self.conv1 = tf.layers.Conv2D(kernels[1], **conv_args)
        self.conv2 = tf.layers.Conv2D(kernels[2], **conv_args)

        self.adv_value = tf.layers.Dense(adv_hidden+val_hidden,
            activation=nonlin)

        self.adv_arm_0 = tf.layers.Dense(adv_hidden, activation=nonlin)
        self.adv_arm_1 = tf.layers.Dense(adv_hidden, activation=nonlin)
        self.adv_pred = tf.layers.Dense(action_dim, activation=nonlin)

        self.val_arm_0 = tf.layers.Dense(val_hidden, activation=nonlin)
        self.val_arm_1 = tf.layers.Dense(val_hidden, activation=nonlin)
        self.val_pred = tf.layers.Dense(action_dim, activation=nonlin)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)

    def predictQ(self, state):
        state = tf.constant(state)
        net = self.conv0(state)
        net = self.conv1(net)
        net = self.conv2(net)
        net = tf.layers.flatten(net)
        net = self.adv_value(net)

        adv, value = tf.split(net, 2, 1)
        adv = self.adv_arm_0(adv)
        adv = self.adv_arm_1(adv)
        adv = self.adv_pred(adv)

        value = self.val_arm_0(value)
        value = self.val_arm_1(value)
        value = self.val_pred(value)

        adv_mean = tf.reduce_mean(adv, 1, keep_dims=True)
        predQ = value + (adv - adv_mean)
        return predQ

    def call(self, state):
        predQ = self.predictQ(state)
        action = tf.argmax(predQ, axis=1)
        return action, predQ

    # Q-Learning update function (Minh, et al. )
    def q_train(self, s_j, a_j, r_j, s_j1, is_end):
        with tf.GradientTape() as tape:
            q_j = self.predictQ(s_j)
            q_prime = tf.cast(self.predictQ(s_j1), tf.float32)

            # Augment target according to the reward singal
            nextQ = q_j.numpy()
            for idx, (a_jx, r_jx, is_end_j) in enumerate(zip(a_j, r_j, is_end)):
                if is_end_j:
                    nextQ[idx, a_jx] = r_jx
                else:
                    nextQ[idx, a_jx] = r_jx + self.gamma * np.max(q_prime[idx,:])
            
            # nextQ = tf.constant(nextQ, dtype=tf.float32)
            loss = self.loss_fn(nextQ, q_prime, a_j)
        grad = tape.gradient(loss, self.variables)

        # print('q_train gradients')
        # for gr, v in zip(grad, self.variables):
        #     print(v.name, gr.shape)

        self.optimizer.apply_gradients(zip(grad, self.variables))
        return loss
    
    def loss_fn(self, nextQ, predQ, actions):
        action_onehot = tf.one_hot(actions, self.action_dim,
            dtype=tf.float32)
        Q = tf.reduce_sum(predQ * action_onehot, 1)
        nextQ = tf.reduce_sum(nextQ * action_onehot, 1)
        # print('loss', nextQ.shape, Q.shape)
        # print(nextQ)
        # print(Q)
        loss = tf.losses.mean_squared_error(labels=nextQ, 
                                            predictions=Q)
        return loss

    """
        with tf.variable_scope(name) as scope:
            ## input is number of orb types + 1 for selected
            self.state = tf.placeholder('float', [None, board_row, board_col, 7], name='state')
            print('state: ', self.state.get_shape())
            self.keep_prob = tf.placeholder_with_default(0.75, None, name='keep_prob')
            print('keep_prob: ', self.keep_prob.get_shape())

            ## Net definition
            net = nonlin(conv(self.state, kernels[0], ksize=3, stride=2, var_scope='h1'))
            print('h1: ', net.get_shape())
            net = nonlin(conv(net, kernels[1], ksize=2, stride=1, var_scope='h2'))
            print('h2: ', net.get_shape())
            net = nonlin(conv(net, kernels[2], ksize=2, stride=1, var_scope='h3'))
            print('h3: ', net.get_shape())
            net = nonlin(conv(net, kernels[3], ksize=2, stride=1, var_scope='h4'))
            print('h4: ', net.get_shape())

            net = tf.layers.flatten(net, name='flatten')
            print('flatten: ', net.get_shape())
            # net = tf.nn.dropout(net, self.keep_prob)

            ## Split advantage and value functions:
            net = nonlin(linear(net, 2048, var_scope='fork'))
            adv_split, val_split = tf.split(net, 2, 1)
            # adv_split = tf.identity(net)
            # val_split = tf.identity(net)

            print('adv_split flat: ', adv_split.get_shape())
            print('val_split flat: ', val_split.get_shape())

            adv_split = nonlin(linear(adv_split, adv_hidden, var_scope='adv'))
            adv_split = nonlin(linear(adv_split, action_dim, var_scope='adv_action'))
            print('adv: ', adv_split.get_shape())

            val_split = nonlin(linear(val_split, val_hidden, var_scope='val'))
            val_split = nonlin(linear(val_split, action_dim, var_scope='val_action'))
            print('value: ', val_split.get_shape())

            adv_mean = tf.reduce_mean(adv_split, 1, keep_dims=True)
            print('adv_mean: ', adv_mean.get_shape())

            ## Double stream -- put them back together
            self.Qpred = val_split + (adv_split - adv_mean)
            print('net output: ', self.Qpred.get_shape())

            ## For predicting
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32, name='actions')
            print('actions: ', self.actions.get_shape())
            self.actions_onehot = tf.one_hot(self.actions, action_dim, 
                dtype=tf.float32)

            ## Predicted Q, with r_t added
            # self.nextQ = tf.placeholder('float', [None,action_dim], name='nextQ')
            self.nextQ = tf.placeholder('float', [None, 5], name='nextQ')
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
    """