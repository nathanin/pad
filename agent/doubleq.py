import numpy as np
import tensorflow as tf
from .models import DuelingQ
from .base import BaseAgent
from .history import History
from .models.ops import updateTargetGraph, updateTarget

class DoubleQAgent(BaseAgent):
    def __init__(self,
                board,
                n_moves=50,
                batch_size=16,
                memory=2048,
                sample_mode='bayes',
                reward_type='combo'):
        super(DoubleQAgent, self).__init__(board, n_moves, reward_type)
        ## Some hyper params
        self.name = 'Dueling-DoubleQAgent'
        self.sample_mode = sample_mode
        self.learning_rate = 1e-4
        self.n_bootstrap = 50
        self.history = History(capacity=memory)
        self.batch_size = batch_size
        self.global_step = 0
        self.update_iter = 100
        self.tau = 0.01


        self.gamma = 0.75
        if self.sample_mode == 'e_greedy':
            self.epsilon = 0.9
        elif self.sample_mode == 'bayes':
            self.epsilon = 0.15

        print('Setting up Tensorflow')
        self.targetQ = DuelingQ(self.board, self.n_actions,
            self.learning_rate, name='target')
        self.onlineQ = DuelingQ(self.board, self.n_actions,
            self.learning_rate, name='online')
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # self._setup_network_v1()
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.onlineQ.init_op)
        self.sess.run(self.targetQ.init_op)

        trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(trainables, self.tau)
        print(self.targetOps)

        print('Agent ready')


    """ Dueling-DQN training """
    def train(self, verbose=False):
        n_moves = self.n_moves

        self.board.refresh_orbs()
        # self.board._select_random()
        self.board.select_orb([0,0]) ## Somewhere in the middle

        move = 0
        total_reward = 0
        previous_combo = 0
        loss = None
        while self.board.selected is not None:
            ## Feed history new observations
            move += 1
            self.global_step += 1

            ## Sample from targetQ
            s_t = self.board.board_2_state()
            a_t = self._sample_action(s_t, verbose=verbose)

            ## Evaluate reward and next state
            terminal_move, selected_pos= self.board.move_orb(a_t)
            s_t1 = self.board.board_2_state()
            r_t = self._eval_reward(terminal_move, selected_pos)
            # r_t = self._eval_distance_reward(selected_pos, terminal_move)
            self.history.store_state(s_t, a_t, r_t, s_t1, terminal_move)

            ## Minibatch update:
            s_j, a_j, r_j, s_j1, is_end = self.history.minibatch(self.batch_size)

            ## Sample from the full model for updating -- this is like test time
            ## Maybe
            if self.sample_mode == 'e_greedy':
                eps_use = 1.0
            elif self.sample_mode == 'bayes':
                eps_use = self.epsilon

            feed_dict = {self.targetQ.state: s_j1, self.targetQ.keep_prob: eps_use}
            q_j = self.sess.run(self.targetQ.Qpred, feed_dict=feed_dict)
            feed_dict = {self.onlineQ.state: s_j1, self.onlineQ.keep_prob: eps_use}
            a_max = self.sess.run(self.onlineQ.action_op, feed_dict=feed_dict)

            ## nextQ = targetQ(mainQ_max)
            nextQ = [0]*self.batch_size
            for idx, (r_jx, is_end_j) in enumerate(zip(r_j, is_end)):
                if is_end_j:
                    nextQ[idx] = r_jx
                else:
                    nextQ[idx] = r_jx + self.gamma * q_j[idx, a_max[idx]]

            feed_dict = {self.onlineQ.nextQ: nextQ,
                         self.onlineQ.state: s_j,
                         self.onlineQ.actions: a_j,
                         self.onlineQ.keep_prob: eps_use }
            _, loss, delta = self.sess.run([self.onlineQ.optimize_op,
                                          self.onlineQ.loss_op,
                                          self.onlineQ.delta],
                                          feed_dict=feed_dict)

            ## Update targetQ
            if self.global_step % self.update_iter == 0:
                if verbose:
                    print('(updating)',
                updateTarget(self.targetOps, self.sess))


            if terminal_move:
                if verbose:
                    print('moves: {} r_t = {} (lr: {}) (mode: {}) (end-turn {})'.format(
                        move, r_t, self.learning_rate, self.sample_mode, terminal_move))
                break

        return r_t, move


    def play(self):
        moves = 0
        self.board.refresh_orbs()
        self.board._select_random()
        while self.board.selected is not None:
            time.sleep(0.15)
            moves += 1
            s_t = self.board.board_2_state()
            feed_dict = {self.onlineQ.state: s_t, self.onlineQ.keep_prob: 1.0}
            a_t = self.sess.run(self.onlineQ.action_op, feed_dict=feed_dict)

            _ = self.board.move_orb(a_t[0])

        combos, n_cleared = self.eval_outcome()
        return combos, n_cleared, moves
