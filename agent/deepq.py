import numpy as np
import tensorflow as tf
import time

from .models import DuelingQ
from .base import BaseAgent
from .history import History

class DeepQAgent(BaseAgent):
    def __init__(self, board, n_moves=100, batch_size=32, memory=1024,
        sample_mode='e_greedy', reward_type='combo'):

        super(DeepQAgent, self).__init__(board, n_moves, reward_type)
        ## Some hyper params
        self.name = 'DeepQAgent'
        self.sample_mode = sample_mode
        self.learning_rate = 1e-6
        self.epsilon = 0.99
        self.gamma = 0.9
        self.history = History(capacity=memory)
        self.batch_size = batch_size
        self.global_step = 0

        print('Setting up Tensorflow')
        # self.targetQ = models.DuelingQ(self.board, self.n_actions)
        self.onlineQ = DuelingQ(self.board, self.n_actions, self.learning_rate)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # self._setup_network_v1()
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.onlineQ.init_op)
        print('Agent ready')


    def train(self, update_iter=5, verbose=False):
        n_moves = self.n_moves

        self.board.refresh_orbs()
        # self.board.select_orb([3,4]) ## Somewhere in the middle
        self.board._select_random()

        move = 0
        total_reward = 0
        previous_combo = 0
        loss = None
        while self.board.selected is not None and move <= n_moves:
            ## Feed history new observations
            move += 1
            self.global_step += 1

            ## Make a move
            s_t = self.board.board_2_state()
            ## Implement the exploration policy
            if np.random.rand(1)[0] < self.epsilon:
                a_t = np.random.multinomial(1, [0.2475]*4+[0.01])
                a_t = np.argmax(a_t)
                # a_t = np.random.choice(range(self.n_actions))
                ## Restrict to legal actions
                # action_space = self.board.selected.get_possible_moves(self.board)
                # a_t_legal = np.random.choice(action_space)
                # a_t = np.random.choice([a_t, a_t_legal])
            else:
                a_t = self._sample_action(s_t, verbose=verbose)


            ## Evaluate reward now, and get the next state
            terminal_move, selected_pos= self.board.move_orb(a_t)
            s_t1 = self.board.board_2_state()
            r_t = self._eval_reward(terminal_move, selected_pos)
            self.history.store_state(s_t, a_t, r_t, s_t1, terminal_move)

            ## Minibatch update:
            # if self.global_step % update_iter == 0:
                # if verbose:
                #     print 'Updating online network'
            s_j, a_j, r_j, s_j1, is_end = self.history.minibatch(self.batch_size)
            feed_dict = {self.onlineQ.state: s_j}
            q_j = self.sess.run(self.onlineQ.Qpred, feed_dict=feed_dict)
            feed_dict = {self.onlineQ.state: s_j1}
            # q_prime = self.sess.run(self.onlineQ.Qpred, feed_dict=feed_dict)

            nextQ = [0]*self.batch_size
            for idx, (r_jx, is_end_j) in enumerate(zip(r_j, is_end)):
                if is_end_j:
                    nextQ[idx] = r_jx
                else:
                    nextQ[idx] = r_jx + self.gamma * q_j[idx, a_j[idx]]

            ## Update
            feed_dict = {self.onlineQ.nextQ: nextQ,
                         self.onlineQ.state: s_j,
                         self.onlineQ.actions: a_j}
            _,loss,delta = self.sess.run([self.onlineQ.optimize_op,
                                          self.onlineQ.loss_op,
                                          self.onlineQ.delta],
                                          feed_dict=feed_dict)

            if verbose:
                print('move {} ep a_t: {} (mode {})'.format(
                    move, a_t, self.sample_mode), end=' ')
                print('loss= {}, r_t = {}'.format(loss, r_t))

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
        while self.board.selected is not None and moves<=self.n_moves:
            time.sleep(0.25)
            moves += 1
            s_t = self.board.board_2_state()
            feed_dict = {self.onlineQ.state: s_t}
            a_t = self.sess.run(self.onlineQ.action_op, feed_dict=feed_dict)

            _ = self.board.move_orb(a_t[0])

        combos, n_cleared = self.eval_outcome()
        return combos, n_cleared, moves
