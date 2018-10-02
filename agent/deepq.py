import numpy as np
import tensorflow as tf
import time
from .models import DuelingQ
from .base import BaseAgent
from .history import History

class DeepQAgent(BaseAgent):
    def __init__(self, board, n_moves=100, batch_size=32, memory=1024,
        target=4, sample_mode='e_greedy', reward_type='combo'):

        super(DeepQAgent, self).__init__(board, n_moves, reward_type)
        ## Some hyper params
        self.name = 'DeepQAgent'
        self.sample_mode = sample_mode
        self.learning_rate = 0.00001
        self.epsilon = 0.9
        self.gamma = 0.8
        self.history = History(capacity=memory)
        self.batch_size = batch_size
        self.global_step = 0

        # Number of combos to accomplish 
        self.target = target

        print('Setting up Tensorflow')
        self.onlineQ = DuelingQ(self.n_actions)
        # self.config = tf.ConfigProto()
        # self.config.gpu_options.allow_growth = True

        # # self._setup_network_v1()
        # self.sess = tf.Session(config=self.config)
        # self.sess.run(self.onlineQ.init_op)
        print('Observing behavior')
        self.observe()

        print('Agent ready')

    # ! blow this up
    # def train(self, update_iter=5, verbose=False):
    #     pass

    def train(self, update_iter=5, verbose=False):
        n_moves = self.n_moves

        self.board.refresh_orbs(seed=None)
        # self.board.select_orb([3,4]) ## Somewhere in the middle
        self.board._select_random(seed=None)

        move = 0
        total_reward = 0

        # Reset previous number of matches register.
        self.prev_matches = 0
        while self.board.selected is not None and move <= n_moves:
            ## Feed history new observations
            move += 1
            self.global_step += 1

            # s_t = self.board.board_2_state()
            # feed_dict = {self.onlineQ.state: s_t}
            # a_t, Qpred = self.sess.run([self.onlineQ.action_op, 
            #     self.onlineQ.Qpred],
            #     feed_dict=feed_dict)
            
            ## Observe the models' behavior
            ## Get action and predicted q value from state
            s_t = self.board.board_2_state()
            a_t, predQ = self.onlineQ(s_t)

            ## Do the predicted action, or a random action
            if np.random.rand(1)[0] < self.epsilon:
                a_t = np.random.choice(range(self.n_actions))
                ## Restrict to legal actions
                action_space = self.board.selected.get_possible_moves(self.board)
                a_t_legal = np.random.choice(action_space)
                a_t = np.random.choice([a_t, a_t_legal])
            else:
                a_t = a_t[0]

            terminal_move, selected_position = self.board.move_orb(a_t)
            s_t1 = self.board.board_2_state()
            r_t = self._eval_reward(terminal_move, selected_position)

            ## Add the new (state, action, reward, new-state) tuple to history
            self.history.store_state(s_t, a_t, r_t, s_t1, terminal_move)

            ## Minibatch update:
            ## Pull a (state, action, reward, ...) from history
            s_j, a_j, r_j, s_j1, is_end = self.history.minibatch(
                self.batch_size)
            loss = self.onlineQ.q_train(s_j, a_j, r_j, s_j1, is_end)

            # feed_dict = {self.onlineQ.state: s_j}
            # q_j = self.sess.run(self.onlineQ.Qpred, feed_dict=feed_dict)
            # q_j = self.onlineQ.predictQ(s_j)
            # feed_dict = {self.onlineQ.state: s_j1}
            # q_prime = self.sess.run(self.onlineQ.Qpred, 
            #     feed_dict=feed_dict)
            # q_prime = self.onlineQ.predictQ(s_j1)

            # nextQ = q_j
            # for idx, (a_jx, r_jx, is_end_j) in enumerate(zip(a_j, r_j, is_end)):
            #     if is_end_j:
            #         nextQ[idx, a_jx] = r_jx
            #     else:
            #         nextQ[idx, a_jx] = r_jx + self.gamma * np.max(q_prime[idx,:])

            ## Update
            # self.onlineQ.update_fn(nextQ, predQ, a_j)
            # feed_dict = {self.onlineQ.nextQ: nextQ, self.onlineQ.state: s_j,
            #              self.onlineQ.actions: a_j}
            # _,loss,delta = self.sess.run([self.onlineQ.optimize_op,
            #                               self.onlineQ.loss_op,
            #                               self.onlineQ.delta],
            #                               feed_dict=feed_dict)

            if verbose:
                print('move {} ep a_t: {} (mode {})'.format(
                    move, a_t, self.sample_mode), end=' ')
                print('loss = {:3.3f} r_t = {}'.format(loss, r_t))

            if terminal_move:
                break

        return r_t, move

    def play(self):
        moves = 0
        self.board.refresh_orbs()
        self.board._select_random()
        terminal_move = False
        while not terminal_move:
            time.sleep(self.board.sleep_time)
            moves += 1
            s_t = self.board.board_2_state()
            a_t, Qpred = self.onlineQ(s_t)
            print('Moving.. {}'.format(a_t))

            terminal_move, newpos = self.board.move_orb(a_t[0])

        combos, _ = self.eval_outcome()
        return combos, moves
