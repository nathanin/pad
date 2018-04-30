import numpy as np
import tensorflow as tf
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
        self.learning_rate = 0.00001
        self.epsilon = 0.9
        self.gamma = 0.8
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

    # '''
    # Populate a replay history by playing games and storing moves
    # '''
    # def observe(self, pause_time=0, verbose=False):
    #     n_moves = self.n_moves
    #
    #     ## Do many runs to fill the memory
    #     while self.history.n < self.history.capacity:
    #
    #         ## Iterate games
    #         self.board.refresh_orbs()
    #         self.board._select_random()
    #         move = 0
    #         total_reward = 0
    #         previous_combo = 0
    #         while self.board.selected is not None and move<=n_moves:
    #             ## Feed history
    #             move += 1
    #             s_t = self.board.board_2_state()
    #             feed_dict = {self.onlineQ.state: s_t}
    #             a_t, Qpred = self.sess.run([self.onlineQ.action_op, self.onlineQ.Qpred],
    #                 feed_dict=feed_dict)
    #
    #             ## Exploration -- in the beginning it will be very high
    #             if np.random.rand(1)[0] < self.epsilon:
    #                 a_t = np.random.choice(range(self.n_actions))
    #                 ## Restrict to legal actions
    #                 action_space = self.board.selected.get_possible_moves(self.board)
    #                 a_t_legal = np.random.choice(action_space)
    #
    #                 # 1/2 chance to be selected from legal actions only
    #                 a_t = np.random.choice([a_t, a_t_legal])
    #             else:
    #                 a_t = a_t[0]
    #
    #             ## Observe altered state(t+1)
    #             illegal_move = self.board.move_orb(a_t)
    #             s_t1 = self.board.board_2_state()
    #
    #             r_t, previous_combo, do_break = self._eval_reward(
    #                 move, illegal_move, previous_combo)
    #
    #             self.history.store_state(s_t, a_t, r_t, s_t1, do_break)
    #
    #             if do_break:
    #                 break
    #
    #     print('Done observing; Agent memory full')



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

            s_t = self.board.board_2_state()
            feed_dict = {self.onlineQ.state: s_t}
            a_t, Qpred = self.sess.run([self.onlineQ.action_op, self.onlineQ.Qpred],
                feed_dict=feed_dict)

            if np.random.rand(1)[0] < self.epsilon:
                a_t = np.random.choice(range(self.n_actions))
                ## Restrict to legal actions
                action_space = self.board.selected.get_possible_moves(self.board)
                a_t_legal = np.random.choice(action_space)
                a_t = np.random.choice([a_t, a_t_legal])
            else:
                a_t = a_t[0]

            illegal_move = self.board.move_orb(a_t)
            s_t1 = self.board.board_2_state()
            r_t, previous_combo, do_break = self._eval_reward(
                move, illegal_move, previous_combo)

            self.history.store_state(s_t, a_t, r_t, s_t1, do_break)

            ## Minibatch update:
            # if self.global_step % update_iter == 0:
                # if verbose:
                #     print 'Updating online network'
            s_j, a_j, r_j, s_j1, is_end = self.history.minibatch(self.batch_size)
            feed_dict = {self.onlineQ.state: s_j}
            q_j = self.sess.run(self.onlineQ.Qpred, feed_dict=feed_dict)
            feed_dict = {self.onlineQ.state: s_j1}
            q_prime = self.sess.run(self.onlineQ.Qpred, feed_dict=feed_dict)

            nextQ = q_j
            for idx, (a_jx, r_jx, is_end_j) in enumerate(zip(a_j, r_j, is_end)):
                if is_end_j:
                    nextQ[idx, a_jx] = r_jx
                else:
                    nextQ[idx, a_jx] = r_jx + self.gamma * np.max(q_prime[idx,:])

            ## Update
            feed_dict = {self.onlineQ.nextQ: nextQ, self.onlineQ.state: s_j}
            _,loss,delta = self.sess.run([self.onlineQ.optimize_op,
                                          self.onlineQ.loss_op,
                                          self.onlineQ.delta],
                                          feed_dict=feed_dict)

            if verbose:
                print('move {} ep a_t: {} (mode {})'.format(
                    move, a_t, self.sample_mode), end='')
                print('loss= {}, r_t = {}'.format(loss, r_t))

            if do_break:
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
