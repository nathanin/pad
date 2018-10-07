import numpy as np
import tensorflow as tf
import time
from .models import DeepQRNN
from .base import BaseAgent
from .history import History

class DeepQRNN_Agent(BaseAgent):
    def __init__(self, board, n_moves=100, batch_size=32, memory=1024,
        target=4, sample_mode='e_greedy', reward_type='combo'):

        super(DeepQRNN_Agent, self).__init__(board, n_moves, reward_type)
        ## Some hyper params
        self.name = 'DeepQAgent'
        self.sample_mode = sample_mode
        self.learning_rate = 0.00001
        self.epsilon = 0.9
        self.gamma = 0.8
        # self.history = History(capacity=memory) # No history with rnn
        self.batch_size = batch_size
        self.global_step = 0

        # Number of combos to accomplish 
        self.target = target

        print('Setting up Tensorflow')
        self.onlineQ = DeepQRNN(self.n_actions)

        # print('Observing behavior')
        # self.observe()

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
        # Reset previous number of matches register.
        self.prev_matches = 0
        self.onlineQ.reset_lstm()
        # Gather a whole game
        s_ts = []; a_ts = []; s_t1s = []; r_ts = []; q_ts = []; q_t1s = []; is_end = []
        with tf.GradientTape() as tape:
            while self.board.selected is not None and move <= n_moves:
                ## Feed history new observations
                move += 1
                self.global_step += 1
                ## Observe the models' behavior
                ## Get action and predicted q value from state
                s_t = self.board.board_2_state()

                # Implement epsilon-greedy sampling:
                a_t, q_t = self.onlineQ(s_t) ## Always get a q_t .. ?
                if np.random.uniform(low=0, high=1, size=1) < self.epsilon:
                    a_t = self.random_action()
                else:
                    a_t = a_t[0]

                terminal_move, selected_position = self.board.move_orb(a_t)
                s_t1 = self.board.board_2_state()
                q_t1 = self.onlineQ.predictQ(s_t1, keep_lstm_state=True)
                r_t = self._eval_reward(terminal_move, selected_position)

                if verbose:
                    print('move {:03d} ep a_t: {} (mode {}) r_t: {}'.format(
                        move, a_t, self.sample_mode, r_t))

                s_ts.append(s_t)
                a_ts.append(a_t)
                s_t1s.append(s_t1)
                r_ts.append(r_t)
                q_ts.append(q_t)
                q_t1s.append(q_t1)
                is_end.append(terminal_move)

                if terminal_move:
                    break
                
            s_ts = tf.concat(s_ts, axis=0)
            a_ts = tf.stack(a_ts, axis=0)
            s_t1s= tf.concat(s_t1s, axis=0)
            r_ts = tf.stack(r_ts, axis=0)
            q_ts = tf.cast(tf.concat(q_ts, axis=0), tf.float32)
            q_t1s = tf.cast(tf.concat(q_t1s, axis=0), tf.float32)
            is_end = tf.stack(is_end, axis=0)

            # print('s_ts', s_ts.shape, s_ts.dtype)
            # print('a_ts', a_ts.shape, a_ts.dtype) 
            # print('s_t1s', s_t1s.shape, s_t1s.dtype) 
            # print('r_ts', r_ts.shape, r_ts.dtype)
            # print('q_ts', q_ts.shape, q_ts.dtype)
            # print('q_t1s', q_t1s.shape, q_t1s.dtype) 
            # print('is_end', is_end.shape, is_end.dtype)

            # nextQ = q_j.numpy()
            nextQ = q_ts.numpy()
            for idx, (a_jx, r_jx, is_end_j) in enumerate(zip(a_ts, r_ts, is_end)):
                if is_end_j:
                    nextQ[idx, a_jx] = r_jx
                else:
                    nextQ[idx, a_jx] = r_jx + self.gamma * np.max(q_t1s[idx,:])
            
            # nextQ = tf.constant(nextQ, dtype=tf.float32)
            loss = self.onlineQ.loss_fn(nextQ, q_t1s, a_ts)
            grads = tape.gradient(loss, self.onlineQ.variables)

        # for gr, v in zip(grads, self.onlineQ.variables):
        #     try:
        #         print(gr.shape, v.name)
        #     except:
        #         print(v.name, 'None grad')
        self.onlineQ.optimizer.apply_gradients(zip(grads, self.onlineQ.variables))

        # loss = self.onlineQ.q_train(s_ts, a_ts, r_ts, s_t1s, q_ts, q_t1s, is_end)
        if verbose:
            print('\t loss = {}'.format(loss))

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
