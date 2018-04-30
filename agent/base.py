import numpy as np


class BaseAgent(object):
    def __init__(self, board, n_moves=10, reward_type='combo'):
        self.board = board
        self.n_moves = n_moves
        self.reward_type = reward_type

        ## Actions - left, up, right, down, stop
        self.n_actions = 5

    def eval_outcome(self):
        combos = 0
        n_matches = self.board.eval_matches()

        ## matches on the first board
        n_cleared = 0
        for orb in self.board.orbs.flatten():
            n_cleared += orb.is_matched

        combos += n_matches
        while n_matches > 0:
            self.board.clear_matches()
            self.board.skyfall()
            n_matches = self.board.eval_matches()
            combos += n_matches
        return combos, n_cleared


    def swap_board(self, newboard):
        self.board = newboard


    """ Action from onlineQ """
    def _sample_action(self, s_t, verbose=False):
        if self.sample_mode == 'e_greedy':
            if np.random.rand(1)[0] < self.epsilon:
                # a_t = np.random.choice(range(self.n_actions))

                ## Restrict to legal actions
                action_space = self.board.selected.get_possible_moves(self.board)
                action_space.append(4)
                a_t = np.random.choice(action_space)

                # a_t = np.random.choice([a_t, a_t_legal])
            else:
                feed_dict = {self.onlineQ.state: s_t, self.onlineQ.keep_prob: 1.0}
                a_t, Qpred = self.sess.run([self.onlineQ.action_op, self.onlineQ.Qpred],
                    feed_dict=feed_dict)
                a_t = a_t[0]

        elif self.sample_mode == 'bayes':
            Qpred_bootstrap = []
            feed_dict = {self.onlineQ.state: s_t, self.onlineQ.keep_prob: self.epsilon}
            for _ in range(self.n_bootstrap):
                a_t, Qpred = self.sess.run([self.onlineQ.action_op, self.onlineQ.Qpred],
                    feed_dict=feed_dict)
                Qpred_bootstrap.append(Qpred)

            Qpred_mean = np.mean(np.vstack(Qpred), 0)
            a_t = np.argmax(Qpred_mean)
            # a_t = a_t[0]
        else:
            print('{} not recognized'.format(self.sample_mode))
            a_t = None
        #/end if
        return a_t


    """ Utility procedure to fill up memory """
    def observe(self, pause_time=0, verbose=False):
        print('Observing {} sampling mode'.format(self.sample_mode))
        n_moves = self.n_moves

        ## Set epsilon-greedy pretraining condition
        ## Bayesian preconditioning seems to give unbalanced initialization
        store_mode = self.sample_mode
        store_epsilon = self.epsilon
        self.sample_mode = 'e_greedy'
        self.epsilon = 0.75

        ## Do many runs to fill the memory
        while self.history.n < self.history.capacity:
            ## Iterate games
            self.board.refresh_orbs()
            self.board.select_orb([0,0])
            move = 0
            total_reward = 0
            previous_combo = 0
            while self.board.selected is not None:
                ## Feed history
                move += 1
                s_t = self.board.board_2_state()

                ## Get an action from target net
                a_t = self._sample_action(s_t)

                ## Observe altered state(t+1)
                terminal_move, selected_pos = self.board.move_orb(a_t)
                s_t1 = self.board.board_2_state()

                # r_t = self._eval_distance_reward(selected_pos, terminal_move)
                r_t = self._eval_reward(terminal_move, selected_pos)

                self.history.store_state(s_t, a_t, r_t, s_t1, terminal_move)

                if terminal_move:
                    break
        ## Restore settings
        self.sample_mode = store_mode
        self.epsilon = store_epsilon
        print('Done observing; Agent memory full')


    def _distance_reward(self, position, terminal_move):
        xb, yb = self.board.shape
        max_dist = np.sqrt(xb**2 + yb**2)
        x,y = position
        # if terminal_move:
        r_t = np.sqrt(x**2 + y**2)/max_dist #+ self.board.move_count
        # else:
        #     r_t = 0

        return r_t

    def _combo_reward(self, terminal_move):
        if terminal_move:
            r_t, _ = self.eval_outcome()
        else:
            r_t = 0

        return r_t

    """ Experiment with mid-move and terminal move rewards """
    def _eval_reward(self, terminal_move, position=None):
        ## We've reached the end of the game
        # if self.board.selected is None:
        if self.reward_type == 'combo':
            r_t = self._combo_reward(terminal_move)
        elif self.reward_type == 'distance':
            assert position is not None, 'Position cannot be None for distance reward'
            r_t = self._distance_reward(position, terminal_move)

        return r_t
