import numpy as np

class BaseAgent(object):
    def __init__(self, board, n_moves=10, reward_type='combo'):
        self.board = board
        self.n_moves = n_moves
        self.reward_type = reward_type

        ## Actions - left, up, right, down, stop
        self.n_actions = 4

        ## Initialize some registers
        self.prev_matches = 0

    def eval_outcome(self):
        n_matches = self.board.eval_matches()
        d_matches = np.abs(self.prev_matches - n_matches)

        # Update the max number of matches
        if d_matches >= 1:
            self.prev_matches = n_matches

        return n_matches, d_matches

    def swap_board(self, newboard):
        self.board = newboard

    def random_action(self):
        action_space = self.board.selected.get_possible_moves(self.board)
        action = np.random.choice(action_space)
        return action

    """ Action from onlineQ """
    def _sample_action(self, s_t, verbose=False):
        if self.sample_mode == 'e_greedy':
            if np.random.rand(1)[0] < self.epsilon:
                a_t = self.random_action()
            else:
                a_t, Qpred = self.onlineQ(s_t)

        elif self.sample_mode == 'bayes':
            Qpred_bootstrap = []
            for _ in range(self.n_bootstrap):
                a_t, Qpred = self.onlineQ(s_t)
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
            # Restart for a new game -- TODO add this register to puzzle
            self.prev_matches = 0
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
                r_t = self._eval_reward(terminal_move, position=selected_pos)

                self.history.store_state(s_t, a_t, r_t, s_t1, terminal_move)

                if terminal_move:
                    break
        ## Restore settings
        self.sample_mode = store_mode
        self.epsilon = store_epsilon

        ##
        # print('History itmes:')
        # for item in self.history.memory.keys():
        #     try:
        #         print(item, self.history.memory[item][0].shape)
        #         print(item, self.history.memory[item][0])
        #     except:
        #         print(item, self.history.memory[item][0])

        print('Done observing; Agent memory full')

        self.history._print_memory()


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
        n_matches, d_matches = self.eval_outcome()
        # r_t, _ = self.eval_outcome()

        if n_matches >= self.target:
            return 10.
        elif terminal_move and n_matches <= self.target:
            return -1.
        elif d_matches > 0:
            return 1.
        else:
<<<<<<< HEAD
            # r_t = 0
            r_t = self.board.eval_matches()
=======
            return 0.
>>>>>>> b55e56576fa662285f1fd691da1a258c2820bf9d


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
