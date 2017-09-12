import numpy as np
import tensorflow as tf
import time
import models

'''

With heavy reference to
https://github.com/asrivat1/DeepLearningVideoGames

'''


class History(object):
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'state_p': [],
            'is_end': []
        }
        self.n = 0
        self.update_cycle = capacity


    def flush_history(self):
        print 'Dumping memory'
        self.n = 0
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'state_p': [],
            'is_end': []
        }


    ''' store new observations in order '''
    def store_state(self, state, action, reward, state_p, is_end):
        if self.n == self.capacity:
            self._pop_state()

        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['state_p'].append(state_p)
        self.memory['is_end'].append(is_end)

        self.n += 1
        self.update_cycle -= 1

        if self.update_cycle == 0:
            print 'History refreshed'
            self.update_cycle = self.capacity


    ''' remove the last (oldest) state '''
    def _pop_state(self):
        for key in self.memory.iterkeys():
            self.memory[key].pop(0)

        self.n -= 1

    ''' return a minibatch '''
    def minibatch(self, batch_size):
        assert batch_size < self.n
        indices = np.random.choice(range(self.n), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        state_ps = []
        is_ends = []
        for index in indices:
            states.append(self.memory['state'][index])
            actions.append(self.memory['action'][index])
            rewards.append(self.memory['reward'][index])
            state_ps.append(self.memory['state_p'][index])
            is_ends.append(self.memory['is_end'][index])

        states = np.concatenate(states, 0)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        state_ps = np.concatenate(state_ps, 0)
        is_ends = np.asarray(is_ends)

        return states, actions, rewards, state_ps, is_ends


    """ Debugging function: print some info about the history state """
    def _print_memory(self):
        actions = np.asarray(self.memory['action'])
        actions = [(actions == k).sum()/float(self.n) for k in range(5)]

        rewards = np.asarray(self.memory['reward'])
        is_ends = np.asarray(self.memory['is_end'])

        print '--------- Experience Summary ----------'
        print 'Experience history with {} entries'.format(self.n)
        print 'Actions distribution:\n\t0:{:2.3f}\n\t1:{:2.3f}\n\t2:{:2.3f}\n\t3:{:2.3f}\n\t4:{:2.3f}'.format(
            actions[0],
            actions[1],
            actions[2],
            actions[3],
            actions[4], )
        print 'Rewards: mean: {} min: {} max: {}'.format(
            np.mean(rewards), rewards.min(), rewards.max() )
        print 'Rewards non-0: {} mean: {} min: {} max: {}'.format(
            (rewards!=0).sum(),
            np.mean(rewards[rewards!=0]),
            rewards[rewards!=0].min(),
            rewards[rewards!=0].max() )
        try:
            print 'Terminal actions: {}'.format((is_ends == True).sum())
        except:
            print 'Teminal actions: None??'


"""
Attempt at OOP
"""
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


'''

Baseline agent that chooses completely randomly at every step
Except that it only chooses from the set of LEGAL moves
so that it always performes n_moves number of moves.

'''

class RandomAgent(BaseAgent):
    def __init__(self, board, n_moves=10):
        super(RandomAgent, self).__init__(board, n_moves)
        self.name = 'RandomAgent'


    def step(self):
        self.board.refresh_orbs()
        # print self.board.board_2_state()

        self.board.selected = self.board.orbs[0,0]  ## EZ start from top left
        # self.board._select_random()
        for _ in range(self.n_moves):
            ## REVIEW this is wack AF
            moves = self.board.selected.get_possible_moves(self.board)
            self.board.move_orb(direction=np.random.choice(moves))

        combos, n_cleared = self.eval_outcome()
        return combos



'''
Made heavy use of:
https://github.com/awjuliani/DeepRL-Agents

I think the first idea to implement is take in a state and return
some action that ought to maximize reward
In such case what is the optimization?

    Deep Q Pseudo Code [Mnih et al, 2015]:

    Initialize replay memory D to size N
        --> Empty
    Initialize action-value function Q with random weights, theta_0
        --> Random Q
    for episode = 1, M do  #--> actor.run_loop()
    	Initialize state s_1  #--> board.refresh_orbs()
    	for t = 1, T do  #--> T = n_moves ## easy
    		With probability EPSILON select random action a_t
    		otherwise select a_t=argmax_a  Q(s_t, a; THETA_i)
    		Execute action a_t in emulator and observe r_t and s_(t+1)
    		Store transition (s_t, a_t, r_t, s_(t+1)) in D
    		Sample a minibatch of transitions (s_j, a_j, r_j, s_(j+1)) from D
    		Set y_j:=
    			r_j for terminal s_(j+1)
    			r_j + GAMMA * max_(a^') Q(s_(j+1), a'| THETA_i) for non-terminal s_(j+1)
    		Perform a gradient step on (y_j - Q(s_j, a_j| THETA_i))^2 with respect to THETA
    	end for
    end for
'''

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
        self.learning_rate = 1e-9
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

        print 'Setting up Tensorflow'
        self.targetQ = models.DuelingQ(self.board, self.n_actions, self.learning_rate)
        self.onlineQ = models.DuelingQ(self.board, self.n_actions, self.learning_rate)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # self._setup_network_v1()
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.onlineQ.init_op)
        self.sess.run(self.targetQ.init_op)

        trainables = tf.trainable_variables()
        self.targetOps = models.updateTargetGraph(trainables, self.tau)
        print self.targetOps

        print 'Agent ready'


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
            print '{} not recognized'.format(self.sample_mode)
            a_t = None
        #/end if
        return a_t


    """ Utility procedure to fill up memory """
    def observe(self, pause_time=0, verbose=False):
        print 'Observing {} sampling mode'.format(self.sample_mode)
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
            # self.board._select_random()
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
        print 'Done observing; Agent memory full'



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
            # if self.sample_mode == 'bayes':
            #     keep_prob = self.epsilon
            # else:
            #     keep_prob = 1.0

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
                    print '(updating)',
                models.updateTarget(self.targetOps, self.sess)


            if terminal_move:
                if verbose:
                    print 'moves: {} r_t = {} (lr: {}) (mode: {}) (end-turn {})'.format(
                        move, r_t, self.learning_rate, self.sample_mode, terminal_move)

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



"""

Single Q Network learner Single Q Network learner
Single Q Network learner Single Q Network learner
Single Q Network learner Single Q Network learner

"""

class DeepQAgent(BaseAgent):
    def __init__(self, board, n_moves=10, batch_size=16, memory=1024):
        super(DeepQAgent, self).__init__(board, n_moves)
        ## Some hyper params
        self.name = 'DeepQAgent'
        self.learning_rate = 0.00001
        self.epsilon = 0.9
        self.gamma = 0.8
        self.history = History(capacity=memory)
        self.batch_size = batch_size
        self.global_step = 0

        print 'Setting up Tensorflow'
        # self.targetQ = models.DuelingQ(self.board, self.n_actions)
        self.onlineQ = models.DuelingQ(self.board, self.n_actions, self.learning_rate)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # self._setup_network_v1()
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.onlineQ.init_op)
        print 'Agent ready'




    '''
    Populate a replay history by playing games and storing moves
    '''
    def observe(self, pause_time=0, verbose=False):
        n_moves = self.n_moves

        ## Do many runs to fill the memory
        while self.history.n < self.history.capacity:

            ## Iterate games
            self.board.refresh_orbs()
            self.board._select_random()
            move = 0
            total_reward = 0
            previous_combo = 0
            while self.board.selected is not None and move<=n_moves:
                ## Feed history
                move += 1
                s_t = self.board.board_2_state()
                feed_dict = {self.onlineQ.state: s_t}
                a_t, Qpred = self.sess.run([self.onlineQ.action_op, self.onlineQ.Qpred],
                    feed_dict=feed_dict)

                ## Exploration -- in the beginning it will be very high
                if np.random.rand(1)[0] < self.epsilon:
                    a_t = np.random.choice(range(self.n_actions))
                    ## Restrict to legal actions
                    action_space = self.board.selected.get_possible_moves(self.board)
                    a_t_legal = np.random.choice(action_space)

                    # 1/2 chance to be selected from legal actions only
                    a_t = np.random.choice([a_t, a_t_legal])
                else:
                    a_t = a_t[0]

                ## Observe altered state(t+1)
                illegal_move = self.board.move_orb(a_t)
                s_t1 = self.board.board_2_state()

                r_t, previous_combo, do_break = self._eval_reward(
                    move, illegal_move, previous_combo)

                self.history.store_state(s_t, a_t, r_t, s_t1, do_break)

                if do_break:
                    break

        print 'Done observing; Agent memory full'



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
            # if verbose:
            #     print 'delta: {}, loss: {}'.format(delta, loss)

            if verbose:
                print 'move {} ep a_t: {} (mode {})'.format(
                    move, a_t, self.sample_mode),
                print 'loss= {}, r_t = {}'.format(loss, r_t)

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
