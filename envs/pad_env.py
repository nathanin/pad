'''
Game board script
'''
import sys, pygame, time
import numpy as np
import gym
from gym import spaces

from .orb import Orb

""" Puzzle environment in OpenAI gym API
move_count -- start at 1 and anneal towards 0 with more moves -- simulate time

"""
class PadEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }
    def __init__(self, shape=[5,6], target=3, max_moves=1000, sleep_time=0.25):
        super(PadEnv, self).__init__()
        self.shape = shape
        self.selected = None
        self.target = target
        self.sleep_time = sleep_time
        self.move_count = 1.0
        self.move_delta = 1.0 / max_moves
        self.prev_max_combo = 0
        self.target_reward = 10.
        self.num_envs = 4

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=6, 
                                            shape=(self.shape[0], self.shape[1], 2), 
                                            # shape=(np.prod(self.shape), 2), 
                                            dtype=np.float32)

        self.orb_size = 35 # orb size in pixels; for drawing
        self.selected_dot = 15 

        self._init_orbs()
        print('Clearing initialized orbs')
        _ = self.reset() #Clear randomly drawn matches
        print('Environment ready')

    def step(self, action, show_selected=True):
        """ AKA. Move orb

        Input:
            action (int)

        Returns:
            state (np array)
            reward (float)
            done (bool)
            info (dict) [<-- empty]
        """
        # if isinstance(action, np.ndarray):
        #     obs = []
        #     rewards = []
        #     dones = []
        #     infos = []
        #     for act in action:
        #         ob, reward, done, info = self.step(act)
        #         obs.append(ob)
        #         rewards.append(reward)
        #         dones.append(done)
        #         infos.append(info)
        #     dones = np.asarray(dones)
        #     return obs, rewards, dones, infos

        ## Update internal clock
        self.move_count -= self.move_delta

        valid_actions = self.selected.get_possible_moves(self)
        # assert action in valid_actions, "Action {} Illegal action".format(action)
        if action not in valid_actions:
            return self._eval_reward(invalid=True)

        position = sr, sc = self.selected.position

        if self.move_count <= 0:
            return self._eval_reward()

        if action==0: # left
            newpos = [sr, sc-1]
        elif action==1: # up
            newpos = [sr-1, sc]
        elif action==2: # right
            newpos = [sr, sc+1]
        elif action==3: # down
            newpos = [sr+1, sc]
        else:
            print('wtf')

        ## Do the swap
        hold = self.orbs[newpos[0], newpos[1]]
        self.orbs[newpos[0], newpos[1]] = self.selected
        self.orbs[position[0], position[1]] = hold

        ## update
        hold.update_position(position)
        self.selected.update_position(newpos)

        return self._eval_reward()

    def reset(self, seed=None):
        """ Reset PadEnv to random board state

        Optionally take a seed argument to replicate exact starting configs.
        """
        if seed is not None:
            np.random.seed(seed)
        self.move_count = 1.0
        self.prev_max_combo = 0
        self.selected = None
        [orb.set_random_type() for orb in self.orbs.flatten()]

        n = self.eval_matches(clear=True)
        while n > 0:
            self.clear_matches(draw=False)
            self.skyfall()
            n = self.eval_matches(clear=True)
        
        self._select_random(seed=seed)

        return self.observe()

    def _eval_reward(self, invalid=False):
        reward = 0.0
        state = self.observe()
        if invalid:
            reward = -1.
            done = True
            return state, reward, done, {}

        n_combos = self.eval_matches()
        if self.move_count <= 0:
            done = True
            reward = n_combos
        else:
            done = False

        if n_combos >= self.target:
            done = True
            reward = self.target_reward
            print('Returning reward = {}'.format(reward))
        elif n_combos > self.prev_max_combo:
            reward = 1.
            self.prev_max_combo = n_combos
        else:
            reward = 0.

        return state, reward, done, {}
            
    '''
    Improvement: smartly generate non-matches
    '''
    def _init_orbs(self):
        n = np.prod(self.shape)
        print('generating {} orbs'.format(n))
        orbs = np.array(
            [Orb(board_shape=self.shape, radius=self.orb_size) for _ in range(n)])
        orbs = orbs.reshape(self.shape)

        for k in range(orbs.shape[0]):
            for m in range(orbs.shape[1]):
                orbs[k,m].update_position([k,m])

        self.orbs = orbs
        self._select_random()

    def _select_random(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.selected = np.random.choice(self.orbs.flatten())

    def _aggregate_combos(self):
        ## Take care of the special case with two parallel combos:
        # Go rows first, if there are any row-wise matches, check the next one for
        # another row match of the same type
        # If types agree, switch the label to the smaller of the two
        # repeat for column
        for dr in range(self.shape[0]-1):
            row = self.orbs[dr,:]
            ref = [orb for orb in self.orbs[dr,:] if orb.is_matched]
            comp = [orb for orb in self.orbs[dr+1,:] if orb.is_matched]

            if len(ref) < 3: continue
            if len(comp) < 3: continue

            for ref_orb in ref:
                pos = ref_orb.position[1]
                comp_orb = self.orbs[dr+1, pos]
                if not comp_orb.is_matched: continue
                if comp_orb.type == ref_orb.type and \
                   comp_orb.match_id != ref_orb.match_id:
                    # print 'DR Reassigning {} {} {} --> {} {} {}'.format(
                        # comp_orb.type, comp_orb.position, comp_orb.match_id,
                        # ref_orb.type, ref_orb.position, ref_orb.match_id)
                    self._switch_combo_label(comp_orb.match_id, ref_orb.match_id)

        for dc in range(self.shape[1]-1):
            row = self.orbs[:, dc]
            ref = [orb for orb in self.orbs[:,dc] if orb.is_matched]
            comp = [orb for orb in self.orbs[:,dc+1] if orb.is_matched]

            if len(ref) < 3: continue
            if len(comp) < 3: continue

            for ref_orb in ref:
                pos = ref_orb.position[0]
                comp_orb = self.orbs[pos, dc+1]
                if not comp_orb.is_matched: continue
                if comp_orb.type == ref_orb.type and \
                   comp_orb.match_id != ref_orb.match_id:
                    # print 'DC Reassigning {} {} {} --> {} {} {}'.format(
                        # comp_orb.type, comp_orb.position, comp_orb.match_id,
                        # ref_orb.type, ref_orb.position, ref_orb.match_id)
                    self._switch_combo_label(comp_orb.match_id, ref_orb.match_id)

        matched_orbs = np.array([orb for orb in self.orbs.flatten() if orb.is_matched])
        match_ids = np.array([orb.match_id for orb in matched_orbs])

        matches = []
        for uid in np.unique(match_ids):
            matches.append(matched_orbs[match_ids == uid])

        # print '_aggregate_combos: matches: {}'.format(len(matches))
        self.matches = matches
        return len(matches)

    '''
    Goddamnit it's switching row, col again
    '''
    def _highlight_selected(self):
        if self.selected is None:
            return
        else:
            sc, sr = self.selected.board_position
            rad = self.orb_size
            pygame.draw.circle( self.screen, (245, 245, 245),
                (sc, sr), self.selected_dot, 0)

    '''
    Logic to evaluate matches in:

    _switch_combo_label()
    _combo_search()

    Think about moving this to a different file
    Can be a new file with lots of factored out FUNCTIONS
    Sorta unnecessary but I just don't want this to get too big.
    '''
    def _switch_combo_label(self, match_id, new_id):
        ## Work on all orbs
        orbs_to_swap = [orb for orb in self.orbs.flatten()
            if orb.match_id == match_id]
        [orb.set_match_id(new_id) for orb in orbs_to_swap]

    def _combo_search(self, orb):
        orb_type = orb.type
        board_r, board_c = self.shape
        pos = r,c = orb.position

        ## Search in the positive direction
        row_match = [orb]
        for d_r in range(r+1,board_r):
            query_orb = self.orbs[d_r, c]
            if query_orb.type == orb_type:
                row_match.append(query_orb)
            else:
                break
        ## Check length of matches
        if len(row_match) >= 3:
            uid = [orb.match_id for orb in row_match if orb.is_matched]

            [o.set_is_matched() for o in row_match]
            if len(uid) == 0:
                new_id = self.combos
                self.combos += 1
            else:
                new_id = min(uid)

            # print 'Assigning match to combo #{}'.format(uid)
            [o.set_match_id(new_id) for o in row_match]
            [self._switch_combo_label(match_id, new_id) for match_id in uid]

        col_match = [orb]
        for d_c in range(c+1,board_c):
            query_orb = self.orbs[r, d_c]
            if query_orb.type == orb_type:
                col_match.append(query_orb)
            else:
                break
        ## Check length of matches
        if len(col_match) >= 3:
            uid = [orb.match_id for orb in col_match if orb.is_matched]

            [o.set_is_matched() for o in col_match]
            if len(uid) == 0:
                new_id = self.combos
                self.combos += 1
            else:
                new_id = min(uid)
            # print 'Assigning match to combo #{}'.format(uid)
            [o.set_match_id(new_id) for o in col_match]
            [self._switch_combo_label(match_id, new_id) for match_id in uid]

            ## Recurse.. how to terminate??
            # [self._combo_search(o) for o in col_match]

    """ Externally callable function to evaluate combos on the board
    """
    def eval_matches(self, clear=True):
        ## First unset selected:
        # print '----------------- EVALUATING MATCHES'
        self.combos = 0
        for orb in self.orbs.flatten():
            # if not orb.is_matched:
            self._combo_search(orb)

        n_matches = self._aggregate_combos()

        ## For intermittent rewarding:
        if not clear:
            [orb.unset_is_matched() for orb in self.orbs.flatten()]

        return n_matches

    def move_skyfall(self):
        """ Do a move with skyfall afterwards
        This does not check for legal moves. 
        """
        position = sr, sc = self.selected.position
        newpos = [sr-1, sc]

        hold = self.orbs[newpos[0], newpos[1]]
        self.orbs[newpos[0], newpos[1]] = self.selected
        self.orbs[position[0], position[1]] = hold

        ## update
        hold.update_position(position)
        self.selected.update_position(newpos)


    def clear_matches(self, draw=True):
        """
        Set matched to 'cleared'
        """
        [orb._clear() for orb in self.orbs.flatten() if orb.is_matched]

        self.matches = None

        ## By iterating over all rows we do it good with one pass
        for column in range(self.shape[1]):
            column_orbs = self.orbs[:,column]
            cleared = [orb for orb in column_orbs if orb.type=='cleared']
            for cleared_orb in cleared:
                self.selected = cleared_orb
                while 1 in self.selected.get_possible_moves(self):
                    ## TODO write hidden fn for this
                    self.move_skyfall()

        ## Deselect
        self.selected = None

    '''
    The actual order doesn't matter except for visualizing the update
    We only evaluate matches after skyfall is done anyway
    '''
    def skyfall(self):
        cleared_orbs = [orb for orb in self.orbs.flatten() if orb.type == 'cleared']
        [orb.set_random_type() for orb in cleared_orbs]


    def select_orb(self, loc):
        self.selected = self.orbs[loc[0], loc[1]]


    def _as_onehot(self, state):
        onehot = np.eye(Orb.n_types)[state]
        onehot = onehot.reshape(self.shape[0], self.shape[1], Orb.n_types)
        return onehot

    def observe(self, flat=False):
        state = [orb.type_code for orb in self.orbs.flatten()]
        state = np.array(state, dtype=np.float32)

        ## Selected represents the selected orb __and__ the internal clock
        ## the clock 'ticks' down linearly towards 0, when the turn is over
        selected = np.zeros(self.shape, dtype = np.float32) #+ self.move_count
        if self.selected is not None:
            sr,sc = self.selected.position
            selected[sr,sc] = self.move_count ## 2 instead of 1

        if flat:
            state = state.reshape(np.prod(self.shape))
            selected = selected.flatten().reshape(np.prod(self.shape))
            state = np.stack([state, selected], axis=-1)
        else:
            state = state.reshape(self.shape[0], self.shape[1], 1)
            # state = self._as_onehot(state)
            state = np.dstack([state, selected])
            # state = np.expand_dims(state, 0)

        return state

