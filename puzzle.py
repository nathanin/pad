'''
Game board script
'''
import sys, pygame, time
import numpy as np

from orb import Orb

"""

move_count -- start at 1 and anneal towards 0 with more moves -- simulate time

"""
class Board(object):
    def __init__(self, shape=[5,6], show=True, max_moves=50, sleep_time=0.25):
        self.orb_size = 45
        self.selected_dot = 15
        self.shape = shape
        self.selected = None
        self.show = show
        self.sleep_time = sleep_time
        self.move_count = 1.0
        self.move_delta = 1.0 / max_moves
        ## Big choice: use np to hold orbs
        ## Later: write a new solution not depending on np
        ## Looks like in order to get something interactive, use Rect()

        if self.show:
            self.surface = self._draw_init()
            self.screen = pygame.display.set_mode(self.size)
            self._init_orbs()
            self.draw_board()
            # pygame.display.flip()
        else:
            self._init_orbs()


        ## Perform the ghost iteration to get a neutral board
        show_hold = self.show
        self.show = False
        n = self.eval_matches()
        while n > 0:
            self.clear_matches()
            self.skyfall()
            n = self.eval_matches()
        self.show = show_hold

    '''
    Improvement: smartly generate non-matches
    '''
    def _init_orbs(self):
        n = np.prod(self.shape)
        print 'generating {} orbs'.format(n)
        orbs = np.array(
            [Orb(board_shape=self.shape, radius=self.orb_size) for _ in range(n)])
        orbs = orbs.reshape(self.shape)

        for k in range(orbs.shape[0]):
            for m in range(orbs.shape[1]):
                orbs[k,m].update_position([k,m])

        self.orbs = orbs
        self._select_random()


    def _draw_init(self):
        pygame.init()
        row, col = self.shape
        self.size = [2*col*self.orb_size, 2*row*self.orb_size]


    def _select_random(self):
        self.selected = np.random.choice(self.orbs.flatten())
        if self.show:
            self._highlight_selected()
            time.sleep(2)
        # self.selected = self.orbs[0,0]
        # print self.selected, self.selected.type, self.selected.position


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
    Here's the one.
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


    def eval_matches(self, no_clear=False, draw=True):
        ## First unset selected:
        # print '----------------- EVALUATING MATCHES'
        self.combos = 0
        for orb in self.orbs.flatten():
            # if not orb.is_matched:
            self._combo_search(orb)

        n_matches = self._aggregate_combos()
        if self.show and draw:
            self.display_matches()
            time.sleep(self.sleep_time)

        ## For intermittent rewarding:
        if no_clear:
            [orb.unset_is_matched() for orb in self.orbs.flatten()]

        return n_matches


    def display_matches(self):
        matches = self.matches

        for combo in matches:
            for orb in combo:
                pygame.draw.circle( self.screen, (15, 15, 15),
                    orb.board_position, self.selected_dot, orb.match_id+1)

        pygame.display.flip()


    '''
    Luckily movements are discretized to one space at a time
    They also amount to swapping the orbs
    coded: 0, 1, 2, 3 : left, up, right, down (clockwise)
    Add diagonal later; not too bad but it doubles the move space.
    '''

    def move_skyfall(self):
        position = sr, sc = self.selected.position
        newpos = [sr-1, sc]

        hold = self.orbs[newpos[0], newpos[1]]
        self.orbs[newpos[0], newpos[1]] = self.selected
        self.orbs[position[0], position[1]] = hold

        ## update
        hold.update_position(position)
        self.selected.update_position(newpos)


    def move_orb(self, action, show_selected=True):
        # take care of legal moves somewhere else;
        # if we get here assume the move is legal;
        position = sr, sc = self.selected.position
        ## Update internal clock
        self.move_count -= self.move_delta
        if action not in self.selected.get_possible_moves(self):
            ## This way only explicitly chosing 4 can terminate the turn
            if action==4 or self.move_count<=0:
                ## Deselect orb; Effectively ending the turn
                self.selected = None
                return True
            else:
                ## Set illegal move to noop
                ## noop is implicit - we want it to be contextual
                if self.move_count <= 0:
                    return True
                else:
                    return False

        if action==0: # left
            newpos = [sr, sc-1]
        elif action==1: # up
            newpos = [sr-1, sc]
        elif action==2: # right
            newpos = [sr, sc+1]
        elif action==3: # down
            newpos = [sr+1, sc]
        else:
            print 'wtf'

        ## Do the swap
        hold = self.orbs[newpos[0], newpos[1]]

        self.orbs[newpos[0], newpos[1]] = self.selected
        self.orbs[position[0], position[1]] = hold

        ## update
        hold.update_position(position)
        self.selected.update_position(newpos)

        if self.show:
            self.draw_board(show_selected=show_selected)

        if self.move_count <= 0:
            return True
        else:
            return False


    def draw_board(self, show_selected=True):
        rad = self.orb_size
        row, col = self.shape
        # sr, sc = self.selected.position
        self.screen.fill((0,0,0))
        for orb in self.orbs.flatten():
            pygame.draw.circle(
                self.screen, orb.color, orb.board_position, orb.radius, 0)

        if show_selected:
            self._highlight_selected()
        pygame.display.flip()
        time.sleep(self.sleep_time)


    '''
    Actually sets type to 'blank'
    Cascade down
    '''
    def clear_matches(self, draw=True):
        ## Set matched to 'cleared'
        [orb._clear() for orb in self.orbs.flatten() if orb.is_matched]

        if self.show and draw:
            self.draw_board(show_selected=False)
            time.sleep(self.sleep_time)
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
                    if self.show and draw:
                        self.draw_board(show_selected=False)

        ## Deselect
        self.selected = None


    '''
    The actual order doesn't matter except for visualizing the update
    We only evaluate matches after skyfall is done anyway
    '''
    def skyfall(self):
        cleared_orbs = [orb for orb in self.orbs.flatten() if orb.type == 'cleared']
        [orb.set_random_type() for orb in cleared_orbs]

        if self.show:
            self.draw_board(show_selected=False)
            time.sleep(self.sleep_time)

    def select_orb(self, loc):
        self.selected = self.orbs[loc[0], loc[1]]

        if self.show:
            self.draw_board(show_selected=False)
            time.sleep(self.sleep_time)



    def refresh_orbs(self):
        self.move_count = 1.0
        show_hold = self.show
        self.show = False
        self.selected = None
        [orb.set_random_type() for orb in self.orbs.flatten()]

        n = self.eval_matches()
        while n > 0:
            self.clear_matches(draw=False)
            self.skyfall()
            n = self.eval_matches()

        self.show = show_hold
        if self.show:
            self.draw_board(show_selected=False)
            time.sleep(2)

        # print 'refresh_orbs: {}'.format(time.time() - tstart)

    ''' Not sure if needed; pulls orb config '''
    def board_2_state(self, flat=False):
        state = [orb.type_code for orb in self.orbs.flatten()]
        state = np.array(state)

        ## Selected represents the selected orb and the internal clock
        ## the clock 'ticks' down linearly towards 0, when the turn is ended
        selected = np.zeros(self.shape, dtype = np.float32) #+ self.move_count
        # selected = np.zeros(self.shape, dtype = np.float32)
        if self.selected is not None:
            sr,sc = self.selected.position
            selected[sr,sc] = self.move_count ## 2 instead of 1
            # selected[sr, sc] = 2.0


        if flat:
            state = state.reshape(1,np.prod(self.shape))
            selected = selected.flatten().reshape(1,np.prod(self.shape))
            state = np.hstack([state, selected])
        else:
            state = state.reshape(self.shape[0], self.shape[1])
            # selected = selected.flatten().reshape(self.shape[0], self.shape[1])
            state = np.expand_dims(np.dstack([state, selected]), 0)

        return state



    """

    Example run loop

    I've re-implemented this loop in `actor.py`

    """
    def run(self):
        self.screen = pygame.display.set_mode(self.size)
        self.draw_board()
        pygame.display.flip()
        n_moves = 20
        play = True

        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    print 'Resuming'
                    play = True

            while play:
                self._select_random()
                for _ in range(n_moves):
                    # Sample from legal moves
                    moves = self.selected.get_possible_moves(self)
                    self.move_orb(direction=np.random.choice(moves))
                    # self.draw_board()
                    # time.sleep(0.25)

                time.sleep(1)
                self.selected = None
                self.draw_board()
                n_matches = self.eval_matches()
                while n_matches > 0:
                    self.display_matches()
                    # time.sleep(0.5)
                    self.clear_matches()
                    # time.sleep(0.5)
                    self.skyfall()
                    # time.sleep(0.5)
                    n_matches = self.eval_matches()

                for event in pygame.event.get():
                    print event
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        ## TODO add keypress screen PAUSE
                        print 'Pausing'
                        play = False



# import board; x=board.Board(); x.run()

if __name__ == '__main__':
    x = Board()
    x.run()
