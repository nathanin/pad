import numpy as np
from .base import BaseAgent
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
