from __future__ import print_function
import sys
import pygame
import numpy as np

from .pad_env import PadEnv
"""
Puzzle environment that supports rendering
"""

class PadEnvRender(PadEnv):
    """
    Pad Environment with pygame drawing to screen.

    :param orb_size: Size in pixels to draw each orb
    """
    def __init__(self, shape=[5,6], target=3, max_moves=200, orb_size=30):
        # Set up the drawing stuff first
        self.shape = shape
        self.orb_size = orb_size
        self.selected_dot = int(orb_size / 2)
        self.surface = self._draw_init()
        self.screen = pygame.display.set_mode(self.size)

        super(PadEnvRender, self).__init__(shape=[5,6], target=3, max_moves=200)
        
    def _draw_init(self):
        pygame.init()
        row, col = self.shape
        self.size = [3*col*self.orb_size, 3*row*self.orb_size]

    def _highlight_selected(self):
        """ Draw a gray circle to indicate the active (selected) orb """
        if self.selected is None:
            return
        else:
            sc, sr = self.selected.board_position
            rad = self.selected_dot

            pygame.draw.circle(self.screen, (245, 245, 245),
                (sc, sr), rad, 0)

    def _display_matches(self):
        """ Draw a light/white-ish circle to indicate matched orbs """
        matches = self.matches
        rad = self.selected_dot

        for combo in matches:
            for orb in combo:
                pygame.draw.circle(self.screen, (15, 15, 15),
                    orb.board_position, rad, orb.match_id+1)

        pygame.display.flip()

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

        n = self.eval_matches(clear=True, reset=True)
        while n > 0:
            self.clear_matches()
            self.skyfall()
            n = self.eval_matches(clear=True, reset=True)
        
        self._select_random(seed=seed)

        return self.observe()

    def eval_matches(self, clear=False, reset=False):
        """ Evaluate and draw matches on board """
        self.combos = 0
        for orb in self.orbs.flatten():
            self._combo_search(orb)
        
        n_matches = self._aggregate_combos()
        if not reset:
            self._display_matches()

        if not clear:
            [orb.unset_is_matched() for orb in self.orbs.flatten()]
        
        return n_matches

    def render(self):
        rad = self.orb_size
        row, col = self.shape
        self.screen.fill((0,0,0)) # black background

        # Add circles at x,y coordinates for orbs
        for orb in self.orbs.flatten():
            pygame.draw.circle(
                self.screen, orb.color, orb.board_position, orb.radius, 0)

        self._highlight_selected()
        pygame.display.flip()  # push to screen
