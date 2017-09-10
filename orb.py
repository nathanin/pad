'''



'''
import random

colors = {
    'blue':   (  0,   0, 255), #blue
    'purple': (148,   0, 211), #purple
    'red':    (255,   0,   0), #red
    'yellow': (255, 255,   0), #yellow
    'green':  (  0, 255,   0), #green
    'pink':   (255, 105, 180), #pink
    'cleared':(  0,   0,   0),
    'hidden': (120, 120, 120),
    'jammer': ( 97,  88,  99)
}

orbtypes = [
            # 'blue',
            # 'red',
            'green',
            'pink',
            'purple',
            'yellow',
            # 'jammer',
            ]

typedict = {
    'blue':   0., #blue
    'purple': 1., #purple
    'red':    2., #red
    'yellow': 3., #yellow
    'green':  4., #green
    'pink':   5., #pink
    'cleared':6.,
    'hidden': 7.,
    'jammer': 8.
}

typedict = {k: idx/len(orbtypes) for idx, k in enumerate(typedict.iterkeys())}

class Orb(object):

    ''' Allow choice of type but default to random '''
    def __init__(self, type=None, board_shape=[5,6], radius=35):
        ## Constants
        self.type = random.choice(orbtypes)
        self.color = colors[self.type]
        self.type_code = typedict[self.type]
        self.position = None
        self.is_matched = False
        self.match_id = None
        self.board_shape = board_shape
        self.radius = radius
        self.board_position = None

    def update_position(self, position):
        rad = self.radius
        r, c = position
        row, col = self.board_shape
        self.position = position
        self.board_position = ( 2*c*rad+rad, 2*r*rad+rad )

    def set_random_type(self):
        self.type = random.choice(orbtypes)
        self.color = colors[self.type]
        self.type_code = typedict[self.type]

    def _clear(self):
        self.type = 'cleared'
        self.color = colors['cleared']
        self.type_code = typedict[self.type]
        self.set_is_matched(False)
        self.set_match_id(None)

    def set_match_id(self, uid):
        self.match_id = uid

    def set_is_matched(self, status=True):
        self.is_matched = status

    def unset_is_matched(self):
        self.match_id = None
        self.is_matched = False

    def get_possible_moves(self, board):
        row, col = board.shape
        r, c = self.position

        # Logic returns a list of possible moves
        # 0-left, 1-up, 2-right, 3-down
        moves = []
        if r >= 1: moves.append(1)
        if c >= 1: moves.append(0)
        if r < row-1: moves.append(3)
        if c < col-1: moves.append(2)

        return moves
