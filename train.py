import numpy as np
import time
import actor
from env.puzzle import PAD
from agent import DeepQAgent

shape = [5,6]
moves = 100
board = PAD(shape=shape, max_moves=moves, show=False)

agent = DeepQAgent(board,
    n_moves=moves,
    batch_size=64,
    memory=5000,
    sample_mode='e_greedy',
    reward_type='combo')

print('Max moves: ', agent.n_moves)
agent.observe()
actor.train_loop(agent)

## Replace board so we can watch some play
board = PAD(shape=shape,
    max_moves=moves,
    show=True,
    sleep_time=0.05)
agent.swap_board(board)
actor.run_loop(agent)
