from __future__ import print_function
import numpy as np
import time
import argparse

import tensorflow as tf
import tensorflow.contrib.eager as tfe

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

import sys
sys.path.insert(0, '..')
import actor
from env.puzzle import PAD
from agent import DeepQAgent, RandomAgent, DeepQRNN_Agent

def main(args):
    shape = [5,6]
    moves = 25
    target = 2
    n_steps = 50000
    board = PAD(shape=shape, 
                target=target,
                max_moves=moves, 
                show=False)
    print('board set up')

    # print('setting up agent')
    agent = DeepQRNN_Agent(board,
        n_moves=moves,
        target=target, 
        batch_size=8,
        memory=2048,
        sample_mode='e_greedy',
        reward_type='combo')
    print('agent set up')

    actor.train_loop(agent, n_steps=n_steps)

    ## Replace board so we can watch some play
    board = PAD(shape=shape,
        target=target,
        max_moves=moves,
        show=True,
        sleep_time=0.1)
    agent.swap_board(board)
    actor.run_loop(agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)

    args = parser.parse_args()
    main(args)