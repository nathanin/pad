'''

Interface for Agents to interact with the board

'''
import pygame
import puzzle
import numpy as np
import agents
import sys
import time


def control_loop(events):
    for event in events:
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            print 'Pausing'
            paused = True
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        print 'Resuming'
                        paused = False


def run_loop(agent, n_steps=25):
    print 'Entering run_loop'
    for step in range(n_steps):
        if agent.board.show:
            control_loop(pygame.event.get())

        time.sleep(1)
        combos, cleared, moves = agent.play()

        print 'Step: {} Combos: {} Cleared: {} Moves: {}'.format(
            step, combos, cleared, moves)



def train_loop(agent, n_steps=100000):
    ## History is later
    # agent.initialize_history()

    ## Just go in
    print 'Entering train_loop'
    reward = [0]
    moves = [0]
    print_iter = 500
    anneal_iter = 1000
    memory_iter = 10000
    for step in range(n_steps):
        if agent.board.show:
            control_loop(pygame.event.get())

        if step % print_iter == 0:
            step_time = time.time()
            print ''*5
            print '-----------------------------------------'
            r_step, moves_step = agent.train(verbose=True)
            print 'Step: {} mean_end_reward: {} mean_moves: {} epsilon: {} time: {}'.format(
                step, np.mean(reward), np.mean(moves), agent.epsilon, time.time()-step_time)
        else:
            r_step, moves_step = agent.train()


        ## E-greedy exploration
        if step % anneal_iter == 0:
            # agent.epsilon *= 500./(step+1)
            agent.epsilon *= 0.9
            agent.epsilon = max(0.05, agent.epsilon)

        ## Bayesian exploration
        if step % anneal_iter == 0:
            agent.epsilon *= 1.15
            agent.epsilon = min(0.9, agent.epsilon)

        # agent.epsilon =  1. / (step+1)

        reward.append(r_step)
        moves.append(moves_step)


if __name__ == '__main__':
    shape = [4,5]
    moves = 50
    board = puzzle.Board(shape=shape, max_moves=moves, show=False)
    # agent = agents.RandomAgent(board, n_moves=10)

    # agent = agents.DeepQAgent(board, n_moves=20, batch_size=16, memory=5400)
    agent = agents.DoubleQAgent(board, n_moves=moves, batch_size=32, memory=7200)

    print 'Max moves: ', agent.n_moves
    agent.observe()
    train_loop(agent)

    ## Replace board so we can watch some play
    board = puzzle.Board(shape=shape, max_moves=moves, show=True)
    agent.swap_board(board)
    run_loop(agent)
