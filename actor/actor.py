'''

Interface for Agents to interact with the board

'''
import pygame
import numpy as np
import sys
import time


def control_loop(events):
    for event in events:
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            print('Pausing')
            paused = True
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        print('Resuming')
                        paused = False


def run_loop(agent, n_steps=25):
    print('Entering run_loop')
    epsilon_hold = agent.epsilon

    if agent.sample_mode == 'e_greedy':
        agent.epsilon = 0.0
    elif agent.sample_mode == 'bayes':
        agent.epsilon = 1.0

    for step in range(n_steps):
        if agent.board.show:
            control_loop(pygame.event.get())

        time.sleep(0.25)
        combos, cleared, moves = agent.play()

        print('Step: {} Combos: {} Cleared: {} Moves: {}'.format(
            step, combos, cleared, moves))

    ## Reinstate eps
    agent.epsilon = epsilon_hold


def train_loop(agent, n_steps=10000):
    ## History is later
    # agent.initialize_history()

    ## Just go in
    print('Entering train_loop')
    reward = [0]
    moves = [0]
    print_iter = history_iter = 250
    anneal_iter = 5000
    memory_iter = 10000
    loop_time = time.time()
    for step in range(n_steps):
        if agent.board.show:
            control_loop(pygame.event.get())

        if step % print_iter == 0:
            print('\n'*5)
            print('-----------------------------------------')
            r_step, moves_step = agent.train(verbose=True)
            try:
                print('Step: {} mean_end_reward: {} mean_moves: {} epsilon: {} time: {}'.format(
                    step, np.mean(reward[step-1000:]),
                    np.mean(moves[step-1000:]),
                    agent.epsilon, time.time()-loop_time))
            except:
                print('Step: {} mean_end_reward: {} mean_moves: {} epsilon: {} time: {}'.format(
                    step, np.mean(reward),
                    np.mean(moves),
                    agent.epsilon, time.time()-loop_time))

            print('Step: {} memory contents summary:'.format(step))
            agent.history._print_memory()
            print('-----------------------------------------')
            loop_time = time.time()
        else:
            r_step, moves_step = agent.train()

        if step % anneal_iter == 0:
            print('step: {} annealing epsilon'.format(step))
            if agent.sample_mode == 'e_greedy':
                ## E-greedy exploration
                agent.epsilon *= 0.9
                agent.epsilon = max(0.1, agent.epsilon)
            elif agent.sample_mode == 'bayes':
                ## Bayesian exploration
                agent.epsilon *= 1.15
                agent.epsilon = min(0.8, agent.epsilon)
            print('eps = {}'.format(agent.epsilon))


        reward.append(r_step)
        moves.append(moves_step)


# if __name__ == '__main__':
#     shape = [5,6]
#     moves = 100
#     board = puzzle.Board(shape=shape, max_moves=moves, show=False)
#
#     agent = agents.DoubleQAgent(board,
#         n_moves=moves,
#         batch_size=512,
#         memory=10000,
#         sample_mode='e_greedy',
#         reward_type='combo')
#
#     print('Max moves: ', agent.n_moves)
#     agent.observe()
#     train_loop(agent)
#
#     ## Replace board so we can watch some play
#     board = puzzle.Board(shape=shape,
#         max_moves=moves,
#         show=True,
#         sleep_time=0.05)
#     agent.swap_board(board)
#     run_loop(agent)
