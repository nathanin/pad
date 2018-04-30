import numpy as np

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
        print('Dumping memory')
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
            print('History refreshed')
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

        print('--------- Experience Summary ----------')
        print('Experience history with {} entries'.format(self.n))
        print('Actions distribution:\n\tLeft: {:2.3f}\n\tUp: {:2.3f}\n\tRight: {:2.3f}\n\tDown: {:2.3f}\n\tEnd: {:2.3f}'.format(
            actions[0],
            actions[1],
            actions[2],
            actions[3],
            actions[4], ))
        print('Rewards: mean: {} min: {} max: {}'.format(
            np.mean(rewards), rewards.min(), rewards.max() ))
        print('Rewards non-0: {} mean: {} min: {} max: {}'.format(
            (rewards!=0).sum(),
            np.mean(rewards[rewards!=0]),
            rewards[rewards!=0].min(),
            rewards[rewards!=0].max() ))
        try:
            print('Terminal actions: {}'.format((is_ends == True).sum()))
        except:
            print('Teminal actions: None??')
