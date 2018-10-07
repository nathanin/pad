import gym
from baselines import logger
from baselines import bench

from baselines import deepq
from baselines.common import models

import sys
sys.path.insert(0, '..')
from envs import PadEnv

def main():
    env = PadEnv()

    conv_kwargs = {'padding': 'VALID'}
    act = deepq.learn(
        env,
        # network=models.mlp(num_hidden=512, num_layers=5),
        network=models.conv_only(
            [(32,2,1),
             (64,2,1),
             (64,2,1),]),
        lr=1e-4,
        total_timesteps=int(5e6),
        buffer_size=int(5e4),
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        dueling=True,
        batch_size=256,
        learning_starts=int(5e4),
        target_network_update_freq=500,
        gamma=0.99,
        print_freq=100,
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
