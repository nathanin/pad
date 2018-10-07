from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari

import sys
sys.path.insert(0, '..')
from envs import PadEnv

def main():
    logger.configure()
    env = PadEnv()
    print('Observation shape', env.observe().shape)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 2, 2), (64, 2, 1), (64, 2, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
    )

    model.save('pong_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
