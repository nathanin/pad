#!/usr/bin/env python3
from stable_baselines import logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy

import sys
# sys.path.insert(0, '/home/ing/projects/pad/')
sys.path.insert(0, '..')
from envs import PadEnv

from stable_baselines.bench import Monitor

"""
About nan rewards
https://github.com/hill-a/stable-baselines/issues/24
"""

def train(env_id, num_timesteps, seed, policy):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    """

    env = Monitor(PadEnv(), './logs', allow_early_resets=True)
    env = DummyVecEnv([lambda: env for _ in range(32)])
    env = VecFrameStack(env, 4)
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    model = PPO2(policy=policy, env=env, n_steps=256, nminibatches=4, lam=0.95, gamma=0.99, noptepochs=4, 
                 ent_coef=.01, learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    try:
        model.learn(total_timesteps=num_timesteps)
    except KeyboardInterrupt:
        print('Keyboard Interrupted')

    model.save('./pad_ppo2.pkl')


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', 
                        help='Policy architecture', 
                        choices=['cnn', 'lstm', 'lnlstm', 'mlp'], 
                        default='cnn')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy)


if __name__ == '__main__':
    main()
