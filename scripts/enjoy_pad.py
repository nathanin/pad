#!/usr/bin/env python3
from stable_baselines import logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy

import sys
import time
sys.path.insert(0, '..')
from envs import PadEnvRender

from stable_baselines.bench import Monitor

"""
About nan rewards
https://github.com/hill-a/stable-baselines/issues/24
"""

def test(env_id, seed, policy):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    """

    # if 'lstm' in policy:
    #     print('LSTM policies not supported for drawing')
    #     return 1
    env = DummyVecEnv([PadEnvRender for _ in range(1)]) # Need for lstm
    # else:
    #     env = PadEnvRender()

    env = VecFrameStack(env, 4)
    model = PPO2.load('./pad_ppo2.pkl', env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            done = done.any()
            episode_rew += rew
            time.sleep(1/24.)

def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    logger.configure()
    test(args.env, seed=args.seed, policy=args.policy)


if __name__ == '__main__':
    main()
