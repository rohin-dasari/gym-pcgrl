import pytest

import sys
sys.path.append('..')

import gym
from ray import tune
import gym_pcgrl
from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv
from ray.rllib.agents.ppo import PPOTrainer


def test_env_setup():
    env = MAPcgrlEnv(prob='zelda')
    tune.register_env('MAPcgrl-zelda-narrow-v0', lambda config: MAPcgrlEnv(prob='zelda')) # refactor to accept config args from rllib
    init_obs = env.reset()
    action_space = env.action_space
    obs_space = env.observation_space
    actions = {
                'empty': 1,
                'solid': 1,
                'player': 1,
                'key': 1,
                'door': 1,
                'bat': 1,
                'scorpion': 1,
                'spider': 1,
            }
    trainer = PPOTrainer(env='MAPcgrl-zelda-narrow-v0')


"""
test to ensure that the positions returned as a part of every new state is
valid according to the narrow representation
"""
def test_narrow_positioning():
    pass


"""
There are three possible conditions that can lead to a finished environment:
    changes >= max_changes
    iterations => max_iterations
    level reaches satisfiable quality
"""
def test_done_conditions():
    pass

#def test_pettingzoo():
#    env = MAPcgrlEnv_pettingzoo(prob='zelda')
#    env = aec_to_parallel(env)
#    env = ss.pettingzoo_env_to_vec_env_v1(env)
#    env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class='stable_baselines3')
#    model = PPO('MultiInputPolicy', env, verbose=1)
#    model.learn(total_timesteps=200000)
#    print(env)




