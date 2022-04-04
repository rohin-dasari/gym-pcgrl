import pytest

import sys
sys.path.append('..')

import gym
import ray
from ray import tune

from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import gym_pcgrl
from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper, MARL_CroppedImagePCGRLWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

#from marl_model import Model
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as Model
from marl_model import CustomFeedForwardModel3D as Model


def env_maker(config):
    return MARL_CroppedImagePCGRLWrapper('MAPcgrl-zelda-narrow-v0', 7)

def gen_policy(obs_space, act_space):
    config = {
            'model': {'custom_model': 'main_model'},
            'gamma': 0.95
            }
    return (None, obs_space, act_space, config)

def test_env_setup():
    ray.init(num_cpus=None)
    tune.register_env('test', lambda config: ParallelPettingZooEnv(env_maker(config))) # refactor to accept config args from rllib
    #env = MAPcgrlEnv(prob='zelda')
    #init_obs = env.reset()
    #action_space = env.action_space
    #obs_space = env.observation_space
    #actions = {
    #            'empty': 1,
    #            'solid': 1,
    #            'player': 1,
    #            'key': 1,
    #            'door': 1,
    #            'bat': 1,
    #            'scorpion': 1,
    #            'spider': 1,
    #        }

    env = env_maker({})
    #env = MAPcgrlEnv(prob='zelda')
    obs_space = env.observation_spaces['empty']
    #import pdb; pdb.set_trace()
    action_space = env.action_spaces['empty']
    ModelCatalog.register_custom_model('main_model', Model)

    policy_mapping_fn = lambda agent: f'policy_{agent}'

    policies = {f'policy_{agent}': gen_policy(obs_space, action_space) for agent in env.agents}
    config = {
                'env': 'test',
                'env_config': {'prob': 'zelda'},
                'num_gpus': 0,
                'multiagent': {
                        'policies': policies,
                        'policy_mapping_fn': policy_mapping_fn
                    },
                'model': {},
                'framework': 'torch'
            }
    results = tune.run('PPO', config=config, verbose=1)

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

if __name__ == '__main__':
    test_env_setup()
    ray.shutdown()


