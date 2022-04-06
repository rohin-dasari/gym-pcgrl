import gym
import ray
from ray import tune

from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import gym_pcgrl
from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv
from gym_pcgrl.parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


from marl_model import CustomFeedForwardModel as Model


def env_maker(config):
    return MARL_CroppedImagePCGRLWrapper('MAPcgrl-binary-narrow-v0', 7)

def gen_policy(obs_space, act_space):
    config = {
            'model': {'custom_model': 'main_model'},
            'gamma': 0.95
            }
    return (None, obs_space, act_space, config)

def train():
    ray.init(num_cpus=None)
    tune.register_env('MAPcgrl-binary-narrow-v0', lambda config: ParallelPettingZooEnv(env_maker(config)))

    env = env_maker({})
    obs_space = env.observation_spaces['empty']
    action_space = env.action_spaces['empty']
    ModelCatalog.register_custom_model('main_model', Model)

    policy_mapping_fn = lambda agent: f'policy_{agent}'

    policies = {f'policy_{agent}': gen_policy(obs_space, action_space) for agent in env.possible_agents}
    config = {
            'env': 'MAPcgrl-binary-narrow-v0',
            'env_config': {'prob': 'binary', 'num_agents': None, 'binary_actions': True},
            'num_gpus': 0,
            'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_mapping_fn
                },
            'model': {},
            'framework': 'torch',
            'output': 'experiments'
            }
    results = tune.run('PPO', config=config, verbose=1)


if __name__ == '__main__':
    train()
    ray.shutdown()

