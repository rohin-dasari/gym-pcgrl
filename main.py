import gym
import ray
from ray import tune

from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import gym_pcgrl
from gym_pcgrl import models
from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv
from gym_pcgrl.parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


from gym_pcgrl.models import CustomFeedForwardModel as Model


def env_maker(config):
    return MARL_CroppedImagePCGRLWrapper('MAPcgrl-binary-narrow-v0', 28, **config)

def gen_policy(obs_space, act_space):
    config = {
            'model': {'custom_model': 'CustomFeedForwardModel'},
            'gamma': 0.95
            }
    return (None, obs_space, act_space, config)

# convert train function to load config from config path
def train():
    ray.init(num_cpus=None)
    tune.register_env('MAPcgrl-binary-narrow-v0', lambda config: ParallelPettingZooEnv(env_maker(config)))

    env_config = {'num_agents': None, 'binary_actions': True}
    env = env_maker(env_config)
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_spaces[sample_agent]
    action_space = env.action_spaces[sample_agent]
    ModelCatalog.register_custom_model('CustomFeedForwardModel', Model)

    policy_mapping_fn = lambda agent: f'policy_{agent}'

    policies = {f'policy_{agent}': gen_policy(obs_space, action_space) for agent in env.possible_agents}
    config = {
            'env': 'MAPcgrl-binary-narrow-v0',
            'env_config': env_config,
            'num_gpus': 0,
            'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_mapping_fn
                },
            'framework': 'torch',
            'output': 'experiments',
            'render_env': True
            }
    # 25000 training iterations = 100M timesteps
    results = tune.run(
            'PPO',
            config=config,
            stop={'training_iteration': 25000},
            checkpoint_at_end=True,
            checkpoint_freq=1000
            )


if __name__ == '__main__':
    train()
    ray.shutdown()

