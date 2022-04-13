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
from gym_pcgrl.utils import parse_config


def train(config_path):
    config = parse_config(config_path)
    # 25000 training iterations = 100M timesteps
    results = tune.run(
            'PPO',
            config=config,
            #stop={'training_iteration': 25000},
            stop={'training_iteration': 4},
            checkpoint_at_end=True,
            #checkpoint_freq=1000
            checkpoint_freq=1
            )


if __name__ == '__main__':
    #config_path = 'configs/binary_actions_maze.yaml'
    config_path = 'configs/full_actions_maze.yaml'
    train(config_path)

