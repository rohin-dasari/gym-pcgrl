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
from ray.rllib.agents.ppo import ppo
from ray.tune.logger import pretty_print
from tqdm import tqdm
import json
from pathlib import Path
import pickle

from pettingzoo.sisl import waterworld_v3
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

def checkpoint(model, result):
    """
    checkpoint the model and the results used during training for that model
    """
    checkpoint = model.save()
    checkpoint_path = Path(checkpoint).parent
    result_data = {}
    result_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'timesteps_total', 'timesteps_this_iter']
    for key in result_keys:
        result_data[key] = result[key]
    with open(Path(checkpoint_path, 'results'), 'w+') as f:
        json.dump(result_data, f)
    print(f'checkpointed model and results at {checkpoint}')


def train(config_path):
    config = parse_config(config_path)
    ### 25000 training iterations = 100M timesteps
    #max_episode_reward = float('-inf')
    ##timesteps = 1e8
    #train_timesteps = 20000
    #timesteps = 0
    #num_checkpoints = 2
    #checkpoint_frequency = train_timesteps // num_checkpoints
    #checkpoint_iter = 0
    #trainer = ppo.PPOTrainer(config=config)
    #while timesteps < train_timesteps:
    #    checkpointed = False
    #    result = trainer.train()
    #    print(pretty_print(result))
    #    reward = result['episode_reward_mean']
    #    
    #    # save best model
    #    if reward > max_episode_reward:
    #        max_episode_reward = reward
    #        checkpoint(trainer, result)
    #        checkpointed = True

    #    if not checkpointed and checkpoint_iter >= checkpoint_frequency == 0:
    #        # checkpoint model
    #        checkpoint(trainer, result)
    #        checkpoint_iter = 0
    #    timesteps += result['timesteps_this_iter']
    #    checkpoint_iter += result['timesteps_this_iter']

    ##save final model
    #checkpoint(trainer, result)

    #def env_creator(args):
    #    return PettingZooEnv(waterworld_v3.env())

    #env = waterworld_v3.env()
    #env.reset()
    #register_env("waterworld", env_creator)
    #config={
    #    # Enviroment specific
    #    "env": "waterworld",
    #    # General
    #    "num_gpus": 0,
    #    "num_workers": 1,
    #    # Method specific
    #    "multiagent": {
    #        "policies": set(env.agents),
    #        "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
    #    },
    #}

    results = tune.run(
            'PPO',
            config=config,
            stop={'training_iteration': 5},
            mode='max',
            metric='episode_reward_mean',
            checkpoint_score_attr='episode_reward_mean',
            keep_checkpoints_num=3,
            checkpoint_freq=1,
            checkpoint_at_end=True
            )
    ray.timeline(filename='timeline.json')


if __name__ == '__main__':
    config_path = 'configs/binary_actions_maze.yaml'
    #config_path = 'configs/full_actions_maze.yaml'
    train(config_path)

