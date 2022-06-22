import argparse
import gym
import ray
from ray import tune
from gym_pcgrl.utils import parse_config


def train(config_path):
    parsed_configs = parse_config(config_path)
    #rllib_config = parse_config(config_path)
    #tune_config = parse_config(config_path)
    results = tune.run(
                **parsed_configs['tune_config'],
                config=parsed_configs['rllib_config']
            )
    #results = tune.run(
    #        'PPO',
    #        config=rllib_config,
    #        stop={'training_iteration': 1},
    #        mode='max',
    #        metric='episode_reward_mean',
    #        checkpoint_score_attr='episode_reward_mean',
    #        keep_checkpoints_num=3,
    #        checkpoint_freq=1,
    #        checkpoint_at_end=True
    #        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_path', type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)

