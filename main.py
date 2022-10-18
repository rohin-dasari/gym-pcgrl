import argparse
import gym
import ray
from ray import tune
from gym_pcgrl.utils import parse_config


def train(config_path):
    parsed_configs = parse_config(config_path)
    results = tune.run(
                **parsed_configs['tune_config'],
                config=parsed_configs['rllib_config'],
                log_to_file=True
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_path', type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)

