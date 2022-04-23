"""
Load trained RLLIB model and use it on a sample environment
"""


from ray import tune
from gym_pcgrl.utils import parse_config
import ray.rllib.agents.ppo as ppo
from gym_pcgrl.utils import env_maker_factory
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import imageio



def restore_trainer(checkpoint_path, config):

    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(str(checkpoint_path))
    return trainer

def build_env(env_name, env_config):
    env_maker = env_maker_factory(env_name)
    env = env_maker(env_config)
    return env


def get_agent_actions(trainer, observations, policy_mapping_fn):
    actions = {}
    for agent_id, agent_obs in observations.items():
        policy_id = policy_mapping_fn(agent_id)
        actions[agent_id] = trainer.compute_single_action(agent_obs, policy_id=policy_id)
    return actions

def rollout(env, trainer, policy_mapping_fn, render=True):
    done = False
    obs = env.reset()
    imageio.imsave('initial_img.png', env.render(mode='rgb_array'))
    print(env.get_info())
    frames = []
    while not done:
        actions = get_agent_actions(trainer, obs, policy_mapping_fn)
        obs, rew, done, info = env.step(actions)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        done = done['__all__']
    print(env.get_info())
    imageio.mimsave('animation.gif', frames)
    imageio.imsave('final_img.png', frame)
    return env.check_success()


def checkpoints_iter(experiment_path):
    experiment_path = Path(experiment_path)
    checkpoints = filter(lambda f: 'checkpoint' in f.name, experiment_path.iterdir())
    return checkpoints

def get_best_checkpoint(experiment_path, config):
    # load progress.csv
    progress = pd.read_csv(Path(experiment_path, 'progress.csv'))

    max_episode_reward = float('-inf')
    max_checkpoint = None
    for checkpoint in checkpoints_iter(experiment_path):
        # get number after underscore in checkpoint
        checkpoint_num = checkpoint.name.split('_')[1].lstrip('0')
        checkpoint_name = Path(checkpoint, f'checkpoint-{checkpoint_num}')
        trainer = restore_trainer(checkpoint_name, config)
        timestep = trainer._timesteps_total
        # look up timestep in progress dataframe
        trainer_performance = progress.loc[progress['timesteps_total'] == timestep]
        trainer_reward = trainer_performance['episode_reward_mean'].values[0]
        if trainer_reward > max_episode_reward:
            max_episode_reward = trainer_reward
            max_checkpoint = trainer
        else:
            del trainer

    return max_checkpoint


def get_success_rate(config, experiment_path,  n_trials=40):
    n_success = 0
    trainer = get_best_checkpoint(experiment_path, config)
    env = build_env(config['env'], config['env_config'])
    policy_mapping_fn = config['multiagent']['policy_mapping_fn']
    for i in tqdm(range(n_trials)):
        success = rollout(env, trainer, policy_mapping_fn)
        n_success += int(success)
        print(n_success)
        # save renderings in experiment path

#config_path = 'configs/full_actions_maze.yaml'
config_path = 'configs/binary_actions_maze.yaml'
config = parse_config(config_path)
config['env_config']['random_tile'] = False
config['explore'] = False
#experiment_path = '/home/rohindasari/ray_results/PPOTrainer_MAPcgrl-binary-narrow-v0_2022-04-20_14-36-22sbcpie2f'
experiment_path = '/home/rohindasari/ray_results/PPO_MAPcgrl-binary-narrow-v0_bfde2_00000_0_2022-04-20_17-20-49'
get_success_rate(config, experiment_path, n_trials=1)

