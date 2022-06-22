"""
Load trained RLLIB model and use it on a sample environment
"""

from pathlib import Path
from uuid import uuid4

from ray import tune
from gym_pcgrl.utils import parse_config
import ray.rllib.agents.ppo as ppo
from gym_pcgrl.utils import env_maker_factory
import pandas as pd
import json
from tqdm import tqdm
import imageio
import numpy as np



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

def collect_action_metadata(env, actions):
    # collect timestep, agent, action, x_pos, y_pos
    data = []
    agent_positions = env.get_agent_positions()
    for agent, action in actions.items():
        agent_pos = agent_positions[agent]
        metadata = {
                    'agent': agent,
                    'action': action,
                    'human': env.get_human_action(agent, action),
                    'xpos': agent_pos['x'],
                    'ypos': agent_pos['y'],
                    'timestep': env.get_iteration(),
                }
        data.append(metadata)
    return data

def rollout(env, trainer, policy_mapping_fn, render=True, initial_level=None):
    done = False
    obs = env.reset()
    rawobs = env.set_state(initial_level=initial_level, initial_positions=env.get_agent_positions())
    obs = env.transform_observations(rawobs)

    #imageio.imsave('initial_img.png', env.render(mode='rgb_array'))
    frames = []
    actions = []
    infos = []
    action_data = []
    infos.append(env.get_metadata())
    frames.append(env.render(mode='rgb_array'))
    while not done:
        actions = get_agent_actions(trainer, obs, policy_mapping_fn)
        action_metadata = collect_action_metadata(env, actions)
        obs, rew, done, info = env.step(actions)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action_data.extend(action_metadata)
        infos.append(env.get_metadata())
        done = done['__all__']
    #imageio.mimsave('animation.gif', frames)
    #imageio.imsave('final_img.png', frame)
    return {
            'success': env.check_success(),
            'frames': frames,
            'actions': action_data,
            'info': infos
            }
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
    max_checkpoint_name = None
    for checkpoint in checkpoints_iter(experiment_path):
        # get number after underscore in checkpoint
        checkpoint_num = checkpoint.name.split('_')[1].lstrip('0')
        checkpoint_name = Path(checkpoint, f'checkpoint-{checkpoint_num}')
        trainer = restore_trainer(checkpoint_name, config)
        iteration = trainer._iteration
        # look up iteration in progress dataframe
        trainer_performance = progress.loc[progress['training_iteration'] == iteration]
        trainer_reward = trainer_performance['episode_reward_mean'].values[0]
        if trainer_reward > max_episode_reward:
            max_episode_reward = trainer_reward
            max_checkpoint = trainer
            max_checkpoint_name = checkpoint
    print(f'Loaded from checkpoint: {max_checkpoint_name}')
    return max_checkpoint

def get_latest_checkpoint(experiment_path, config):
    progress = pd.read_csv(Path(experiment_path, 'progress.csv'))
    
    latest_training_iter = float('-inf')
    latest_checkpoint = None
    latest_checkpoint_name = None
    for checkpoint in checkpoints_iter(experiment_path):
        checkpoint_num = checkpoint.name.split('_')[1].lstrip('0')
        checkpoint_name = Path(checkpoint, f'checkpoint-{checkpoint_num}')
        trainer = restore_trainer(checkpoint_name, config)
        iteration = trainer._iteration
        if iteration > latest_training_iter:
            latest_training_iter = iteration
            latest_checkpoint = trainer
            latest_checkpoint_name = checkpoint
    print(f'Loaded from checkpoint: {latest_checkpoint_name}')
    return trainer

def load_level(lvl_dir, lvl_id):
    with open(Path(lvl_dir, f'level_{lvl_id}.txt'), 'r') as f:
        return np.loadtxt(f)


def save_metrics(results, logdir, level_id):
    # save metrics in an organized directory structure
    logdir = Path(logdir)
    logdir.mkdir(exist_ok=True)
    leveldir = Path(logdir, level_id)
    leveldir.mkdir(exist_ok=True)
    # save success data
    with open(Path(leveldir, 'success.json'), 'w+') as f:
        f.write(json.dumps({'success': bool(results['success'])}))
    # save action data
    actions_df = pd.DataFrame(results['actions'])
    actions_df.to_csv(Path(leveldir, 'actions.csv'))
    # save info data
    info_df = pd.DataFrame(results['info'])
    info_df.to_csv(Path(leveldir, 'info.csv'))
    # save renderings
    frames = results['frames']
    imageio.imsave(Path(leveldir, 'initial_map.png'), frames[0])
    imageio.imsave(Path(leveldir, 'final_map.png'), frames[-1])
    # save gifs
    imageio.mimsave(Path(leveldir, 'frames.gif'), frames)


def collect_metrics(config, checkpoint_loader_type, experiment_path,  out_path, n_trials=40, lvl_dir=None):
    n_success = 0
    checkpoint_loader = get_checkpoint_loader(checkpoint_loader_type)
    trainer = checkpoint_loader(experiment_path, config)
    env = build_env(config['env'], config['env_config'])
    policy_mapping_fn = config['multiagent']['policy_mapping_fn']
    for i in tqdm(range(n_trials)):
        if lvl_dir is None:
            initial_level=None
        else:
            initial_level = load_level(lvl_dir, i)
        results = rollout(env, trainer, policy_mapping_fn, initial_level=initial_level)
        n_success += results['success']
        save_metrics(results, out_path, str(i))
    # metrics returned by this thing:
    # success rate: {n_success, n_trials}
    # actions: [[timestep, agentid, x_pos, y_pos, action]]
    # save evaluation metadata
    metadata = {
            'success_rate': n_success/n_trials,
            'success_count': n_success,
            'n_trials': n_trials,
            'checkpoint_loader': checkpoint_loader

            }
    return n_success

def get_checkpoint_loader(checkpoint_loader_name):
    if checkpoint_loader_name == 'best':
        return get_best_checkpoint
    elif checkpoint_loader_name == 'latest':
        return get_latest_checkpoint
    else:
        raise ValueError('Script only supports selecting the latest or best checkpoints')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--experiment_path'
            '-e',
            dest='experiment_path',
            type=str,
            required=True
            )

    parser.add_argument(
            '--checkpoint_loader',
            dest='checkpoint_loader',
            type=str,
            default='best',
            help='accepts args from the set [latest, best]. Allows user to choose which checkpoint to load',
            required=True
            )

    parser.add_argument(
            '--config',
            '-c',
            dest='config_path',
            type=str,
            help='path to configuration file used during training',
            required=True

            )

    parser.add_argument(
            '--out_path',
            '-o',
            dest='out_path',
            type=str,
            default=None
            )

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = Path(args.experiment_path, f'eval_{uuid4()}')

    print(f'Writing Evaluation Results to: {args.out_path}')

    #config_path = 'configs/full_actions_maze.yaml'
    #config_path = 'configs/binary_actions_maze.yaml'
    config_path = args.config_path
    lvl_dir = 'binary_levels'
    config = parse_config(config_path)['rllib_config']
    config['env_config']['random_tile'] = False
    config['explore'] = False
    #experiment_path = '/home/rohindasari/ray_results/PPOTrainer_MAPcgrl-binary-narrow-v0_2022-04-20_14-36-22sbcpie2f'

    success_count = collect_metrics(
            config,
            args.checkpoint_loader,
            args.experiment_path,
            args.out_path,
            n_trials=40,
            lvl_dir=lvl_dir
            )
    print(f'Success Rate: {success_count}')

