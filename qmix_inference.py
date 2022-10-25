"""
Load trained RLLIB model and use it on a sample environment
"""

from pathlib import Path
from uuid import uuid4

from ray import tune
from gym_pcgrl.utils import parse_config, load_config
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.qmix as qmix
from gym_pcgrl.utils import env_maker_factory
import pandas as pd
import json
from tqdm import tqdm
import imageio
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt


from qmix_test import make_grouped_env, register_grouped_env


def restore_trainer(checkpoint_path, config):
    #trainer = qmix.QMixTrainer(config=config)
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(str(checkpoint_path))
    return trainer

def build_env(env_name, env_config, is_parallel):
    env_maker = env_maker_factory(env_name, is_parallel)
    env = env_maker(env_config)
    return env


def get_agent_actions(trainer, observations, policy_mapping_fn):
    actions = {}
    for agent_id, agent_obs in observations.items():
        policy_id = policy_mapping_fn(agent_id)
        actions[agent_id] = trainer.compute_single_action(agent_obs, policy_id=policy_id)
    return actions

def qmix_get_agent_actions(trainer, observations):
    actions = {}
    #import pdb; pdb.set_trace()
    # observations must be passed as a tuple
    actions = trainer.compute_single_action(tuple(observations.values()))
    return {agent: action for agent, action in zip(observations.keys(), actions)}

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

def rollout(env, trainer, policy_mapping_fn=None, render=True, initial_level=None):
    done = False
    obs = env.reset()
    agent_positions = {}

    rawobs = env.set_state(
                initial_level = initial_level,
                initial_positions = env.get_agent_positions()
            )
    obs = env.transform_observations(rawobs)

    frames = []
    actions = []
    infos = []
    action_data = []
    infos.append(env.get_metadata())
    frames.append(env.render(mode='rgb_array'))
    initial_map = env.get_map()
    while not done:
        #actions = qmix_get_agent_actions(trainer, obs)
        #actions['empty'] = 0 # hardcode solid agent to no-op
        actions = get_agent_actions(trainer, obs, policy_mapping_fn)
        action_metadata = collect_action_metadata(env, actions)
        obs, rew, done, info = env.step(actions)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action_data.extend(action_metadata)
        infos.append(env.get_metadata())
        done = done['__all__']
    return {
            'success': env.check_success(),
            'initial_map': initial_map,
            'final_map': env.get_map(),
            'frames': frames,
            'actions': action_data,
            'info': infos,
            'agent_heatmaps': env.get_agent_heatmaps(), # spatial information about changes
            'tile_heatmaps': env.get_tile_heatmaps(), # spatial information about changes
            'legend': env.get_agent_color_mapping(),
            'cumulative_rewards': env.get_cumulative_rewards()
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
    return latest_checkpoint

def get_checkpoint_by_name(experiment_path, checkpoint_name, config):
    checkpoint_path = Path(
                experiment_path,
                checkpoint_name,
                f"checkpoint-{checkpoint_name.split('_')[1].lstrip('0')}"
            )
    trainer = restore_trainer(checkpoint_path, config)
    print(f'Loaded from checkpoint: {checkpoint_name}')
    return trainer

def load_level(lvl_dir, lvl_id):
    with open(Path(lvl_dir, f'level_{lvl_id}.txt'), 'r') as f:
        return np.loadtxt(f)


def save_heatmaps(logdir, heatmaps):

    logdir.mkdir(exist_ok=True)
    for name, heatmap in heatmaps.items():
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('changes', rotation=-90, va="bottom")
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        fig.savefig(Path(logdir, f'{name}_heatmap.png'), dpi=400)
        plt.close(fig) # close figure to prevent memory issues


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

    #save heatmaps
    agent_heatmap_dir = Path(leveldir, 'agent_heatmaps')
    tile_heatmap_dir = Path(leveldir, 'tile_heatmaps')
    save_heatmaps(agent_heatmap_dir, results['agent_heatmaps'])
    save_heatmaps(tile_heatmap_dir, results['tile_heatmaps'])

    # save legend from figure
    with open(Path(leveldir, 'rendering_legend.json'), 'w+') as f:
        f.write(json.dumps(results['legend']))

    # save numpy array of level maps
    np.savetxt(Path(leveldir, 'initial_map.txt'), results['initial_map'])
    np.savetxt(Path(leveldir, 'final_map.txt'), results['final_map'])

    # save cumulative_rewards
    with open(Path(leveldir, 'cumulative_rewards.json'), 'w+') as f:
        f.write(json.dumps(results['cumulative_rewards']))

def prepare_config_for_inference(config_path):
    rllib_config = parse_config(config_path)['rllib_config']
    rllib_config['env_config']['random_tile'] = False
    rllib_config['explore'] = False
    return rllib_config

def qmix_get_unwrapped_env(env):
    # wow, I love gym and rllib :)!
    return env.to_base_env()._unwrapped_env.env

def collect_metrics(
        config_path,
        checkpoint_loader_type,
        experiment_path,
        out_path,
        n_trials=40,
        lvl_dir=None):

    n_success = 0
    config = load_config(config_path) # why am I loading config twice?
    rllib_config = prepare_config_for_inference(config_path)

    #config = {
    #        "rollout_fragment_length": 4,
    #        "train_batch_size": 32,
    #        "exploration_config": {
    #            "final_epsilon": 0.0,
    #        },
    #        "num_workers": 0,
    #        "mixer": 'qmix',
    #        "env_config": {
    #            "binary": True,
    #            'random_tile': False,
    #            'max_iterations': 500
    #        },
    #        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #        "num_gpus": 0,
    #        'model': {
    #            'custom_model': 'CustomFeedForwardModel'
    #            },
    #        'env': 'grouped_env',
    #        'explore': False
    #        }

    #env = qmix_get_unwrapped_env(
    #        make_grouped_env(
    #                'Parallel_MAPcgrl-binary-narrow-v0',
    #                28,
    #                **config['env_config']
    #            )
    #        )

    #register_grouped_env()
    trainer = load_checkpoint(
            checkpoint_loader_type,
            experiment_path,
            rllib_config
            #config
            )

    env = build_env(
            rllib_config['env'],
            rllib_config['env_config'],
            config['is_parallel']
            )
    #env = build_env(
    #        'Parallel_MAPcgrl-binary-narrow-v0', config['env_config'], True
    #        )

    policy_mapping_fn = rllib_config['multiagent']['policy_mapping_fn']
    for i in tqdm(range(n_trials)):
        if lvl_dir is None:
            initial_level=None
        else:
            initial_level = load_level(lvl_dir, i)
        results = rollout(env, trainer, policy_mapping_fn, initial_level=initial_level)
        #results = rollout(env, trainer, None, initial_level=initial_level)
        n_success += results['success']
        save_metrics(results, out_path, str(i))

    metadata = {
            'success_rate': float(n_success/n_trials),
            'success_count': int(n_success),
            'n_trials': n_trials,
            'checkpoint_loader': checkpoint_loader_type,
            'trainer_iteration': int(trainer._iteration)
            }
    with open(Path(out_path, 'metadata.json'), 'w+') as f:
        f.write(json.dumps(metadata))
    return n_success

def load_checkpoint(checkpoint_loader_name, experiment_path, config):
    if checkpoint_loader_name == 'best':
        return get_best_checkpoint(experiment_path, config)
    elif checkpoint_loader_name == 'latest':
        return get_latest_checkpoint(experiment_path, config)
    else:
        return get_checkpoint_by_name(experiment_path, checkpoint_loader_name, config)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--experiment_path',
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
            help='accepts args from the set [latest, best]. A user can also pass in a specefic checkpoint name',
            required=False
            )

    parser.add_argument(
            '--config',
            '-c',
            dest='config_path',
            type=str,
            help='path to configuration file used during training',
            required=False
            )

    parser.add_argument(
            '--out_path',
            '-o',
            dest='out_path',
            type=str,
            default=None
            )

    parser.add_argument(
        '--n_trials',
        '-n',
        dest='n_trials',
        type=int,
        default=40
    )

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = Path(
                args.experiment_path,
                f'eval_{args.checkpoint_loader}_{uuid4()}'
                )

    print(f'Writing Evaluation Results to: {args.out_path}')

    #config_path = 'configs/full_actions_maze.yaml'
    #config_path = 'configs/binary_actions_maze.yaml'
    #config_path = args.config_path
    ## save a copy of the original config in the experiment location
    lvl_dir = 'binary_levels'
    #config = parse_config(config_path)['rllib_config']
    #config['env_config']['random_tile'] = False
    #config['explore'] = False
    #experiment_path = '/home/rohindasari/ray_results/PPOTrainer_MAPcgrl-binary-narrow-v0_2022-04-20_14-36-22sbcpie2f'

    success_count = collect_metrics(
            args.config_path,
            args.checkpoint_loader,
            args.experiment_path,
            args.out_path,
            n_trials=args.n_trials,
            lvl_dir=lvl_dir
            )
    print(f'Success Rate: {success_count}')
    print(f'Successfully wrote evaluation results to: {args.out_path}')





