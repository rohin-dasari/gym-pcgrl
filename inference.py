"""
Run a trained agent and get generated maps
"""
import json
from pathlib import Path
import model
import numpy as np
import pandas as pd
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import os
import imageio


def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    evaldir = Path('eval')
    evaldir.mkdir(exist_ok=True)

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    success_count = 0
    for i in range(kwargs.get('trials', 1)):
        init_level = np.loadtxt(Path('binary_levels', f'level_{i}.txt'))
        trial_dir = Path(evaldir, str(i))
        os.makedirs(trial_dir, exist_ok=True)
        obs = env.reset()
        base_env = env.envs[0]
        base_env.set_state(init_level, {'x': 0, 'y': 0})
        frames = []
        frames.append(env.render(mode='rgb_array'))
        dones = False
        actions = []
        infos = []
        while not dones:
            action, _ = agent.predict(obs)
            human_action = base_env.get_human_action(action[0])
            positions = base_env.get_positions()
            action_metadata = {
                    'agent': 'default',
                    'action': action,
                    'human': human_action,
                    'xpos': positions['x'],
                    'ypos': positions['y'],
                    'timestep': base_env.get_iteration()
                    }
            actions.append(action_metadata)
            obs, _, dones, info = env.step(action)
            infos.append(info)
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
        success_count += int(env.envs[0].check_success())
        # save success
        with open(Path(trial_dir, 'success.json'), 'w+') as f:
            f.write(json.dumps({'success': int(env.envs[0].check_success())}))
        # save renderings
        imageio.imwrite(Path(trial_dir, 'initial_frame.png'), frames[0])
        imageio.v2.mimwrite(Path(trial_dir, 'animation.gif'), frames)
        imageio.imwrite(Path(trial_dir, 'final_frame.png'), frames[-2])
        # save actions
        actions_df = pd.DataFrame(actions)
        actions_df.to_csv(Path(trial_dir, 'actions.csv'))
        # save info
        pd.DataFrame(infos).to_csv(Path(trial_dir, 'infos.csv'))
    print('n_success: ', success_count)


################################## MAIN ########################################
game = 'binary'
representation = 'narrow'
#model_path = 'runs/binary_narrow_2_log/best_model.pkl'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.4,
    'trials': 40,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
