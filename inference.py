"""
Run a trained agent and get generated maps
"""
from pathlib import Path
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs

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

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    evaldir = Path('eval')
    evaldir.mkdir(exist_ok=True)
    for i in range(kwargs.get('trials', 1)):
        obs = env.reset()
        frames = []
        infos = []
        actions = []
        dones = False
        frames.append(env.render(mode='rgb_array'))
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            frames.append(env.render(mode='rgb_array'))
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
        success = env.check_success()
        time.sleep(0.2)

    # save success
    # save frames
    # save actions
    # save

################################## MAIN ########################################
game = 'binary'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.4,
    'trials': 1,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
