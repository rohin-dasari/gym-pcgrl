import pytest

import sys
sys.path.append('..')

from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv


def test_():
    env = MAPcgrlEnv(prob='zelda')
    init_obs = env.reset()
    action_space = env.action_space
    print(env._heatmaps)
    obs_space = env.observation_space
    actions = {
                'empty': 1,
                'solid': 1,
                'player': 1,
                'key': 1,
                'door': 1,
                'bat': 1,
                'scorpion': 1,
                'spider': 1,
            }
    #print(init_obs['player']['map'])
    obs, r, d, i = env.step(actions)
    #print(obs['player']['map'])
    print(r)
    print(i['changes'])

    pass





