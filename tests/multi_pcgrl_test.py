import pytest
import gym
import ray
from ray import tune

from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import gym_pcgrl
from gym_pcgrl.envs.multi_pcgrl_env import MAPcgrlEnv
from gym_pcgrl.parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv



"""
test to ensure that a new position is returned every time by the environment
"""
def test_default_agent_initialization():
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
    init_obs = env.reset()
    assert isinstance(init_obs, dict)
    assert len(init_obs) == 2

def test_custom_agent_initialization():
    num_agents = 3
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=num_agents, binary_actions=False)
    init_obs = env.reset()
    assert isinstance(init_obs, dict)
    assert len(init_obs) == num_agents

def test_apply_action_binary_actions():
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
    init_obs = env.reset()
    empty_pos = init_obs['empty']['pos']
    solid_pos = init_obs['solid']['pos']
    # check that the map state observed by both agents is the same
    assert (init_obs['empty']['map'] == init_obs['solid']['map']).all()
    # set actions to add tile
    actions = {'empty': 1, 'solid': 1}
    obs, rews, dones, infos = env.step(actions)
    # if the initial positions of the agents are not the same, then check to find the expected value
    assert (obs['empty']['map'] == obs['solid']['map']).all()
    if not (empty_pos == solid_pos).all():
        empty_x, empty_y = empty_pos
        assert obs['empty']['map'][empty_y][empty_x] == 0
        solid_x, solid_y = solid_pos
        assert obs['solid']['map'][solid_y][solid_x] == 1

def test_apply_action():
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=2, binary_actions=False)
    init_obs = env.reset()
    empty_pos = init_obs[0]['pos']
    solid_pos = init_obs[1]['pos']
    # check that the map state observed by both agents is the same
    assert (init_obs[0]['map'] == init_obs[1]['map']).all()
    # set actions to add tile
    actions = {0: 1, 1: 1}
    obs, rews, dones, infos = env.step(actions)
    # If the initial positions of the agents are not the same, then check to
    # find the expected value.
    # It is possible for the stored positions to not have the expected values
    # if the positions are the same.
    assert (obs[0]['map'] == obs[1]['map']).all()
    if not (empty_pos == solid_pos).all():
        empty_x, empty_y = empty_pos
        assert obs[0]['map'][empty_y][empty_x] == 0
        solid_x, solid_y = solid_pos
        assert obs[1]['map'][solid_y][solid_x] == 0



"""
There are three possible conditions that can lead to a finished environment:
    changes >= max_changes
    iterations => max_iterations
    level reaches satisfiable quality
"""
def test_done_conditions():
    pass


def test_reward():
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
    init_obs = env.reset()
    empty_pos = init_obs['empty']['pos']
    solid_pos = init_obs['solid']['pos']
    # check that the map state observed by both agents is the same
    assert (init_obs['empty']['map'] == init_obs['solid']['map']).all()

    # check that the agents rewards 

def test_update_heatmap():
    pass


def test_agent_action():
    pass


