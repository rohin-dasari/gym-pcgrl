import pytest
import imageio
from copy import deepcopy
import numpy as np
import gym
import gym_pcgrl

@pytest.fixture
def binary_env():
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
    return env

@pytest.fixture
def full_env():
    num_agents = 3
    env = gym.make('MAPcgrl-binary-narrow-v0', num_agents=num_agents, binary_actions=False)
    return env

def test_binary_env_agent_step(binary_env):
    """
    Perform a single step through a binary environment
    """
    
    # GIVEN
    env = binary_env
    env.reset()
    action = 1 # tell the currently selected agent to place a tile
    first_agent = env.agent_selection
    agent_obs = env.observe_current_agent()
    agent_x, agent_y = agent_obs['pos']

    # WHEN
    env.step(action)

    # THEN
    # test that change was made in correct place
    assert env.observations[first_agent]['map'][agent_y][agent_x] \
            == env.get_tile_map()[first_agent]
    # test that agent selection was update
    assert env.agent_selection != first_agent
    # test that reward is not set until last agent goes
    assert env.rewards['empty'] == 0


def test_set_agent_level_and_position():
    pass


def test_against_single_agent_env():
    """
    (1) Create a standard single agent environment
    (2) Create a multi-agent environment with a single agent under the full
    action space
    (3) Set the initial level and positions of the maps
    (4) If the same action is applied to each environment, then the
    environments should behave the same, e.g., the same positions should be
    modified, the same rewards should be received, etc.
    """

    # GIVEN
    # - a multiagent environment with a single agent and a standard single
    # agent environment with the same initial conditions

    # create a standard single agent environment
    single_agent_env = gym.make('binary-narrow-v0')

    # create a multi-agent environment with a single agent
    multi_agent_env = gym.make(
            'MAPcgrl-binary-narrow-v0',
            num_agents=1,
            binary_actions=False
            )

    # Load initial level string
    with open('./sample_binary_level.txt', 'r') as level_file:
        initial_level = np.loadtxt(level_file)
    # Set level for multi-agent environment
    multi_obs = multi_agent_env.reset(initial_level=initial_level, initial_positions={0: {'x': 0, 'y': 0}})
    multi_obs = {0: multi_obs}

    # Set level for single-agent environment
    single_agent_env.reset(initial_level=initial_level, initial_position={'x': 0, 'y': 0})


    # WHEN - the same action is applied to both environments, the same state
    # and reward should be observed for both environments
    # 50 timesteps
    multi_agent_frames = []
    single_agent_frames = []
    for i in range(50):
        # confirm that environment state is the same between the two environments
        np.testing.assert_array_equal(multi_agent_env.get_map(), single_agent_env.get_map())

        assert multi_agent_env.get_agent_positions()[0] == single_agent_env.get_agent_position()
        assert multi_agent_env.get_rep_stats() == single_agent_env.get_rep_stats()

        # apply the same action
        multi_obs, multi_rewards, multi_dones, _ = multi_agent_env.step(1)
        img = multi_agent_env.render(mode='rgb_array')
        #frames.append(img)
        single_obs, single_rewards, single_dones, _ = single_agent_env.step(1)

        # THEN - the reward received by the agents in their respective environments
        # should be the same
        assert multi_rewards[0] == single_rewards


