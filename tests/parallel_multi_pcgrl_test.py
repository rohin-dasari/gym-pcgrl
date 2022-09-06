import pytest
import gym
import numpy as np
import gym_pcgrl

@pytest.fixture
def binary_env():
    env = gym.make('Parallel_MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
    return env

@pytest.fixture
def full_env():
    num_agents = 3
    env = gym.make('Parallel_MAPcgrl-binary-narrow-v0', num_agents=num_agents, binary_actions=False)
    return env

@pytest.fixture
def sample_level():
    with open('./sample_binary_level.txt', 'r') as level_file:
        initial_level = np.loadtxt(level_file)
        return initial_level

"""
test to ensure that a new position is returned every time by the environment
"""
def test_default_agent_initialization(binary_env):
    init_obs = binary_env.reset()
    assert isinstance(init_obs, dict)
    assert len(init_obs) == 2

def test_custom_agent_initialization(full_env):
    init_obs = full_env.reset()
    assert isinstance(init_obs, dict)
    assert len(init_obs) == len(full_env.agents)

def test_apply_action_binary_env(binary_env):
    init_obs = binary_env.reset()
    empty_pos = init_obs['empty']['pos']
    solid_pos = init_obs['solid']['pos']
    # check that the map state observed by both agents is the same
    assert (init_obs['empty']['map'] == init_obs['solid']['map']).all()
    # set actions to add tile
    actions = {'empty': 1, 'solid': 1}
    obs, rews, dones, infos = binary_env.step(actions)
    # if the initial positions of the agents are not the same, then check to find the expected tile on the map
    # if the positions are the same, then the tile that is placed is dependent on the last agent that goes
    assert (obs['empty']['map'] == obs['solid']['map']).all()
    if not (empty_pos == solid_pos).all():
        empty_x, empty_y = empty_pos
        assert obs['empty']['map'][empty_y][empty_x] == 0
        solid_x, solid_y = solid_pos
        assert obs['solid']['map'][solid_y][solid_x] == 1

#def test_apply_action_full_env(full_env):
#    init_obs = full_env.reset()
#    agent0_pos = init_obs[0]['pos']
#    agent1_pos = init_obs[1]['pos']
#    agent2_pos = init_obs[1]['pos']
#    # check that the map state observed by both agents is the same
#    assert (init_obs[0]['map'] == init_obs[1]['map']).all()
#    # set actions to add tile
#    actions = {0: 1, 1: 1, 2: 1}
#    obs, rews, dones, infos = full_env.step(actions)
#    # If the initial positions of the agents are not the same, then check to
#    # find the expected value.
#    # It is possible for the stored positions to not have the expected values
#    # if the positions are the same.
#    assert (obs[0]['map'] == obs[1]['map']).all()
#    if not (empty_pos == solid_pos).all():
#        empty_x, empty_y = empty_pos
#        assert obs[0]['map'][empty_y][empty_x] == 0
#        solid_x, solid_y = solid_pos
#        assert obs[1]['map'][solid_y][solid_x] == 0



def test_set_state(binary_env):
    from gym_pcgrl.envs.reps.representation import Representation
    from gym_pcgrl.envs.probs import PROBLEMS
    from gym_pcgrl.envs.helper import get_int_prob, get_string_map
    import numpy as np

    prob_name = 'binary'
    rep = Representation()
    prob = PROBLEMS[prob_name]()
    tile_probs = get_int_prob(prob._prob, prob.get_tile_types())
    rep.reset(prob._width, prob._height, tile_probs)
    level = rep._map

    init_obs = binary_env.reset(initial_level=level)
    np.testing.assert_equal(level, binary_env.get_map())
    # also assert that rep stats are the same


def test_human_actions(binary_env):
    init_obs = binary_env.reset()
    tile = 'empty'
    action = 1
    #action = env.get_human_action(tile, action)
    action = binary_env.get_human_action(tile, action)
    assert action == f'place {tile}'



def test_against_single_agent_env(sample_level):
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
            'Parallel_MAPcgrl-binary-narrow-v0',
            num_agents=1,
            binary_actions=False
            )

    # Load initial level string
    # Set level for multi-agent environment
    multi_obs = multi_agent_env.reset(initial_level=sample_level, initial_positions={0: {'x': 0, 'y': 0}})
    multi_obs = {0: multi_obs}

    # Set level for single-agent environment
    single_agent_env.reset(initial_level=sample_level, initial_position={'x': 0, 'y': 0})


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
        multi_obs, multi_rewards, multi_dones, _ = multi_agent_env.step({0: 1})
        img = multi_agent_env.render(mode='rgb_array')
        #frames.append(img)
        single_obs, single_rewards, single_dones, _ = single_agent_env.step(1)

        # THEN - the reward received by the agents in their respective environments
        # should be the same
        assert multi_rewards[0] == single_rewards


#"""
#There are three possible conditions that can lead to a finished environment:
#    changes >= max_changes
#    iterations => max_iterations
#    level reaches satisfiable quality
#"""
#def test_done_conditions():
#    pass
#
#
#def test_reward():
#    env = gym.make('Parallel_MAPcgrl-binary-narrow-v0', num_agents=None, binary_actions=True)
#    init_obs = env.reset()
#    empty_pos = init_obs['empty']['pos']
#    solid_pos = init_obs['solid']['pos']
#    # check that the map state observed by both agents is the same
#    assert (init_obs['empty']['map'] == init_obs['solid']['map']).all()
#
#    # check that the agents rewards 
#
#def test_update_heatmap():
#    pass
#
#
#def test_agent_action():
#    pass


