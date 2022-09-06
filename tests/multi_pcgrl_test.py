import pytest
import numpy as np
import random
from tqdm import tqdm
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

@pytest.fixture
def sample_level():
    with open('./sample_binary_level.txt', 'r') as level_file:
        initial_level = np.loadtxt(level_file)
        return initial_level

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


#def test_observation_structure(binary_env):
#    env = binary_env
#    env.reset()
#    action = 1
#    obs = env.observe_current_agent()
#    print(obs)
#    newobs, _, _, _ = env.step(action)
#    print('-------------')
#    print(newobs)
#    
#    pass

def test_set_agent_level_and_position():
    pass


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
            'MAPcgrl-binary-narrow-v0',
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
        multi_obs, multi_rewards, multi_dones, _ = multi_agent_env.step(1)
        img = multi_agent_env.render(mode='rgb_array')
        #frames.append(img)
        single_obs, single_rewards, single_dones, _ = single_agent_env.step(1)

        # THEN - the reward received by the agents in their respective environments
        # should be the same
        assert multi_rewards[0] == single_rewards


def test_against_parallel_env(sample_level):
    """
    (1) Create a standard sequential multii-agent environment with two agents (binary actions)
    (2) Create a standard parallel environment with two agents (binary actions)
    (3) set the initital level and positions of each environment
    (4) IF the same action is applied to each environment, then the environments should behave the same, e.g.,
    each environment should produce the same observations and rewards
    """

    # GIVEN
    # - a standard sequential multi-agent environment with binary actions
    initial_positions = {
        'empty': {
            'x': 0,
            'y': 0
        },
        'solid': {
            'x': 0,
            'y': 1
        }
    }
    seq_env = gym.make('MAPcgrl-binary-narrow-v0').unwrapped
    seq_env.reset(initial_level=sample_level, initial_positions=initial_positions)

    # - a standard parallel multi-agent environemnt with binary actions
    par_env = gym.make('Parallel_MAPcgrl-binary-narrow-v0')
    par_env.reset(initial_level=sample_level, initial_positions=initial_positions)

    assert seq_env.get_agent_ids() == par_env.get_agent_ids()
    #print(seq_env.get_map())
    #print(par_env.get_map())
    old_map = seq_env.get_map()
    np.testing.assert_array_equal(seq_env.get_map(), par_env.get_map())
    
    # WHEN 
    # - the same actions are applied in each environment for each agent
    max_iters = 7644
    
    actions = []
    for _ in range(max_iters):
        # pick a random action
        action = {
            'empty': random.randint(0, 1),
            'solid': random.randint(0, 1)
        }
        actions.append(action)

    # step through the sequential environment
    timestep = 0

    seq_actions = []
    for i in range(max_iters):
        seq_actions.append({})

    for agent in tqdm(seq_env.agent_iter(), total=max_iters):
        obs, reward, done, info = seq_env.last()
        seq_env.step(actions[timestep][agent] if not done else None)
        seq_actions[timestep][agent] = actions[timestep][agent]
        print(agent)
        if not seq_env.agents:
            break
        if seq_env.agent_is_last(): # something seems wrong with this function
            timestep += 1
        #if 'empty' in seq_actions[timestep] and 'solid' in seq_actions[timestep]:
        #    timestep += 1
    print(seq_env.infos['__common__'])


    par_actions = []
    for t in tqdm(range(max_iters)):
        obs, reward, done, info = par_env.step(actions[t])
        par_actions.append(actions[t])
        if done['__all__']:
            break
    print(info)

    min_thing = min(timestep, len(par_actions))
    for i in range(min_thing):
        is_same = seq_actions[i] == par_actions[i]
        if not is_same:
            print(seq_actions[i], par_actions[i], i)

    np.testing.assert_array_equal(seq_env.get_map(), par_env.get_map())
    
