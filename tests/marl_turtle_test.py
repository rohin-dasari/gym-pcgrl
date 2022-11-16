from copy import deepcopy
from itertools import product
import pytest
import numpy as np
import gym
import gym_pcgrl
from gym_pcgrl.envs.reps import MARL_TurtleRepresentation
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.helper import get_int_prob
from gym_pcgrl.utils import env_maker_factory
from gym_pcgrl.envs import Parallel_MAPcgrlEnv



@pytest.mark.parametrize('action,warp', product(list(range(4)), [True, False]))
def test_movement_action(action, warp):
    prob = PROBLEMS['zelda']()
    tile_types = prob.get_tile_types()
    rep = MARL_TurtleRepresentation(
            agents = tile_types,
            tiles = tile_types,
            binary_actions = False,
            warp = warp
            )
    tile_probs = get_int_prob(prob._prob, tile_types)
    rep.reset(prob._width, prob._height, tile_probs)
    initial_positions = deepcopy(rep.agent_positions)
    direction = rep._dirs[action]
    actions = {agent: action for agent in tile_types}
    rep.update(actions)
    
    # check that the agents were moved to the correct positions
    for agent in tile_types:
        old_pos = initial_positions[agent]
        new_pos = rep.agent_positions[agent]
        is_upper_edge = old_pos['y'] == 0
        is_lower_edge = old_pos['y'] == prob._height - 1
        is_left_edge = old_pos['x'] == 0
        is_right_edge = old_pos['x'] == prob._width - 1


        if is_left_edge and direction == (-1, 0):
            assert new_pos['y'] - old_pos['y'] == 0
            if warp:
                assert new_pos['x'] == prob._width - 1
            else:
                assert new_pos['x'] == 0
        elif is_right_edge and direction == (1, 0):
            assert new_pos['y'] - old_pos['y'] == 0
            if warp:
                assert new_pos['x'] == 0
            else:
                assert new_pos['x'] == prob._width - 1

        elif is_upper_edge and direction == (0, -1):
            assert new_pos['x'] - old_pos['x'] == 0
            if warp:
                assert new_pos['y'] == prob._height - 1
            else:
                assert new_pos['y'] == 0
        elif is_lower_edge and direction == (0, 1):
            assert new_pos['x'] - old_pos['x'] == 0
            if warp:
                assert new_pos['y'] == 0
            else:
                assert new_pos['y'] == prob._height - 1
        else:

            assert new_pos['x'] - old_pos['x'] == direction[0]
            assert new_pos['y'] - old_pos['y'] == direction[1]


def test_tile_placement_binary_action_space():
    prob = PROBLEMS['zelda']()
    tile_types = prob.get_tile_types()
    rep = MARL_TurtleRepresentation(
            agents = tile_types,
            tiles = tile_types,
            binary_actions = True
            )
    tile_probs = get_int_prob(prob._prob, tile_types)
    rep.reset(prob._width, prob._height, tile_probs)
    # {0,1,2,3} actions are reserved for movement, therefore 4 corresponds to placing a tile
    actions = {agent: 4 for agent in tile_types} # make each agent place its tile
    initial_map = deepcopy(rep._map)
    rep.update(actions)
    level_map = rep._map

    for agent, pos in rep.agent_positions.items():
        x, y = pos['x'], pos['y']
        tile_idx = level_map[y][x]
        assert agent == tile_types[tile_idx]


@pytest.mark.parametrize('action', list(range(5)))
def test_pcgrl_env_setup(action):
    env = gym.make('Parallel_MAPcgrl-zelda-marl_turtle-v0', num_agents=None, binary_actions=True, max_iterations=500)
    env.reset()
    actions = {agent: action for agent in env.possible_agents}
    i = 0
    done = False
    while not done:
        obs, rew, dones, _ = env.step(actions)
        done = dones['__all__']
        i += 1
    assert i == 500

def test_pcgrl_env_with_wrapper():
    env_maker = env_maker_factory('Parallel_MAPcgrl-zelda-marl_turtle-v0', is_parallel=True)
    env_config = {
            'num_agents': None,
            'binary_actions': True,
            'max_iterations': 500
            }
    env = env_maker(env_config)
    import pdb; pdb.set_trace()


def test_rep_with_grouping():
    prob = PROBLEMS['zelda']()
    tile_types = prob.get_tile_types()
    rep = MARL_TurtleRepresentation(
            agents = tile_types,
            tiles = tile_types,
            binary_actions = False,
            warp = False,
            groups = {
                    1: ['key', 'door'],
                    2: ['scorpion', 'spider', 'bat', 'player'],
                    3: ['empty', 'solid']
                }
            )
    tile_probs = get_int_prob(prob._prob, tile_types)
    rep.reset(prob._width, prob._height, tile_probs)
    action_space = rep.get_action_space()
    observation_space = rep.get_observation_space(prob._width, prob._height, len(prob.get_tile_types()), 500)
    original_map = np.copy(rep._map)
    pos = rep.agent_positions[1]
    print(rep.tile_id_map)
    print(rep._map[pos['y'], pos['x']])
    rep.apply_action(1, 4)
    print(rep._map[pos['y'], pos['x']])


def test_env_with_grouping():
    # create groups
    groups = {
            1: ['key', 'door'],
            2: ['scorpion', 'spider', 'bat', 'player'],
            3: ['empty', 'solid']
        }
    env = Parallel_MAPcgrlEnv(
            prob='zelda',
            rep='marl_turtle',
            groups=groups,
            binary_actions=False,
            max_iterations=500
            )
    env.reset()
    env.step({1: 0, 2: 0, 3: 0})

def test_wrapped_env_with_grouping():
    pass
