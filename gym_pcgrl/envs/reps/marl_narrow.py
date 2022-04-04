import warnings
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
When a new representation is created, the agent positions and default map are
not set. This is done manually by the user through the reset method. If a user
tries to call a method in this representation before resetting the environment,
they would receive dubious values that don't make sense. This decorator warns
users to reset their environment before it executes a method.

Parameters:
    method (Callable)

Returns:
    _wrapper (Callable)
"""
def reset_check(method):
    def _wrapper(*args, **kwargs):
        self = args[0]
        if not self._reset:
            warnings.warn(f'The environment should be reset before calling {method.__name__}', RuntimeWarning)
        return method(*args, **kwargs)
    return _wrapper


class MARL_NarrowRepresentation(NarrowRepresentation):
    """
    Initialize all parameters
    In the single-agent narrow representation, the agent is given the current
    state and a specefic coordinate location of a space that it can modify.
    In the multi-agent narrow representation, each agent is supplied with the
    full current state and each agent is given its own position to modify
    """
    # Note that random tile is set to a default value in the parent class
    # Here, we allow the user to override this value through the random_value parameter
    def __init__(self, agents, random_tile=False):
        super().__init__()
        self.agents = agents

        # instead of storing positions as direct attributes of the class,
        # we need to store them in this dictionary
        # each key corresponds to an unique agent id
        # each value corresponds to that agent's position
        self._random_tile = random_tile
        self.agent_positions = {}
        self._reset = False


    """
    Resets the environment and the starting positions for each agent randomly

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob):
        self._reset = True
        super().reset(width, height, prob)
        # initialize starting position for each agent
        # there is a unique agent for each tile type
        self.agent_positions = {}
        for agent in self.agents:
            self.agent_positions[agent] = {
                        'x': self._random.randint(width),
                        'y': self._random.randint(height)
                    }

    """
    Gets the action space used by the narrow representation. For the MARL
    formulation of the narrow PCGRL representation, each agent is limited to
    placing or removing a single game asset, limiting the action space to 2
    options: place a tile or no-op

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Discrete: the action space used by that narrow representation which
        correspond to which value for each tile type
    """
    def get_action_space(self, width, height, num_tiles):
        return {agent: spaces.Discrete(2) for agent in self.agents}


    """
    Get the observation space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Dict: the observation space used by that representation. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation_space(self, width, height, num_tiles, max_changes):
        obs_space = {}
        for agent in self.agents:
            obs = spaces.Dict({
                "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width)),
                "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
                "heatmap": spaces.Box( 
                    low=0,
                    high=max_changes,
                    dtype=np.uint8,
                    shape=(height, width)
                    )
                })
            obs_space[agent] = obs

        return obs_space

    """
    Get current observations for all agents

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    @reset_check
    def get_observations(self):
        return {agent: self.get_observation(agent) for agent in self.agents}

    @reset_check
    def get_observation(self, agent):
        position = self.agent_positions[agent]
        return OrderedDict({
            "map": self._map.copy(),
            "pos": np.array([position['x'], position['y']], dtype=np.uint8)
            })

    """
    Update the narrow representation with the input action

    Place the asset for the specified agent and return a new position by
    incrementing the horizontal positions from left to right

    Parameters:
        agent: an agent id
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    @reset_check
    def apply_action(self, agent, action):
        change = 0
        pos = self.agent_positions[agent]
        x, y = pos['x'], pos['y']
        if action == 0: # no-op
            return change, x, y

        tile_id = action-1
        change += int(self._map[y][x] != action-1)
        self._map[y][x] = tile_id

        # update for the agent's next position
        width, height = self._map.shape[1], self._map.shape[0]
        if self._random_tile:
            x = self._random.randint(width)
            y = self._random.randint(height)
        else:
            x += 1
            if x >= width:
                x = 0
                y += 1
                if y >= height:
                    y = 0
        # update positions
        pos['x'] = x
        pos['y'] = y
        return change, pos['x'], pos['y']

    """
    """
    @reset_check
    def update(self, actions):
        updates = []
        for agent, action in actions.items():
            update = self.apply_action(agent, action)
            updates.append(update)
        return updates

