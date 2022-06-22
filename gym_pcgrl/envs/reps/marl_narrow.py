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
    def __init__(self, agents, tiles, random_tile=True, binary_actions=True):
        super().__init__()

        self.binary_actions = binary_actions
        if self.binary_actions:
            assert len(agents) == len(tiles), \
                    "If you each agent to have isolated control over an " + \
                    "individual tile type, the number of agents must be " + \
                    "equal to the number of tiles in the problem"

        self.agents = agents
        self.tiles = tiles

        self.tile_id_map = {tile: i for i, tile in enumerate(self.tiles)}

        # instead of storing positions as direct attributes of the class,
        # we need to store them in this dictionary
        # each key corresponds to an unique agent id
        # each value corresponds to that agent's position
        self.agent_positions = {}
        self._random_tile = random_tile
        print(self._random_tile)
        self._reset = False


    """
    Resets the environment and the starting positions for each agent randomly

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob, initial_level=None, initial_positions=None):
        self._reset = True
        super().reset(width, height, prob, initial_level)
        # initialize starting position for each agent
        # there is a unique agent for each tile type
        self.agent_positions = {}
        if initial_positions is None:
            for agent in self.agents:
                self.agent_positions[agent] = {
                            'x': self._random.randint(width),
                            'y': self._random.randint(height)
                        }
        else:
            self.agent_positions = initial_positions

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
        num_actions = 2 if self.binary_actions else len(self.tiles) + 1
        return {agent: spaces.Discrete(num_actions) for agent in self.agents}


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

        if action != 0:
            if self.binary_actions:
                tile_id = self.tile_id_map[agent]
            else:
                tile_id = action - 1
            change += int(self._map[y][x] != tile_id)
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
        self.agent_positions[agent] = pos
        return change, x, y # return number of changes and new position

    """
    """
    @reset_check
    def update(self, actions):
        updates = []
        for agent, action in actions.items():
            update = self.apply_action(agent, action)
            updates.append(update)
        return updates


    def draw_rect(self, tile_size, color):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x), color)
            x_graphics.putpixel((1,x), color)
            x_graphics.putpixel((tile_size-2,x), color)
            x_graphics.putpixel((tile_size-1,x), color)
        for y in range(tile_size):
            x_graphics.putpixel((y,0), color)
            x_graphics.putpixel((y,1), color)
            x_graphics.putpixel((y,tile_size-2), color)
            x_graphics.putpixel((y,tile_size-1), color)
        return x_graphics


    """
    Modify the level image with a red rectangle around the tile that is
    going to be modified

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        colors = [(255, 0, 0, 255), (0, 255, 0, 255)] # expand to handle more agents
        for color, agent in zip(colors, self.agents):
            rect = self.draw_rect(tile_size, color)
            x_pos = (self.agent_positions[agent]['x']+border_size[0])*tile_size
            x_pos_width = (self.agent_positions[agent]['x']+border_size[0]+1)*tile_size
            y_pos = (self.agent_positions[agent]['y']+border_size[1])*tile_size
            y_pos_height = (self.agent_positions[agent]['y']+border_size[1]+1)*tile_size
            lvl_image.paste(rect, (x_pos, y_pos, x_pos_width, y_pos_height), rect)
        return lvl_image
