import warnings
from copy import copy
from random import randint
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict
from matplotlib.pyplot import get_cmap


"""
The turtle representation where the agent is trying to modify the position of the
turtle or the tile value of its current location similar to turtle graphics.
The difference with narrow representation is the agent now controls the next tile to be modified.
"""
class MARL_TurtleRepresentation(TurtleRepresentation):

    def __init__(self, agents, tiles, binary_actions=True, warp=False):

        super().__init__(warp)
        self.binary_actions = binary_actions

        self.agents = agents
        #self.agent_color_mapping = {
        #            agent: (randint(1,255), randint(1,255), randint(1,255), 255) 
        #            for agent in self.agents
        #        }
        self.agent_color_mapping = self.init_cmap()
        self.tiles = tiles

        self.tile_id_map = {tile: i for i, tile in enumerate(self.tiles)}

        # instead of storing positions as direct attributes of the class,
        # we need to store them in this dictionary
        # each key corresponds to an unique agent id
        # each value corresponds to that agent's position
        self.agent_positions = {}
        self._reset = False


    def init_cmap(self):
        colors = get_cmap('Pastel1').colors[:len(self.agents)]
        def convert_to_255(channels):
            return tuple([int(255*channel) for channel in channels])

        return {agent: convert_to_255(c) for agent, c in zip(self.agents, colors)}

    """
    Resets the environment and the starting positions for each agent randomly

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob, initial_level=None, initial_positions=None):
        self._reset = True
        super().reset(width, height, prob)
        # initialize starting position for each agent
        # there is a unique agent for each tile type
        self.agent_positions = {}
        if initial_positions is None:
            for i, agent in enumerate(self.agents):
                # To Do: Hardcoded fixed starting positions
                self.agent_positions[agent] = {
                            'x': i,
                            'y': 0,
                        }
                #self.agent_positions[agent] = {
                #            'x': self._random.randint(width),
                #            'y': self._random.randint(height)
                #        }
        else:
            self.agent_positions = initial_positions

    """
    for all cases, each agent can either move up, left, right, or down
    For the binary case, each agent is given a choice to place the tile
        - Note this is different from the narrow case, where each agent is
          given the option to either not place (no-op) or place a tile
        - In the turtle, case, we have removed the option for no-op. If the
          agent doesn't place its tile, it must move
    For the full case, each agent is free to place any thing it chooses
    Working on agent groupings:
        - Each agent can choose to place any tile in its group
        - For example, the door and key tiles can be assigned to a single
          group. The agent for this group can place either doors or keys
    """
    def get_action_space(self):
        num_actions = 1 if self.binary_actions else len(self.tiles)
        return {agent: spaces.Discrete(len(self._dirs) + num_actions) for agent in self.agents}


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

    def get_new_position(self, agent, direction):
        pos = self.agent_positions[agent]
        curr_x, curr_y = pos['x'], pos['y']
        x_dir, y_dir = direction[0], direction[1]
        new_x, new_y = curr_x + x_dir, curr_y + y_dir
        # handle x coordinate out of bounds
        if new_x < 0:
            if self._warp:
                new_x += self._map.shape[1]
            else:
                new_x = 0
        elif new_x >= self._map.shape[1]:
            if self._warp:
                new_x -= self._map.shape[1]
            else:
                new_x = self._map.shape[1] - 1


        # handle y coordinate out of bounds
        if new_y < 0:
            if self._warp:
                new_y += self._map.shape[0]
            else:
                new_y = 0
        elif new_y >= self._map.shape[0]:
            if self._warp:
                new_y -= self._map.shape[0]
            else:
                new_y = self._map.shape[0] - 1

        return new_x, new_y

    """
    Get current observations for all agents

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observations(self):
        return {agent: self.get_observation(agent) for agent in self.agents}

    def get_observation(self, agent):
        position = self.agent_positions[agent]
        return OrderedDict({
            "map": self._map.copy(),
            "pos": np.array([position['x'], position['y']], dtype=np.uint8)
            })


    """
    the turtle representation doesn't have a no-op action
    """
    def apply_action(self, agent, action):
        change = 0
        pos = self.agent_positions[agent]
        curr_x, curr_y = pos['x'], pos['y']
        if action < len(self._dirs): # handle change position case
            direction = self._dirs[action]
            new_x, new_y = self.get_new_position(agent, direction)
            pos['x'], pos['y'] = new_x, new_y
            return change, curr_x, curr_y
        else: # handle tile placement case
            if self.binary_actions:
                tile_id = self.tile_id_map[agent]
            else:
                tile_id = action - len(self._dirs)
            change += int(self._map[curr_y][curr_x] != tile_id)
            self._map[curr_y][curr_x] = tile_id
            return change, curr_x, curr_y



    """
    """
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
        #colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)] # expand to handle more agents
        for agent, color in self.agent_color_mapping.items():
            rect = self.draw_rect(tile_size, color)
            x_pos = (self.agent_positions[agent]['x']+border_size[0])*tile_size
            x_pos_width = (self.agent_positions[agent]['x']+border_size[0]+1)*tile_size
            y_pos = (self.agent_positions[agent]['y']+border_size[1])*tile_size
            y_pos_height = (self.agent_positions[agent]['y']+border_size[1]+1)*tile_size
            lvl_image.paste(rect, (x_pos, y_pos, x_pos_width, y_pos_height), rect)
        return lvl_image

    def get_human_readable_action(self, agent, action):
        if action < len(self._dirs):
            # handle movement actions
            if action == 0:
                return 'move left'
            elif action == 1:
                return 'move right'
            elif action == 2:
                return 'move down'
            elif action == 3:
                return 'move up'

        else:
            # handle tile placement actions
            if self.binary_actions:
                return f'place {agent}'
            else:
                tile_id = action - len(self._dirs)
                return f'place {self.tiles[tile_id]}'
