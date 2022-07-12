from copy import deepcopy
from collections import defaultdict
import functools
import numpy as np
import gym
from gym import spaces
import PIL
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map



"Multi-agent PCGRL Gym Environment"
class Parallel_MAPcgrlEnv(PcgrlEnv, ParallelEnv):
    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(
                self,
                num_agents=None,
                prob="binary",
                rep="marl_narrow",
                binary_actions=True,
                change_percentage=0.2,
                **kwargs
            ):

        self._prob = PROBLEMS[prob]()
        tile_types = self._prob.get_tile_types()
        self.tile_types = tile_types
        self.binary_actions = binary_actions
        if binary_actions:
            self.possible_agents = tile_types
        else:
            assert num_agents is not None, "The number of agents must be explicitly provided"
            self.possible_agents = list(range(num_agents))
        self.agent_name_mapping = {i: agent for i, agent in enumerate(self.possible_agents)}

        # random_tile will be False by default (if it is not passed as an argument)
        self._rep = REPRESENTATIONS[rep](
                    self.possible_agents,
                    tiles=tile_types,
                    binary_actions=binary_actions,
                    random_tile=True if 'random_tile' in kwargs and kwargs['random_tile'] else False
                )
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(change_percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        self.seed()
        self.viewer = None

        self.action_spaces = self._get_action_spaces()
        self.observation_spaces = self._get_observation_spaces()


    """
    convert an integer action id to a human readable action
    """
    def get_human_action(self, agent, action):
        if action == 0:
            return 'no-op'
        if self.binary_actions:
            return f'place {agent}'
        else:
            tile_id = action - 1
            return f'place {self.tile_types[tile_id]}'

    def get_heatmaps(self):
        return self._heatmaps

    """
    """
    def _get_observation_spaces(self):
        return  self._rep.get_observation_space(
                    self._prob._width,
                    self._prob._height,
                    self.get_num_tiles(),
                    self._max_changes
                )

    """
    """
    def _get_action_spaces(self):
        return self._rep.get_action_space(
                    self._prob._width,
                    self._prob._height,
                    self.get_num_tiles()
                )

    """
    """
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    """
    """
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    """
    """
    def get_agent_ids(self):
        return self.agents

    """
    Returns the observation an agent can currently make
    """
    def observe(self, agent):
        observation = self._rep.get_observation(agent)
        observation["heatmap"] = self._heatmaps[agent].copy()
        return observation

    def get_map(self):
        return self._rep._map

    def get_agent_positions(self):
        return self._rep.agent_positions

    def set_state(self, initial_level=None, initial_positions=None):
        """
        Used to set the map and initial positions of the agents outside of the
        reset method. When wrapping this environment in gym wrappers, one cannot
        pass arguments to the reset method to explicitly set the map and agent
        positions, therefore, a user should call this method (if this
        environment is wrapped in gym wrappers) to set the map and agent
        positions.
        """
        initial_level = deepcopy(initial_level)
        initial_positions = deepcopy(initial_positions)

        tile_probs = get_int_prob(self._prob._prob, self._prob.get_tile_types())
        self._rep.reset(self._prob._width, self._prob._height, tile_probs, initial_level)
        if initial_positions is not None:
            self._rep.agent_positions = initial_positions
        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._heatmaps[agent].copy()
        self.observations = observations
        return observations


    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def get_iteration(self):
        return self._iteration

    def reset_rewards(self):
        self.rewards = {agent: 0 for agent in self.agents}

    def reset(self, initial_level=None, initial_positions=None):
        self.agents = self.possible_agents[:]
        self.agent_actions_history = defaultdict(list) # create a log to save actions that the agent takes


        initial_level = deepcopy(initial_level)
        initial_positions = deepcopy(initial_positions)

        self._changes = 0
        self._iteration = 0
        self._heatmaps = self.init_heatmaps()
        tile_probs = get_int_prob(self._prob._prob, self._prob.get_tile_types())
        self._rep.reset(self._prob._width, self._prob._height, tile_probs, initial_level, initial_positions)
        self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.dones['__all__'] = False
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}

        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._heatmaps[agent].copy()

        # only necessary for nonparallel environments
        #self._agent_selector = agent_selector(self.agents)
        #self.agent_selection = self._agent_selector.next()
        #self.updates = []
        self.observations = observations
        return observations

    """
    Define dimensions of heatmap in the observation space for each agent
    """
    def init_heatmaps(self):
        height, width = self._prob._height, self._prob._width
        heatmaps = {agent: np.zeros((height, width)) for agent in self.agents}
        return heatmaps

    def get_rep_stats(self):
        return self._rep_stats

    """
    Advance the environment using a specific action

    Parameters:
        actions: a dictionary of actions that are used to advance the
        environment. Each key represents an agent and each value represents the
        action chosen by that agent

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, actions):

        for agent, action in actions.items():
            self.agent_actions_history[agent].append(action)

        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats

        # update game state based on selected actions
        updates = self._rep.update(actions)

        changes = [self.update_heatmap(agent, update) for agent, update in zip(self.agents, updates)]
        new_stats = old_stats
        if sum(changes) > 0:
            new_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
            self._rep_stats = new_stats

        # get next state
        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._heatmaps[agent].copy()
        self.observations = observations

        # compute reward
        # assume shared reward signal
        reward = self._prob.get_reward(new_stats, old_stats)
        rewards = {agent: reward for agent in self.agents}

        # check game end conditions
        # assume shared done signal
        done = self.check_done(new_stats, old_stats)
        dones = {agent: done for agent in self.agents}
        dones['__all__'] = done
        self.dones = dones


        # collect metadata
        info = self.get_metadata()
        return observations, rewards, dones, info

    def get_metadata(self):
        common_metadata = self._prob.get_debug_info(self._rep_stats)
        common_info = {}
        common_info["iterations"] = self._iteration
        common_info["changes"] = self._changes
        common_info["max_iterations"] = self._max_iterations
        common_info["max_changes"] = self._max_changes
        common_info['metadata'] = common_metadata
        info = {agent: {} for agent in self.agents}
        info['__common__'] = common_info
        return info


    """
    """
    def update_heatmap(self, agent, update):
        change, x, y = update
        if change == 0:
            return change
        self._changes += change
        self._heatmaps[agent][y][x] += 1.0
        return change

    """
    Check for done condition in the environment. There are three conditions
    that can lead to a finished environment:
    1. The environment has reached a satisfiable level of quality (this is
    determined by the problem's `get_episode_over` method)

    2. The maximum number of changes is reached

    3. The maximum number of steps is reached

    Parameters:
        old_stats: stats regarding the representation from the previous timestep
    """
    def check_done(self, new_stats, old_stats):
        return self.check_success() or \
                self._changes >= self._max_changes or \
                self._iteration >= self._max_iterations

    def check_success(self):
        return self._prob.get_episode_over(self._rep_stats)

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_spaces = self._get_action_spaces()
        self.observation_spaces = self._get_observation_spaces()

    def get_agent_color_mapping(self):
        if hasattr(self._rep, 'agent_color_mapping'):
            return self._rep.agent_color_mapping
        else:
            raise RuntimeError('Must call render method in order to compute mapping')

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen
