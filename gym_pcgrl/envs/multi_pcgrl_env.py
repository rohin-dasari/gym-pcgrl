from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

import numpy as np
import gym
from gym import spaces
import PIL
from pettingzoo import AECEnv



"Multi-agent PCGRL Gym Environment"
class MAPcgrlEnv(PcgrlEnv, AECEnv):
    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    # number of agents is determined by the number of game assets
    # update representation
    # check reward signal -> for a simple shared signal, do I need to change stuff?
    # how to define different observations and actions for each agent
    # assume all agents have same model, assign roles at environment build time
    def __init__(self, prob="binary", rep="marl_narrow", context=None, rep_kwargs={}, **kwargs):


        self._prob = PROBLEMS['zelda']() # hardcoded problsm
        self.agents = self._prob.get_tile_types()

        self.possible_agents = self.agents
        self.agent_name_mapping = {i: agent for i, agent in enumerate(self.possible_agents)}

        self._rep = REPRESENTATIONS['marl_narrow'](self.agents)
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        self.seed()
        self.viewer = None

        self.dones = {agent: False for agent in self.agents}
        self.dones['__all__'] = False

        self.action_spaces = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_spaces = self._rep.get_observation_space(
                self._prob._width,
                self._prob._height,
                self.get_num_tiles(),
                self._max_changes
                )


    """
    """
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    """
    """
    def action_space(self, agent):
        return self.action_spaces[agent]

    """
    """
    def get_agent_ids(self):
        return self.agents


    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    # return n observations for each agent
    # does every agent need its own heatmap
    def reset(self):
        self._changes = 0
        self._iteration = 0
        tile_probs = get_int_prob(self._prob._prob, self._prob.get_tile_types())
        self._rep.reset(self._prob._width, self._prob._height, tile_probs)
        self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmaps = self.init_heatmaps()

        self.dones = {agent: False for agent in self.agents}
        self.dones['__all__'] = False

        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._heatmaps[agent].copy()
        return observations

    """
    Define dimensions of heatmap in the observation space for each agent
    """
    def init_heatmaps(self):
        height, width = self._prob._height, self._prob._width
        heatmaps = {agent: np.zeros((height, width)) for agent in self.agents}
        return heatmaps

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
        for agent in self._heatmaps:
            observations[agent]["heatmap"] = self._heatmaps[agent].copy()

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
        common_metadata = self._prob.get_debug_info(new_stats, old_stats)
        common_info = {}
        common_info["iterations"] = self._iteration
        common_info["changes"] = self._changes
        common_info["max_iterations"] = self._max_iterations
        common_info["max_changes"] = self._max_changes
        info = {}
        info['__common__'] = {'metadata': common_metadata, **common_info}
        for agent in self.agents:
            info[agent] = {}

        #return the values
        return observations, rewards, dones, info

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

        return self._prob.get_episode_over(new_stats, old_stats) or \
                self._changes >= self._max_changes or \
                self._iteration >= self._max_iterations

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
        self.action_spaces = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_spaces = self._rep.get_observation_space(
                self._prob._width,
                self._prob._height,
                self.get_num_tiles(),
                self._max_changes
                )
