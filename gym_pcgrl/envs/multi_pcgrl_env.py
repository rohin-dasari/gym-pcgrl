from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

import numpy as np
import gym
from gym import spaces
import PIL


"Multi-agent PCGRL Gym Environment"
class MAPcgrlEnv(PcgrlEnv):
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
    def __init__(self, prob="binary", rep="marl_narrow"):
        self._prob = PROBLEMS[prob]()
        self.agents = self._prob.get_tile_types()
        self._rep = REPRESENTATIONS[rep](self.agents)
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self._heatmaps = self.init_heatmaps()

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

        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._heatmaps[agent].copy()
        return observations

    """
    Define dimensions of heatmap in the observation space for each agent
    """
    def init_heatmaps(self):
        for obs in self.observation_space.values():
            obs.spaces['heatmap'] = spaces.Box(
                    low=0,
                    high=self._max_changes,
                    dtype=np.uint8,
                    shape=(self._prob._height, self._prob._width)
                    )
        
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
        print(old_stats)
        # update the current state to the new state based on the taken action
        #change, x, y = self._rep.update(actions)
        # update game state based on selected actions
        updates = self._rep.update(actions)

        changes = [self.update_heatmap(agent, update) for agent, update in zip(self.agents, updates)]
        if sum(changes) > 0:
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))

        # get next state
        observations = self._rep.get_observations()
        for agent in self._heatmaps:
            observations["heatmap"] = self._heatmaps[agent].copy()

        # compute reward
        # assume shared reward signal
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        rewards = [reward for _ in self.agents]

        # check game end conditions
        done = self.check_done(old_stats)

        # collect metadata
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes

        #return the values
        return observations, rewards, done, info

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
    def check_done(self, old_stats):
        return self._prob.get_episode_over(self._rep_stats,old_stats) or \
                self._changes >= self._max_changes or \
                self._iteration >= self._max_iterations

