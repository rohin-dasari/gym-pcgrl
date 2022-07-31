from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from .parallel_multi_pcgrl_env import Parallel_MAPcgrlEnv

import functools
import numpy as np
import gym
from gym import spaces
import PIL
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class MAPcgrlEnv(Parallel_MAPcgrlEnv, AECEnv):
    def __init__(
                self,
                num_agents=None,
                prob="binary",
                rep="marl_narrow",
                binary_actions=True,
                change_percentage=0.2,
                **kwargs
            ):
        # inherit all methods from Parallel_MAPcgrlEnv
        super().__init__(num_agents, prob, rep, binary_actions, change_percentage, **kwargs)


    def observe_current_agent(self):
        return self.observe(self.agent_selection)

    def reset(self, initial_level=None, initial_positions=None):
        # call super's init
        obs = super().reset(initial_level, initial_positions)
        # set agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        return self.observe_current_agent()

    def step(self, action):
        agent = self.agent_selection
        if self.dones[agent]:
            return self._was_done_step(action)

        self._iteration += 1
        self.agent_actions_history[agent].append(action)

        # update cumulative rewards
        #taken from https://www.pettingzoo.ml/environment_creation#example-custom-environment
        # why do we set _cumulative_rewards to 0 for each step?
        # shouldn't cumulative rewards be keep getting added to each time step
        self._cumulative_rewards[agent] = 0
        
        # store action of current agent
        self.state[self.agent_selection] = action

        # update level map
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update game state based on selected actions
        [update] = self._rep.update({agent: action})
        # update heatmap
        n_changes = self.update_heatmap(agent, update, action)
        if n_changes > 0:
            new_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
            # update rep stats
            self._rep_stats = new_stats

        # update observations
        observations = self._rep.get_observations()
        for agent, obs in observations.items():
            obs["heatmap"] = self._agent_heatmaps[agent].copy()
        self.observations = observations

        if self._agent_selector.is_last():
            # update rewards for all agents
            reward = self._prob.get_reward(self._rep_stats, old_stats)
            self.rewards = {agent: reward for agent in self.agents}
        else:
            # make sure agent rewards are all set to 0
            self.reset_rewards()

        # if agent is done, set dones for all agents
        done = self.check_done(self._rep_stats, old_stats)
        dones = {agent: done for agent in self.agents}
        dones['__all__'] = done
        self.dones = dones

        # collect metadata for all agents
        info = self.get_metadata()
        self.infos = info

        # select next agent
        self.agent_selection = self._agent_selector.next()

        # add .rewards to ._cumulative_rewards
        self._accumulate_rewards()


        # petting zoo does not require that step() returns these elements,
        # but gym wrappers do
        return self.observations, self.rewards, self.dones, self.infos

    def get_tile_map(self):
        """
        obtain the mapping between the tile types and their integer encodings
        """
        return self._rep.tile_id_map





        
    


