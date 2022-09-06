import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

import gym_pcgrl
from .wrapper_utils import get_env



"""
According to the QMIX paper:
    "To speed up the learning, we share the parameters of the agent networks across all agents. Because of this, a one-hot
encoding of the agent id is concatenated onto each agent’s observations"
"""

class GroupedWrapper(MultiAgentEnv, gym.Wrapper):
    def __init__(self, env):
        # modify the observation and action space to be tuples
        # wrap env in grouped Wrapper
        self.env = get_env(env)
        super().__init__(self.env)
        self._agent_ids = set(self.env.possible_agents)
        sample_agent = self.env.possible_agents[0]
        self.observation_space = self.env.observation_space(sample_agent)
        self.action_space = self.env.action_space(sample_agent)


    def _obs(self):
        return self.env.observations

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)
