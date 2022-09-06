import numpy as np
import gym
from gym import spaces
import PIL
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pcgrl.envs.helper import get_string_map
from .parallel_multi_pcgrl_env import Parallel_MAPcgrlEnv


class Grouped_MAPcgrlEnv(MultiAgentEnv):
    """
    Implement a multi-agent environment with both local and global state space
    """
    def __init__(env, **kwargs):
        if isinstance(env, str):
            self.env = gym.make(env, **kwargs)
        else:
            self.env = env


        # update observation space to have local and global state
        # local state => {position, map}
        # global state => {map}
        self.action_space = self.env.action_space
        self.observation_space = {}

