from copy import copy
import gym
from gym.spaces import Tuple, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

import gym_pcgrl
from .wrapper_utils import get_env
from .parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper_Parallel



"""
According to the QMIX paper:
    To speed up the learning, we share the parameters of the agent networks across all agents.
    Because of this, a one-hot encoding of the agent id is concatenated onto each agent’s observations"
"""

class GroupedWrapper(MultiAgentEnv, gym.Wrapper):
    def __init__(self, env):
        # modify the observation and action space to be tuples
        # wrap env in grouped Wrapper
        self.env = get_env(env)
        super().__init__()
        self._agent_ids = set(self.env.possible_agents)
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.env.possible_agents)}
        sample_agent = self.env.possible_agents[0]
        # for two agents, the agent id can be represented as either 0 or 1
        # for more agents, the id needs to be one hot encoded
        self.observation_space = Dict(
                    {
                        'obs': self.env.observation_space(sample_agent),
                        'id': Discrete(1) # add an encoding for the agent id 
                    }
                )
        #self.observation_space = self.env.observation_space(sample_agent)
        self.action_space = self.env.action_space(sample_agent)


    def _obs(self):
        return self._add_id_to_obs(self.env.observations)

    def add_id_to_obs(self, observation):
        newobs = {}
        for agent, obs in observation.items():
            newobs[agent] = {
                    'obs': obs,
                    'id': self.agent_name_mapping[agent]
                }

        return newobs

    def reset(self):
        obs = self.env.reset()
        return self.add_id_to_obs(obs)

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        obs_with_id = self.add_id_to_obs(obs)
        return obs_with_id, rew, done, info

    def set_state(self, initial_level, initial_positions):
        return self.env.set_state(
                    initial_level,
                    initial_positions
                )
    def render(self, mode='human'):
        return self.env.render(mode)


def make_grouped_env(env_name, crop_size, **kwargs):

    env = MARL_CroppedImagePCGRLWrapper_Parallel(
                env_name,
                crop_size,
                **kwargs
            )

    grouped_env = GroupedWrapper(env)
    groups = {
            'group1': grouped_env.possible_agents
        }

    tuple_obs_space = Tuple(
                [grouped_env.observation_space \
                        for _ in grouped_env.possible_agents]
            )
    tuple_act_space = Tuple(
                [grouped_env.action_space \
                        for _ in grouped_env.possible_agents]
            )



    return GroupedWrapper(env).with_agent_groups(
                                            groups,
                                            obs_space = tuple_obs_space,
                                            act_space = tuple_act_space
                                        )


