import gym
import gym_pcgrl

import numpy as np
import math
import os

from .wrapper_utils import get_env

MAPCGRL_ENV = 'MAPcgrlEnv'

#get_mapcgrl_obj = lambda env: env if "MAPcgrlEnv" in str(type(env)) else get_mapcgrl_obj(env.env)
get_mapcgrl_obj = lambda env: env if "MAPcgrlEnv" in str(type(env)) else get_mapcgrl_obj(env.env)

class MARL_Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        self.env = get_env(game)
        get_mapcgrl_obj(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        obs_spaces = self.env.observation_spaces
        sample_obs_space = obs_spaces[self.env.possible_agents[0]]

        assert all('pos' in obs.spaces.keys() for obs in obs_spaces.values()), \
                'This wrapper only works for representations that have a position'

        assert name in sample_obs_space.spaces.keys(), \
                f'This wrapper only works for observation spaces with a {name} key'

        assert len(sample_obs_space.spaces[name].shape) == 2, \
                'This wrapper only works on 2D arrays'

        self.name = name
        self.size = crop_size
        self.pad = crop_size // 2
        self.pad_value = pad_value

        obs_space = gym.spaces.Dict({})
        for (k, s) in sample_obs_space.items():
            obs_space.spaces[k] = s

        high_value = obs_space[name].high.max()
        obs_space.spaces[name] = gym.spaces.Box(
                low=0,
                high=high_value,
                shape=(crop_size, crop_size),
                dtype=np.uint8
                )

        obs_spaces = {}
        for agent in self.env.possible_agents:
            obs_spaces[agent] = obs_space
        self.observation_spaces = obs_spaces

        
    def last(self):
        obs, rew, done, info = self.env.last()
        return self.transform(obs), rew, done, info

    def observe(self, agent):
        return self.transform(self.env.observe(agent))


    def _was_done_step(self, action):
        self.env._was_done_step(action)

    def step(self, action):
        if action == None:
            return self._was_done_step(action)

        self.env.step(action)
        self.observations = self.transform_observations(self.env.observations)
        obs = self.transform(self.env.observe_current_agent())
        rew = self.env.get_current_agent_rewards()
        done = self.env.get_current_agent_done()
        info = self.env.get_current_agent_info()
        return obs, rew, done, info

    def reset(self):
        self.env.reset()
        self.observations = self.transform_observations(self.env.observations)
        obs = self.transform(self.env.observe_current_agent())
        return obs

    def transform_observations(self, observations):
        return {agent: self.transform(obs) for agent, obs in observations.items()}

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs['pos']

        # View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y+self.size, x:x+self.size]
        obs[self.name] = cropped
        return obs


class MARL_OneHotEncoding(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        self.env = get_env(game)
        get_mapcgrl_obj(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        obs_spaces = self.env.observation_spaces
        sample_obs_space = obs_spaces[self.env.possible_agents[0]]
        
        assert name in sample_obs_space.spaces.keys(), \
                f'This wrapper only works for representations that have a {name} key'

        self.name = name
        obs_space_name = sample_obs_space[self.name]
        self.dim = obs_space_name.high.max() - obs_space_name.low.min() + 1

        obs_space = gym.spaces.Dict({})
        # make a shallow copy of the observation spaces
        for (k,s) in sample_obs_space.items():
            obs_space.spaces[k] = s
        shape = sample_obs_space[name].shape
        self.dim = sample_obs_space[name].high.max() - sample_obs_space[name].low.min() + 1

        new_shape = [v for v in shape]
        new_shape.append(self.dim)
        obs_space[name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)

        obs_spaces = {}
        for agent in self.env.possible_agents:
            obs_spaces[agent] = obs_space
        self.observation_spaces = obs_spaces

    def last(self):
        obs, rew, done, info = self.env.last()
        return self.transform(obs), rew, done, info

    def observe(self, agent):
        return self.transform(self.env.observe(agent))

    def _was_done_step(step, action):
        self.env._was_done_step(action)

    def step(self, action):
        if action == None:
            return self._was_done_step(action)

        self.env.step(action)
        self.observations = self.transform_observations(self.env.observations)
        obs = self.transform(self.env.observe_current_agent())
        rew = self.env.get_current_agent_rewards()
        done = self.env.get_current_agent_done()
        info = self.env.get_current_agent_info()
        return obs, rew, done, info

    def reset(self):
        self.env.reset()
        self.observations = self.transform_observations(self.env.observations)
        obs = self.transform(self.env.observe_current_agent())
        return obs

    def transform_observations(self, observations):
        return {agent: self.transform(obs) for agent, obs in observations.items()}

    def transform(self, obs):
        old = obs[self.name]
        obs[self.name] = np.eye(self.dim)[old]
        return obs


class MARL_ToImage(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        self.env = get_env(game)
        get_mapcgrl_obj(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        obs_spaces = self.env.observation_spaces
        sample_obs_space = obs_spaces[self.env.possible_agents[0]]
    
        obs_spaces = {}
        for agent in self.env.possible_agents:
            obs_spaces[agent] = gym.spaces.Box(
                                low=0,
                                high=sample_obs_space['map'].high.max(),
                                shape=(*sample_obs_space['map'].shape, 1)
                            )
        self.observation_spaces = obs_spaces

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def last(self):
        obs, rew, done, info = self.env.last()
        return self.transform(obs), rew, done, info

    def observe(self, agent):
        return self.transform(self.env.observe(agent))

    def _was_done_step(self, action):
        self.env._was_done_step(action)

    def step(self, action):
        if action == None:
            return self._was_done_step(action)

        self.env.step(action)
        self.observations = self.transform_observations(self.env.observations)
        obs = self.transform(self.env.observe_current_agent())
        rew = self.env.get_current_agent_rewards()
        done = self.env.get_current_agent_done()
        info = self.env.get_current_agent_info()
        return obs, rew, done, info

    def reset(self):
        self.env.reset()
        obs = self.transform(self.env.observe_current_agent())
        self.observations = self.transform_observations(self.env.observations)
        return obs

    def transform_observations(self, observations):
        return {agent: self.transform(obs) for agent, obs in observations.items()}

    def transform(self, obs):
        #final = np.empty([])
        #final = obs[self.name].reshape(self.shape[0], self.shape[1], -1)
        #return final
        return obs['map'][..., np.newaxis]


"""
The wrappers we use for narrow and turtle experiments
"""
class MARL_CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = get_mapcgrl_obj(gym.make(game, **kwargs))
        #self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        envs = []
        cropped_env = MARL_Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')
        envs.append(cropped_env)
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            onehot_env = MARL_OneHotEncoding(cropped_env, 'map')
            envs.append(onehot_env)
        ## Indices for flatting
        flat_indices = ['map']
        ### Final Wrapper has to be ToImage or ToFlat
        toimage_env = MARL_ToImage(envs[-1], flat_indices)
        envs.append(toimage_env)
        self.env = toimage_env
        self.envs = envs
        super().__init__(self.env)

    def transform_observations(self, observations):
        return {agent: self.transform(obs) for agent, obs in observations.items()}

    def transform(self, obs):
        for env in self.envs:
            obs = env.transform(obs)
        return obs

