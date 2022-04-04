import numpy as np
from gym.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


NONE = 3 #?

class raw_env(AECEnv):
    def __init__(self):
        self.possible_agents = [f'player_{i}' for i in range(2)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(4) for agent in self.possible_agents}

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: 3 for agent in self.agents}
        self.observations = {agent: 3 for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.reward = 1

            self.num_moves += 1

            self.dones = {agent: self.num_moves >= 100 for agent in self.agents}


            for i in self.agents:
                self.observations[i] = self.state[self.agents[1-self.agent_name_mapping[i]]]

        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = 3
            self._clear_rewards()


        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

def gen_policy(obs_space, act_space):
    config = {
            'model': {'custom_model': 'main_model', 'custom_model_config': {}},
            'gamma': 0.95
            }
    return (None, obs_space, act_space, config)

def env_maker(config):
    return raw_env()

if __name__ == '__main__':
    #from marl_model import CustomFeedForwardModel as Model
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as Model
    from ray import tune
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model('main_model', Model)

    env = raw_env()
    obs_space = env._observation_spaces['player_0']
    action_space = env._action_spaces['player_0']
    policies = {f'policy_{agent}': gen_policy(obs_space, action_space) for agent in env.possible_agents}
    policy_mapping_fn = lambda agent: f'policy_{agent}'


    tune.register_env('test_env', lambda config: PettingZooEnv(env_maker(config)))


    config = {
            'env': 'test_env',
            'env_config': {},
            'num_gpus': 0,
            'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_mapping_fn
                },
            'model': {},
            'framework': 'torch'
            }
    results = tune.run('PPO', config=config, verbose=1)
    #env.reset()
    

