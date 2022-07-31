import gym
import random
import unittest

import ray

from ray.tune.registry import register_env
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.examples.env.multi_agent import BasicMultiAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv

class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""

    def __init__(self, num):
        self.agents = {}
        self.agentID = 0
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False
        
    def spawn(self):
        agentID = self.agentID
        self.agents[agentID] = MockEnv(25)
        self.agentID += 1
        return agentID

    def reset(self):
        self.agents = {}
        self.spawn()
        self.resetted = True
        self.dones = set()
        
        obs = {}
        for i, a in self.agents.items():
           obs[i] = a.reset()

        return obs

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)

        if random.random() > 0.75:
           i = self.spawn()
           obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
           if done[i]:
              self.dones.add(i)

        if len(self.agents) > 1 and random.random() > 0.25:
           keys = list(self.agents.keys())
           key  = random.choice(keys)
           done[key] = True
           del self.agents[key]

        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

class TestMultiAgentEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_train_multi_agent_cartpole_single_policy(self):
        n = 10
        register_env("basic_multi_agent",
                     lambda _: BasicMultiAgent({'num_agents': 10}))
        pg = PGTrainer(
            env="basic_multi_agent",
            config={
                "num_workers": 0,
                "framework": "torch",
            })
        for i in range(50):
            result = pg.train()
            print("Iteration {}, reward {}, timesteps {}".format(
                i, result["episode_reward_mean"], result["timesteps_total"]))
            if result["episode_reward_mean"] >= 50 * n:
                return
        raise Exception("failed to improve reward")

if __name__ == '__main__':
   TestMultiAgentEnv().test_train_multi_agent_cartpole_single_policy()
