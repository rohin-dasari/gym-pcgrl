import pytest
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from gym_pcgrl.parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper_Parallel
from gym_pcgrl.multiagent_wrappers import MARL_CroppedImagePCGRLWrapper

# GIVEN

# WHEN

# THEN

def test_wrapper():
    # create a new environment
    env = MARL_CroppedImagePCGRLWrapper(
                    'MAPcgrl-binary-narrow-v0',
                    28,
                    binary_actions=True
                )
    reset_obs = env.reset()
    done = False
    while not done:
        obs, rew, dones, info = env.step(1)
        done = dones['__all__']
    env.step(None)
    #print(type(obs))
    #print(type(rew))
    #print(type(dones))
    #print(type(info))
        #print(type(reset_obs))
    #print(type(obs))
    #print(type(env.observe('empty')))
    #print(type(env))

    # wrap environment

    # perform a step
    pass

