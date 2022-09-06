import pytest
from gym.spaces import Tuple

from gym_pcgrl.envs import Grouped_MAPcgrlEnv
from gym_pcgrl.wrappers import MARL_CroppedImagePCGRLWrapper_Parallel
from gym_pcgrl.wrappers import GroupedWrapper




def test_basic():
    env_name = 'Parallel_MAPcgrl-binary-narrow-v0'
    env = MARL_CroppedImagePCGRLWrapper_Parallel(env_name, 28)
    grouped_env = GroupedWrapper(env)
    print(grouped_env.possible_agents)
    
    print('ahhhhhh')
    print(env.observation_space('empty').shape)
    print(grouped_env.observation_space.shape)


    #groups = {
    #        'group1': grouped_env.possible_agents
    #    }

    #tuple_obs_space = Tuple(
    #            [grouped_env.observation_space \
    #                    for _ in grouped_env.possible_agents]
    #        )
    #tuple_act_space = Tuple(
    #            [grouped_env.action_space \
    #                    for _ in grouped_env.possible_agents]
    #        )


    #GroupedWrapper(env).with_agent_groups(groups, obs_space=tuple_obs_space, act_space=tuple_act_space)



