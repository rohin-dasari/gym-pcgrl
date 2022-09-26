import pytest
from gym.spaces import Tuple

from gym_pcgrl.envs import Grouped_MAPcgrlEnv
from gym_pcgrl.wrappers import MARL_CroppedImagePCGRLWrapper_Parallel
from gym_pcgrl.wrappers import GroupedWrapper, make_grouped_env




def test_basic():
    env = make_grouped_env(
            'Parallel_MAPcgrl-binary-narrow-v0',
            28,
            **{
                'binary_actions': True 
            }
        )


    obs = env.reset()


