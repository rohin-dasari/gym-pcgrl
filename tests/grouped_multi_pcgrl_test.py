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



def test_max_iterations_limit():
    

    env_config = {
                'num_agents': None,
                'binary_actions': True,
                'max_iterations': 500
                }
    env_name = 'Parallel_MAPcgrl-binary-narrow-v0'
    env = MARL_CroppedImagePCGRLWrapper_Parallel(env_name, 28, **env_config)
    grouped_env = GroupedWrapper(env)
    grouped_env.reset()

    # NOTE: at the time of writing this test, all end conditions, other than
    # the max iterations, have been removed
    for i in range(env_config['max_iterations']):
        actions = {'empty': 0, 'solid': 0}
        _, _, done, _ = grouped_env.step(actions)
        if grouped_env.get_iteration() < env_config['max_iterations']:
            assert done['__all__'] == False
        else:
            assert done['__all__'] == True



