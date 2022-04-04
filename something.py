from gym_pcgrl.wrappers import MARL_CroppedImagePCGRLWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

#env = ParallelPettingZooEnv(MARL_CroppedImagePCGRLWrapper('MAPcgrl-zelda-narrow-v0', 7))
env = ParallelPettingZooEnv(MARL_CroppedImagePCGRLWrapper('MAPcgrl-zelda-narrow-v0', 7))
obs = env.reset()

print(obs['empty'])
print(type(obs['empty']))

