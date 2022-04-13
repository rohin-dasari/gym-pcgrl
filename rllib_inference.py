"""
Load trained RLLIB model and use it on a sample environment
"""


from gym_pcgrl.utils import parse_config
import ray.rllib.agents.ppo as ppo
from gym_pcgrl.utils import env_maker_factory


def restore_trainer(config_path, checkpoint_path):
    config_path = 'configs/full_actions_maze.yaml'
    config = parse_config(config_path)

    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    return trainer

def build_env(env_name):
    env_maker = env_maker_factory(env_name)
    env = env_maker(env_name)
    return env


def get_agent_actions(observations, policy_mapping_fn):
    pass

def rollout(config):
    done = False
    obs = env.reset()
    while not done:
        actions = {}

    pass

config_path = 'configs/full_actions_maze.yaml'
checkpoint_path = '/home/rohindasari/ray_results/PPO/PPO_MAPcgrl-binary-narrow-v0_eed2c_00000_0_2022-04-13_14-52-57/checkpoint_000004/checkpoint-4'

trainer = retore_trainer(config_path, checkpoint_path)
env = build_env(config['env'])

done = False
obs = env.reset()


while not done:
    actions = {}
    for agent_id, agent_obs in obs.items():
        policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
        actions[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
    
    obs, rew, done, info = env.step(actions)
    frame = env.render()
    done = done['__all__']
    




