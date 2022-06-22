"""
utilities for parsing config and running experiments
"""
import yaml
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from gym_pcgrl.parallel_multiagent_wrappers import MARL_CroppedImagePCGRLWrapper_Parallel
#from gym_pcgrl.multiagent_wrappers import MARL_CroppedImagePCGRLWrapper

def gen_policy(obs_space, action_space, model):
    pass

def gen_policy(obs_space, act_space, model_config):
    config = {
            'model': model_config,
            'gamma': 0.95
            }
    return (None, obs_space, act_space, config)

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config

def env_maker_factory(env_name):
    def env_maker(env_config):
        # crop size is harcoded, move to config
        return MARL_CroppedImagePCGRLWrapper_Parallel(env_name, 28, **env_config)

    tune.register_env(env_name, lambda config: ParallelPettingZooEnv(env_maker(config)))
    return env_maker

def parse_rllib_config(config_file):
    """
    construct a valid rllib trainer config from a flat config
    """
    
    config = load_config(config_file)

    env_maker = env_maker_factory(config['env'])
    #def env_maker(env_config):
    #    return MARL_CroppedImagePCGRLWrapper(config['env'], 28, **config)
    #tune.register_env(config['env'], lambda config: ParallelPettingZooEnv(env_maker(config)))

    env = env_maker(config['env_config'])
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_spaces[sample_agent]
    action_space = env.action_spaces[sample_agent]

    policy_mapping_fn = lambda agent: f'policy_{agent}'
    policies = {f'policy_{agent}': gen_policy(obs_space, action_space, config['model_config']) for agent in env.possible_agents}
    return {
            'env': config['env'],
            'env_config': config['env_config'],
            'num_gpus': config['num_gpus'],
            'framework': config['framework'],
            'render_env': config['render_env'],
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn
                }
            }

def parse_tune_config(config_file):
    config = load_config(config_file)
    ## Tune API parameters
    #algorithm: PPO
    #stop:
    #    training_iteration: 1
    #mode: max
    #metric: episode_reward_mean
    #checkpoint_score_attr: episode_reward_mean
    #keep_checkpoints_num: 3
    #checkpoint_freq: 1
    #checkpoint_at_end: true
    return {
            'run_or_experiment': config['algorithm'],
            'stop': config['stop'],
            'mode': config['mode'],
            'metric': config['metric'],
            'checkpoint_score_attr': config['checkpoint_score_attr'],
            'keep_checkpoints_num': config['keep_checkpoints_num'],
            'checkpoint_freq': config['checkpoint_freq'],
            'checkpoint_at_end' : config['checkpoint_at_end'],
            }

def parse_config(config_file):
    return {
            'rllib_config': parse_rllib_config(config_file),
            'tune_config': parse_tune_config(config_file)
            }


