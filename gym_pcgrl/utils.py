"""
utilities for parsing config and running experiments
"""
import yaml
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
from gym_pcgrl.wrappers import MARL_CroppedImagePCGRLWrapper_Parallel
from gym_pcgrl.wrappers import MARL_CroppedImagePCGRLWrapper

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

def env_maker_factory(env_name, is_parallel):
    def env_maker(env_config):
        # crop size is harcoded, move to config
        if is_parallel:
            return MARL_CroppedImagePCGRLWrapper_Parallel(env_name, 28, **env_config)
        else:
            return MARL_CroppedImagePCGRLWrapper(env_name, 28, **env_config)

    if is_parallel:
        tune.register_env(
                env_name,
                lambda config: ParallelPettingZooEnv(env_maker(config))
                )
    else:
        tune.register_env(
                env_name,
                lambda config: PettingZooEnv(env_maker(config))
                )
    return env_maker


def parse_qmix_config(config_file):
    """
    """
    # register environment with grouped wrappers

    config = load_config(config_file)

    is_parallel = 'Parallel' in config['rllib_trainer_config']['env']
    env_maker = env_maker_factory(
            config['rllib_trainer_config']['env'],
            config['is_parallel']
            )
    env = env_maker(config['rllib_trainer_config']['env_config'])
    # what classifies as global state?
        # un-cropped / padded map
        # other agent positions
    # Make wrapper to generate global and local states

    agents = env.possible_agents
    sample_agent = agents[0]
    obs_space = Tuple(
        [
            
        ]
    )
    obs_space = env.observation_spaces[sample_agent]
    action_space = env.action_spaces[sample_agent]

    # make grouped version of environment
    env = env_maker(config['rllib_trainer_config']['env_config'])
    # wrap environment in GroupeAgentsWrapper (rllib.env.wrappers.group_agentsa_wrapper)
    # defined groups
    # convert obs and act spaces to tuples


    # add model config
    #config  = {
    #    'mixer': ,
    #    'env_config': ,
    #    'env': 'QMIX', 


    #}
    
    pass

def parse_rllib_config(config_file):
    """
    construct a valid rllib trainer config from a flat config
    """
    
    config = load_config(config_file)

    is_parallel = 'Parallel' in config['rllib_trainer_config']['env']
    env_maker = env_maker_factory(
            config['rllib_trainer_config']['env'],
            config['is_parallel']
            )

    env = env_maker(config['rllib_trainer_config']['env_config'])
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_spaces[sample_agent]
    action_space = env.action_spaces[sample_agent]

    policy_mapping_fn = lambda agent: f'policy_{agent}'
    policies = {f'policy_{agent}': gen_policy(obs_space, action_space, config['model_config']) for agent in env.possible_agents}
    # RLLIB parameters
    # env
    # env_config
    # num_gpus
    # framework
    # render_env
    return {
            **config['rllib_trainer_config'],
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
            **config['tune_api_config'],
            }

def parse_config(config_file):
    return {
            'rllib_config': parse_rllib_config(config_file),
            'tune_config': parse_tune_config(config_file)
            }


