from ray import tune
from ray.tune import register_env
from gym_pcgrl.wrappers import wrappers

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk


register_env('single_agent_env', lambda config: make_env('binary-narrow-v0', 'narrow', **config)())

config = {
        'env': 'single_agent_env',
        'env_config': {'cropped_size': 28},
        'num_gpus': 0,
        'framework': 'torch',
        'render_env': False,
        'model': {
            'custom_model': 'CustomFeedForwardModel'
            }
        }


tune.run(
            run_or_experiment='PPO',
            config=config,
            stop={
                'training_iteration': 2
                },
            mode='max',
            metric='episode_reward_mean',
            checkpoint_score_attr='episode_reward_mean',
            keep_checkpoints_num=3,
            checkpoint_freq=1,
            checkpoint_at_end=True,
            
        )


