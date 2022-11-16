from gym.spaces import Tuple
from ray import tune
from ray.tune import register_env
from gym_pcgrl import wrappers
from gym_pcgrl.utils import make_grouped_env, register_grouped_env



if __name__ == '__main__':
    env_name = 'Parallel_MAPcgrl-binary-marl_turtle-v0'
    #env_name = 'Parallel_MAPcgrl-zelda-narrow-v0'
    register_grouped_env(env_name)
    config = {
            "rollout_fragment_length": 1,
            "train_batch_size": 32,
            "exploration_config": {
                "final_epsilon": 0.0,
            },
            "num_workers": 0,
            "mixer": 'qmix',
            "env_config": {
                "binary": True
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            'model': {
                'custom_model': 'CustomFeedForwardModel'
                },
            'env': 'grouped_env',
            }


    tune.run(
            run_or_experiment='QMIX',
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

