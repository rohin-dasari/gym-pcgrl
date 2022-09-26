from ray import tune
from ray.tune import register_env
from gym_pcgrl import wrappers



def register_grouped_env():
    register_env(
            'grouped_env',
            lambda config: wrappers.make_grouped_env(
                            'Parallel_MAPcgrl-binary-narrow-v0',
                            28,
                            **config
                        )
            )


if __name__ == '__main__':
    register_grouped_env()
    config = {
            "rollout_fragment_length": 4,
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

