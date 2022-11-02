from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v3

if __name__ == "__main__":

    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env())

    env = env_creator({})
    import pdb;pdb.set_trace()
    register_env("waterworld", env_creator)

    obs_space = env.observation_space
    act_space = env.action_space

    policies = {"shared_policy": (None, obs_space, act_space, {})}

    # for all methods
    policy_ids = list(policies.keys())

    tune.run(
        "APEX_DDPG",
        stop={"episodes_total": 10},
        checkpoint_freq=10,
        local_dir="my_results",
        config={

            # Enviroment specific
            "env": "waterworld",

            # General
            "num_gpus": 0,
            "num_workers": 1,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: "shared_policy"),
            },
            "evaluation_interval": 1,
            "evaluation_config": {
                "record_env": "videos",
                "render_env": False,
            },
        },
    )
