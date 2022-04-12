from gym.envs.registration import register
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl import models

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    for rep in REPRESENTATIONS.keys():
        register(
            id='{}-{}-v0'.format(prob, rep),
            entry_point='gym_pcgrl.envs:PcgrlEnv',
            kwargs={"prob": prob, "rep": rep}
        )

        register(
            id='MAPcgrl-{}-{}-v0'.format(prob, rep),
            entry_point='gym_pcgrl.envs.multi_pcgrl_env:MAPcgrlEnv',
            kwargs={"prob": prob, "rep": f'marl_{rep}'}
        )


