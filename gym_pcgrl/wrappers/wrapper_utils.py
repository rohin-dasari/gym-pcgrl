import gym


"""
Unpack a nested object and return the object that matches the type specefied by
name

Parameters
----------

env: gym.env or gym.wrapper
name: str
"""
def get_object(env, name):
    # should probably assert the type for env here to avoid infinite recursion
    if name in str(type(env)):
        return env
    get_object(env.env, name)

"""
"""
def get_env(game):
    if isinstance(game, str):
        env = gym.make(game)
    else:
        env = game
    return env
