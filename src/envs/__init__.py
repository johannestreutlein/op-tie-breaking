from functools import partial
#rom smac.env import MultiAgentEnv, StarCraft2Env
from .levergame import LeverGame
from .multiagentenv import MultiAgentEnv

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
#REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

REGISTRY["twostagelevergame"] = partial(env_fn, env=LeverGame)
REGISTRY["asymmetriclevergame"] = partial(env_fn, env=LeverGame)

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
