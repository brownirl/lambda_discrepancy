"""
Starting point for all environments. We follow the Gymnasium gym.Env API.
"""
from pathlib import Path

import gymnax
from jax import random

from definitions import ROOT_DIR

from .battleship import Battleship
from .classic import load_pomdp, load_jax_pomdp
from .pocman import PocMan
from .rocksample import RockSample
from .tmaze import TMaze

from .jax_wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    MaskObservationWrapper,
    LogWrapper,
    ClipAction,
    VecEnv,
    NormalizeVecReward,
    NormalizeVecObservation,
    ActionConcatWrapper
)


def get_gymnax_env(env_name: str,
                   rand_key: random.PRNGKey,
                   normalize_env: bool = False,
                   gamma: float = 0.99,
                   action_concat: bool = False):

    mask_dims = None
    envs_dir = Path(ROOT_DIR) / 'porl' / 'envs'

    pomdp_dir = envs_dir / 'classic' / 'POMDP'
    pomdp_files = [pd.stem for pd in pomdp_dir.iterdir()]

    fo_pomdp = 'fully_observable' in env_name
    if fo_pomdp:
        env_name = env_name.split('_')[-1]

    if env_name.startswith('tmaze_'):
        hallway_length = int(env_name.split('_')[-1])
        env = TMaze(hallway_length=hallway_length)
        env_params = env.default_params

    elif env_name in pomdp_files:
        env = load_jax_pomdp(env_name, fully_observable=fo_pomdp)
        if hasattr(env, 'gamma'):
            gamma = env.gamma
        env_params = env.default_params
    elif env_name.startswith('battleship'):
        rows = cols = 10
        ship_lengths = (5, 4, 3, 2)
        if env_name == 'battleship_5':
            rows = cols = 5
            ship_lengths = (3, 2)
        elif env_name == 'battleship_3':
            rows = cols = 3
            ship_lengths = (2, )

        env = Battleship(rows=rows, cols=cols, ship_lengths=ship_lengths)
        env_params = env.default_params

    elif env_name == 'pocman':
        env = PocMan()
        env_params = env.default_params

    elif 'rocksample' in env_name:  # [rocksample, rocksample_15_15]

        if len(env_name.split('_')) > 1:
            config_path = Path(ROOT_DIR, 'porl', 'envs', 'configs', f'{env_name}_config.json')
            env = RockSample(rand_key, config_path=config_path)
        else:
            env = RockSample(rand_key)
        env_params = env.default_params

        # We normalize for rocksample!
        # normalize_env = True

    else:
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)

    if hasattr(env, 'gamma'):
        print(f"Overwriting args gamma {gamma} with env gamma {env.gamma}.")
        gamma = env.gamma

    if action_concat:
        env = ActionConcatWrapper(env)

    env = LogWrapper(env, gamma=gamma)

    if mask_dims is not None:
        env = MaskObservationWrapper(env, mask_dims=mask_dims)

    # Vectorize our environment
    env = VecEnv(env)
    if normalize_env:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)
    elif 'rocksample' in env_name:
        env = NormalizeVecReward(env, gamma)
    return env, env_params



