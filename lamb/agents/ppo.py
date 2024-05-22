from typing import NamedTuple

import jax
from jax import numpy as jnp
import numpy as np


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def env_step(runner_state, unused, network, env, env_params):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    value, action, log_prob = (
        value.squeeze(0),
        action.squeeze(0),
        log_prob.squeeze(0),
    )

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition
