from functools import partial
import math
from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
from jax import nn, random
import numpy as np


def normalize(arr: np.ndarray, axis=-1) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        normalized_arr = arr / np.expand_dims(arr.sum(axis), axis)
    normalized_arr = np.nan_to_num(normalized_arr)
    return normalized_arr


def unif_simplex(n: int):
    """
    Samples uniformly random vector from a simplex in n-dimensions.
    Taken from https://stackoverflow.com/questions/65154622/sample-uniformly-at-random-from-a-simplex-in-python
    """
    logits = np.random.exponential(scale=1., size=n)
    return logits / sum(logits)


def glorot_init(shape: Sequence[int], scale: float = 0.5) -> jnp.ndarray:
    return np.random.normal(size=shape) * scale


# @jit
def all_t_discounted_returns(discounts: np.ndarray, rewards: np.ndarray):
    """
    Calculating discounted returns for every time step.
    Taken from
    https://github.com/samlobel/grl/blob/4f7837ec7ea48d9f167420ac59c8d65b1d332161/grl/baselines/rnn_agent.py#L39
    """
    # Since we're calculating returns for each reward, we need to step timestep back by one
    shunted_discounts = np.concatenate([np.ones_like(discounts[0:1]), discounts[:-1]])

    # Calculate discounting starting at t=0
    t0_discounts = np.cumprod(shunted_discounts)

    # calculate discounted return for each time step
    discounted_rewards = rewards * t0_discounts

    # Calculate t = 0 discounted returns.
    overdiscounted_returns = np.cumsum(discounted_rewards[::-1])[::-1]
    returns = overdiscounted_returns / np.maximum(t0_discounts, 1e-10)
    return returns


def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None, zero_mask: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets)**2

    # if we have a zero mask, we take the mean over non-zero elements.
    if zero_mask is not None:
        masked_squared_diff = squared_diff * zero_mask
        return jnp.sum(masked_squared_diff) * (1 / zero_mask.sum())

    return jnp.mean(squared_diff)


def euclidian_dist(arr1: jnp.ndarray, arr2: jnp.ndarray):
    return jnp.linalg.norm(arr1 - arr2, 2)


def unif_simplex(n: int):
    """
    Samples uniformly random vector from a simplex in n-dimensions.
    Taken from https://stackoverflow.com/questions/65154622/sample-uniformly-at-random-from-a-simplex-in-python
    """
    logits = np.random.exponential(scale=1., size=n)
    return logits / sum(logits)


def generate_random_uniform_memory_fn(n_mem_states: int, n_obs: int, n_actions: int):
    T_mem = np.zeros((n_actions, n_obs, n_mem_states, n_mem_states))

    for a in range(n_actions):
        for ob in range(n_obs):
            for m in range(n_mem_states):
                T_mem[a, ob, m] = unif_simplex(n_mem_states)
    return T_mem


def generate_random_discrete_memory_fn(n_mem_states: int, n_obs: int, n_actions: int):
    unif_mem = generate_random_uniform_memory_fn(n_mem_states, n_obs, n_actions)
    discrete_mem = (np.expand_dims(np.max(unif_mem, axis=-1), -1) == unif_mem).astype(float)
    return discrete_mem


def reverse_softmax(dists: jnp.ndarray, eps: float = 1e-20) -> jnp.ndarray:
    """
    A fast and efficient way to turn a distribution
    into softmax-able parameters, where
    dist = softmax(reverse_softmax(dist))
    :param dists: distribution to un-softmax. We assume that the last dimension sums to 1.
    """
    # c = jnp.log(jnp.exp(dists).sum(axis=-1))
    # params = jnp.log(dists) + c
    params = jnp.log(dists + jnp.array(eps, dtype=dists.dtype))
    return params

"""
Taken from https://github.com/toshikwa/rljax/blob/master/rljax/util/distribution.py
"""


@jax.jit
def gaussian_log_prob(
        log_std: jnp.ndarray,
        noise: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions.
    """
    return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
        log_std: jnp.ndarray,
        noise: jnp.ndarray,
        action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions and tanh transformation.
    """
    return gaussian_log_prob(log_std, noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)


@jax.jit
def evaluate_gaussian_and_tanh_log_prob(
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
        action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of gaussian distributions and tanh transformation given samples.
    """
    noise = (jnp.arctanh(action) - mean) / (jnp.exp(log_std) + 1e-8)
    return gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian(
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
        key: jnp.ndarray,
        return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from gaussian distributions.
    """
    std = jnp.exp(log_std)
    noise = random.normal(key, std.shape)
    action = mean + noise * std
    if return_log_pi:
        return action, gaussian_log_prob(log_std, noise).sum(axis=1, keepdims=True)
    else:
        return action


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian_and_tanh(
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
        key: jnp.ndarray,
        return_log_pi: bool = True,
) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Sample from gaussian distributions and tanh transforamation.
    """
    std = jnp.exp(log_std)
    noise = random.normal(key, std.shape)
    action = jnp.tanh(mean + noise * std)
    if return_log_pi:
        return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)
    else:
        return action


def positional_choice(key, a, p):
    return jax.random.choice(key, a, p=p)

