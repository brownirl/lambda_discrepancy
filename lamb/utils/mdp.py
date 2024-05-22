from jax import jit
import jax.numpy as jnp
import numpy as np
from typing import Union

from lamb.mdp import MDP, POMDP


@jit
def functional_get_occupancy(pi_ground: jnp.ndarray, mdp: Union[MDP, POMDP]):
    Pi_pi = pi_ground.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)

    # A*C_pi(s) = b
    # A = (I - \gamma (T^π)^T)
    # b = P_0
    A = jnp.eye(mdp.state_space.n) - mdp.gamma * T_pi.transpose()
    b = mdp.p0
    return jnp.linalg.solve(A, b)

def pomdp_get_occupancy(pi: jnp.ndarray, pomdp: POMDP):
    pi_ground = pomdp.phi @ pi
    return functional_get_occupancy(pi_ground, pomdp)

@jit
def get_p_s_given_o(phi: jnp.ndarray, occupancy: jnp.ndarray):
    repeat_occupancy = jnp.repeat(occupancy[..., None], phi.shape[-1], -1)

    # Q vals
    p_of_o_given_s = phi.astype(float)
    w = repeat_occupancy * p_of_o_given_s

    p_pi_of_s_given_o = w / (w.sum(axis=0) + 1e-10)
    return p_pi_of_s_given_o

@jit
def functional_create_td_model(p_pi_of_s_given_o: jnp.ndarray, pomdp: POMDP):
    # creates an (n_obs * n_obs) x 2 array of all possible observation to observation pairs.
    # we flip here so that we have curr_obs, next_obs (order matters).
    obs_idx_product = jnp.flip(
        jnp.dstack(jnp.meshgrid(jnp.arange(pomdp.phi.shape[-1]),
                                jnp.arange(pomdp.phi.shape[-1]))).reshape(-1, 2), -1)

    # this gives us (n_obs * n_obs) x states x 1 and (n_obs * n_obs) x 1 x states
    curr_s_given_o = p_pi_of_s_given_o[:, obs_idx_product[:, 0]].T[..., None]
    next_o_given_s = jnp.expand_dims(pomdp.phi[:, obs_idx_product[:, 1]].T, 1)

    # outer product here
    o_to_next_o = jnp.expand_dims(curr_s_given_o * next_o_given_s, 1)

    # This is p(o, s, a, s', o')
    # the probability that o goes to o', via each path (s, a) -> s'.
    # Shape is (n_obs * n_obs) x |A| x |S| x |S|
    T_contributions = pomdp.T * o_to_next_o

    # |A| x (n_obs * n_obs)
    T_obs_obs_flat = T_contributions.sum(-1).sum(-1).T

    # |A| x n_obs x n_obs
    T_obs_obs = T_obs_obs_flat.reshape(pomdp.T.shape[0], pomdp.phi.shape[-1], pomdp.phi.shape[-1])

    # You want everything to sum to one
    denom = T_obs_obs_flat.T[..., None, None]
    denom_no_zero = denom + (denom == 0).astype(denom.dtype)

    R_contributions = (pomdp.R * T_contributions) / denom_no_zero
    R_obs_obs_flat = R_contributions.sum(-1).sum(-1).T
    R_obs_obs = R_obs_obs_flat.reshape(pomdp.R.shape[0], pomdp.phi.shape[-1], pomdp.phi.shape[-1])

    return T_obs_obs, R_obs_obs

@jit
def get_td_model(pomdp: POMDP, pi: jnp.ndarray):
    pi_state = pomdp.phi @ pi
    occupancy = functional_get_occupancy(pi_state, pomdp)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    return functional_create_td_model(p_pi_of_s_given_o, pomdp)

# @jit
def all_t_discounted_returns(discounts: np.ndarray, rewards: np.ndarray):
    """
    Calculating discounted returns for every time step.
    Taken from
    https://github.com/samlobel/grl/blob/4f7837ec7ea48d9f167420ac59c8d65b1d332161/grl/baselines/rnn_agent.py#L39
    FOR SOME REASON! There's a memory leak when we use jnp here.
    I suspect it's in either the jnp.cumsum or jnp.cumprod.
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

def to_dict(T, R, gamma, p0, phi, Pi_phi, Pi_phi_x=None):
    return {
        'T': T,
        'R': R,
        'gamma': gamma,
        'p0': p0,
        'phi': phi,
        'Pi_phi': Pi_phi,
        'Pi_phi_x': Pi_phi_x,
    }

def get_perf(info: dict):
    return (info['state_vals_v'] * info['p0']).sum()

