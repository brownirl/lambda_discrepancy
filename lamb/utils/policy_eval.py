import jax.numpy as jnp
from functools import partial
from typing import Union
from jax import jit

from lamb.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from lamb.mdp import MDP, POMDP

@jit
def analytical_pe(pi_obs: jnp.ndarray, pomdp: POMDP):
    # observation policy, but expanded over states
    pi_state = pomdp.phi @ pi_obs

    # MC*
    state_v, state_q = functional_solve_mdp(pi_state, pomdp)
    state_vals = {'v': state_v, 'q': state_q}

    occupancy = functional_get_occupancy(pi_state, pomdp)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    mc_vals = functional_solve_pomdp(state_q, p_pi_of_s_given_o, pi_obs)

    # TD
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, pomdp)
    td_model = MDP(T_obs_obs, R_obs_obs, pomdp.p0 @ pomdp.phi, gamma=pomdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_obs, td_model)
    td_vals = {'v': td_v_vals, 'q': td_q_vals}

    return state_vals, mc_vals, td_vals, {
        'occupancy': occupancy,
        'R_obs_obs': R_obs_obs,
        'T_obs_obs': T_obs_obs
    }

@jit
def functional_solve_mdp(pi: jnp.ndarray, mdp: Union[MDP, POMDP]):
    """
    Solves for V using linear equations.
    For all s, V_pi(s) = sum_s' sum_a[T(s'|s,a) * pi(a|s) * (R(s,a,s') + gamma * V_pi(s'))]
    """
    Pi_pi = pi.transpose()[..., None]
    T_pi = (Pi_pi * mdp.T).sum(axis=0) # T^π(s'|s)
    R_pi = (Pi_pi * mdp.T * mdp.R).sum(axis=0).sum(axis=-1) # R^π(s)

    # A*V_pi(s) = b
    # A = (I - \gamma (T^π))
    # b = R^π
    A = (jnp.eye(mdp.state_space.n) - mdp.gamma * T_pi)
    b = R_pi
    v_vals = jnp.linalg.solve(A, b)

    R_sa = (mdp.T * mdp.R).sum(axis=-1) # R(s,a)
    q_vals = (R_sa + (mdp.gamma * mdp.T @ v_vals))

    return v_vals, q_vals

@jit
def functional_solve_pomdp(mdp_q_vals: jnp.ndarray, p_pi_of_s_given_o: jnp.ndarray,
                           pi_abs: jnp.ndarray):
    # Q vals
    pomdp_q_vals = mdp_q_vals @ p_pi_of_s_given_o

    # V vals
    pomdp_v_vals = (pomdp_q_vals * pi_abs.T).sum(0)

    return {'v': pomdp_v_vals, 'q': pomdp_q_vals}

@partial(jit, static_argnames='lambda_')
def lstdq_lambda(pi: jnp.ndarray, pomdp: Union[MDP, POMDP], lambda_: float = 0.9):
    """Solve for V, Q using LSTD(λ)

    For the definition of LSTD(λ) see https://arxiv.org/pdf/1405.3229.pdf

    We replace state features with state-action features as described in section 2 of
    https://arxiv.org/pdf/1511.08495.pdf
    """
    T_ass = pomdp.T
    R_ass = pomdp.R

    a, s, _ = T_ass.shape
    phi = pomdp.phi if hasattr(pomdp, 'phi') else jnp.eye(s)

    o = phi.shape[1]
    sa = s * a
    oa = o * a

    gamma = pomdp.gamma
    s0 = pomdp.p0

    pi_sa = phi @ pi

    as_0 = (s0[:, None] * pi_sa).T.reshape(sa)

    # State-action to state-action transition kernel
    P_asas = jnp.einsum('ijk,kl->ijlk', T_ass, pi_sa)
    P_as_as = P_asas.reshape((sa, sa))

    # State-action reward function
    R_as = jnp.einsum('ijk,ijk->ij', T_ass, R_ass).reshape((sa, ))

    # Compute the state-action distribution as a diagonal matrix
    I = jnp.eye(sa)
    occupancy_as = jnp.linalg.solve((I - gamma * P_as_as.T), as_0)
    mu = occupancy_as / jnp.sum(occupancy_as)
    D_mu = jnp.diag(mu)

    # Compute the state-action to obs-action observation function
    # (use a copy of phi for each action)
    phi_as_ao = jnp.kron(jnp.eye(a), phi)

    # Solve the linear system for Q(s,a), replacing the state features with state-action features
    #
    # See section 2 of https://arxiv.org/pdf/1511.08495.pdf
    D_eps_ao = 1e-10 * jnp.eye(oa)
    phi_D_mu = phi_as_ao.T @ D_mu
    A = (phi_D_mu @ (I - gamma * P_as_as) @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as,
                                                             phi_as_ao))
    b = phi_D_mu @ jnp.linalg.solve(I - gamma * lambda_ * P_as_as, R_as)
    Q_LSTD_lamb_as = (phi_as_ao @ jnp.linalg.solve(A + D_eps_ao, b)).reshape((a, s))

    # Compute V(s)
    V_LSTD_lamb_s = jnp.einsum('ij,ji->j', Q_LSTD_lamb_as, pi_sa)

    # Convert from states to observations
    occupancy_s = occupancy_as.reshape((a, s)).sum(0)
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy_s)

    Q_LSTD_lamb_ao = Q_LSTD_lamb_as @ p_pi_of_s_given_o
    V_LSTD_lamb_o = V_LSTD_lamb_s @ p_pi_of_s_given_o

    return V_LSTD_lamb_o, Q_LSTD_lamb_ao, {'occupancy': occupancy_s}
