import jax.numpy as jnp
from jax import nn, lax, jit
from functools import partial

from lamb.utils.mdp import functional_get_occupancy, get_p_s_given_o, functional_create_td_model
from lamb.utils.policy_eval import analytical_pe, lstdq_lambda, functional_solve_mdp
from lamb.utils.policy import deconstruct_aug_policy
from lamb.utils.math import reverse_softmax
from lamb.memory import memory_cross_product
from lamb.mdp import MDP, POMDP
"""
The following few functions are loss function w.r.t. memory parameters, mem_params.
"""

def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None, zero_mask: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets)**2

    # if we have a zero mask, we take the mean over non-zero elements.
    if zero_mask is not None:
        masked_squared_diff = squared_diff * zero_mask
        return jnp.sum(masked_squared_diff) * (1 / zero_mask.sum())

    return jnp.mean(squared_diff)

def seq_sarsa_loss(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, gamma: jnp.ndarray,
                   next_q: jnp.ndarray, next_a: jnp.ndarray):
    """
    sequential version of sarsa loss
    First axis of all tensors are the sequence length.
    """
    target = r + gamma * next_q[jnp.arange(next_a.shape[0]), next_a]
    target = lax.stop_gradient(target)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target

def seq_sarsa_mc_loss(q: jnp.ndarray, a: jnp.ndarray, ret: jnp.ndarray):
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - ret

def seq_sarsa_lambda_discrep(q_td: jnp.ndarray, q_mc: jnp.ndarray, a: jnp.ndarray):
    q_vals_td = q_td[jnp.arange(a.shape[0]), a]
    q_vals_mc = q_mc[jnp.arange(a.shape[0]), a]
    q_vals_mc = lax.stop_gradient(q_vals_mc)

    return q_vals_td - q_vals_mc

def weight_and_sum_discrep_loss(diff: jnp.ndarray,
                                occupancy: jnp.ndarray,
                                pi: jnp.ndarray,
                                pomdp: POMDP,
                                value_type: str = 'q',
                                error_type: str = 'l2',
                                alpha: float = 1.,
                                flip_count_prob: bool = False):
    c_o = occupancy @ pomdp.phi
    count_o = c_o / c_o.sum()

    if flip_count_prob:
        count_o = nn.softmax(-count_o)

    count_mask = (1 - jnp.isclose(count_o, 0, atol=1e-12)).astype(float)
    uniform_o = (jnp.ones(pi.shape[0]) / count_mask.sum()) * count_mask
    # uniform_o = jnp.ones(pi.shape[0])

    p_o = alpha * uniform_o + (1 - alpha) * count_o

    weight = (pi * p_o[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()
    return loss

@partial(jit,
         static_argnames=[
             'value_type', 'error_type', 'lambda_0', 'lambda_1', 'alpha', 'flip_count_prob'
         ])
def discrep_loss(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False): # initialize static args
    if lambda_0 == 0. and lambda_1 == 1.:
        _, mc_vals, td_vals, info = analytical_pe(pi, pomdp)
        lambda_0_vals = td_vals
        lambda_1_vals = mc_vals
    else:
        # TODO: info here only contains state occupancy, which should lambda agnostic.
        lambda_0_v_vals, lambda_0_q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_0)
        lambda_1_v_vals, lambda_1_q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_1)
        lambda_0_vals = {'v': lambda_0_v_vals, 'q': lambda_0_q_vals}
        lambda_1_vals = {'v': lambda_1_v_vals, 'q': lambda_1_q_vals}

    diff = lambda_1_vals[value_type] - lambda_0_vals[value_type]
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    loss = weight_and_sum_discrep_loss(diff,
                                       c_s,
                                       pi,
                                       pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)

    return loss, lambda_1_vals, lambda_0_vals

def mem_discrep_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False): # initialize with partial
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = discrep_loss(pi,
                              mem_aug_pomdp,
                              value_type,
                              error_type,
                              lambda_0=lambda_0,
                              lambda_1=lambda_1,
                              alpha=alpha,
                              flip_count_prob=flip_count_prob)
    return loss

def obs_space_mem_discrep_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0.,
        lambda_1: float = 1.,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    """
    Memory discrepancy loss on the TD(0) estimator over observation space.

    """
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)

    n_mem_states = mem_params.shape[-1]
    pi_obs = pi[::n_mem_states]
    mem_lambda_0_v_vals, mem_lambda_0_q_vals, mem_info = lstdq_lambda(pi,
                                                                      mem_aug_pomdp,
                                                                      lambda_=lambda_0)
    lambda_1_v_vals, lambda_1_q_vals, info = lstdq_lambda(pi_obs, pomdp, lambda_=lambda_1)

    counts_mem_aug_flat_obs = mem_info['occupancy'] @ mem_aug_pomdp.phi
    counts_mem_aug_flat = jnp.einsum('i,ij->ij', counts_mem_aug_flat_obs, pi).T # A x OM

    counts_mem_aug = counts_mem_aug_flat.reshape(pomdp.action_space.n, -1,
                                                 n_mem_states) # A x O x M

    denom_counts_mem_aug_unmasked = counts_mem_aug.sum(axis=-1, keepdims=True)
    denom_mask = (denom_counts_mem_aug_unmasked == 0).astype(float)
    denom_counts_mem_aug = denom_counts_mem_aug_unmasked + denom_mask
    prob_mem_given_oa = counts_mem_aug / denom_counts_mem_aug

    unflattened_lambda_0_q_vals = mem_lambda_0_q_vals.reshape(pomdp.action_space.n, -1,
                                                              n_mem_states)
    reformed_lambda_0_q_vals = (unflattened_lambda_0_q_vals * prob_mem_given_oa).sum(axis=-1)

    lambda_1_vals = {'v': lambda_1_v_vals, 'q': lambda_1_q_vals}
    lambda_0_vals = {
        'v': (reformed_lambda_0_q_vals * pi_obs.T).sum(0),
        'q': reformed_lambda_0_q_vals
    }

    diff = lambda_1_vals[value_type] - lambda_0_vals[value_type]

    # set terminal counts to 0
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)

    loss = weight_and_sum_discrep_loss(diff,
                                       c_s,
                                       pi_obs,
                                       pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)
    return loss

def policy_discrep_loss(pi_params: jnp.ndarray,
                        pomdp: POMDP,
                        value_type: str = 'q',
                        error_type: str = 'l2',
                        lambda_0: float = 0.,
                        lambda_1: float = 1.,
                        alpha: float = 1.,
                        flip_count_prob: bool = False): # initialize with partial
    pi = nn.softmax(pi_params, axis=-1)
    loss, mc_vals, td_vals = discrep_loss(pi,
                                          pomdp,
                                          value_type,
                                          error_type,
                                          lambda_0=lambda_0,
                                          lambda_1=lambda_1,
                                          alpha=alpha,
                                          flip_count_prob=flip_count_prob)
    return loss, (mc_vals, td_vals)

def pg_objective_func(pi_params: jnp.ndarray, pomdp: POMDP):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    pi_abs = nn.softmax(pi_params, axis=-1)
    pi_ground = pomdp.phi @ pi_abs

    # Terminals have p(S) = 0.
    occupancy = functional_get_occupancy(pi_ground, pomdp) * (1 - pomdp.terminal_mask)

    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, occupancy)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, pomdp)
    td_model = MDP(T_obs_obs, R_obs_obs, pomdp.p0 @ pomdp.phi, gamma=pomdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_abs, td_model)
    p_init_obs = pomdp.p0 @ pomdp.phi
    return jnp.dot(p_init_obs, td_v_vals), (td_v_vals, td_q_vals)

def augmented_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):
    """
    Policy gradient objective function:
    sum_{s_0} p(s_0) v_pi(s_0)
    """
    augmented_pi_probs = nn.softmax(augmented_pi_params)
    mem_probs, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs)
    mem_logits = reverse_softmax(mem_probs)
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)
    return pg_objective_func(action_policy_probs, mem_aug_mdp)

def mem_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):
    augmented_pi_probs = nn.softmax(augmented_pi_params, axis=-1)
    mem_logits, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs)
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)
    O, M, A = action_policy_probs.shape
    return pg_objective_func(reverse_softmax(action_policy_probs).reshape(O * M, A), mem_aug_mdp)

def unrolled_mem_pg_objective_func(augmented_pi_params: jnp.ndarray, pomdp: POMDP):# O, M, AM
    augmented_pi_probs_unflat = nn.softmax(augmented_pi_params, axis=-1)
    mem_logits, action_policy_probs = deconstruct_aug_policy(augmented_pi_probs_unflat)# A,O,M->M ; O,M->A
    O, M, A = action_policy_probs.shape
    mem_aug_mdp = memory_cross_product(mem_logits, pomdp)

    pi_probs = action_policy_probs.reshape(O * M, A)
    aug_pi_probs = augmented_pi_probs_unflat.reshape(O * M, A * M)

    pi_ground = mem_aug_mdp.phi @ pi_probs  # pi: (S * M, A)
    occupancy = functional_get_occupancy(pi_ground, mem_aug_mdp)  # eta: S * M
    om_occupancy = occupancy @ mem_aug_mdp.phi  # om_eta: O * M

    # Calculate our Q vals over A x O * M
    p_pi_of_s_given_o = get_p_s_given_o(mem_aug_mdp.phi, occupancy) # P(SM|OM)
    T_obs_obs, R_obs_obs = functional_create_td_model(p_pi_of_s_given_o, mem_aug_mdp) # T(O'M'|O,M,A); R(O,M,A)
    td_model = MDP(T_obs_obs, R_obs_obs, mem_aug_mdp.p0 @ mem_aug_mdp.phi, gamma=mem_aug_mdp.gamma)
    td_v_vals, td_q_vals = functional_solve_mdp(pi_probs, td_model)  # q: (A, O * M)

    # expand over A * M
    mem_probs = nn.softmax(mem_logits, axis=-1) # (A, O, M, M)
    mem_probs_omam = jnp.moveaxis(mem_probs, 0, -2)  # (O, M, A, M)
    mem_probs_omam = mem_probs_omam.reshape(O * M, A, M)  # (OM, A, M)
                #(OM,A,M)                        #(A, OM)^T => (OM, A) => (OM, A, 1)
    am_q_vals = mem_probs_omam * jnp.expand_dims(td_q_vals.T, -1)  # (OM, A, M)
    am_q_vals = am_q_vals.reshape(O * M, A * M)  # (OM, AM)

    # Don't take gradients over eta or Q
    weighted_am_q_vals = jnp.expand_dims(om_occupancy, -1) * am_q_vals
    weighted_am_q_vals = lax.stop_gradient(weighted_am_q_vals)
    return (weighted_am_q_vals * aug_pi_probs).sum(), (td_v_vals, td_q_vals)

def mem_bellman_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0,
        lambda_1: float = 1.,  # NOT CURRENTLY USED!
        residual: bool = False,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = bellman_loss(pi,
                              mem_aug_pomdp,
                              value_type,
                              error_type,
                              alpha,
                              lambda_=lambda_0,
                              residual=residual,
                              flip_count_prob=flip_count_prob)
    return loss

@partial(jit, static_argnames=['value_type', 'error_type', 'alpha', 'residual', 'flip_count_prob'])
def bellman_loss(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        value_type: str = 'q',
        error_type: str = 'l2',
        alpha: float = 1.,
        lambda_: float = 0.,
        residual: bool = False,
        flip_count_prob: bool = False): # initialize static args

    # First, calculate our TD(0) Q-values
    v_vals, q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_)
    vals = {'v': v_vals, 'q': q_vals}
    assert value_type == 'q'

    c_s = info['occupancy']
    # Make TD(0) model
    p_pi_of_s_given_o = get_p_s_given_o(pomdp.phi, c_s)
    T_aoo, R_aoo = functional_create_td_model(p_pi_of_s_given_o, pomdp)

    # Tensor for AxOxOxA (obs action to obs action)
    T_pi_aooa = jnp.einsum('ijk,kl->ijkl', T_aoo, pi)
    R_ao = R_aoo.sum(axis=-1)

    # Calculate the expected next value for each (a, o) pair
    expected_next_V_given_ao = jnp.einsum('ijkl,kl->ij', T_pi_aooa, q_vals.T)  # A x O

    # Our Bellman error
    target = R_ao + pomdp.gamma * expected_next_V_given_ao
    if not residual:
        target = lax.stop_gradient(target)
    diff = target - q_vals

    # R_s_o = pomdp.R @ pomdp.phi  # A x S x O

    # expanded_R_s_o = R_s_o[..., None].repeat(pomdp.action_space.n, axis=-1)  # A x S x O x A

    # repeat the Q-function over A x O
    # Multiply that with p(O', A' | s, a) and sum over O' and A' dimensions.
    # P(O' | s, a) = T @ phi, P(A', O' | s, a) = P(O' | s, a) * pi (over new dimension)
    # pr_o = pomdp.T @ pomdp.phi
    # pr_o_a = jnp.einsum('ijk,kl->ijkl', pr_o, pi)
    # expected_next_Q = jnp.einsum('ijkl,kl->ijkl', pr_o_a, q_vals.T)
    # expanded_Q = jnp.expand_dims(
    #     jnp.expand_dims(q_vals.T, 0).repeat(pr_o_a.shape[1], axis=0),
    #     0).repeat(pr_o_a.shape[0], axis=0)  # Repeat over OxA -> O x A x O x A
    # diff = expanded_R_s_o + pomdp.gamma * expected_next_Q - expanded_Q


    # set terminal counts to 0
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    loss = weight_and_sum_discrep_loss(diff, c_s, pi, pomdp,
                                       value_type=value_type,
                                       error_type=error_type,
                                       alpha=alpha,
                                       flip_count_prob=flip_count_prob)

    return loss, vals, vals

def mem_tde_loss(
        mem_params: jnp.ndarray,
        pi: jnp.ndarray,
        pomdp: POMDP, # input non-static arrays
        value_type: str = 'q',
        error_type: str = 'l2',
        lambda_0: float = 0,
        lambda_1: float = 1.,  # NOT CURRENTLY USED!
        residual: bool = False,
        alpha: float = 1.,
        flip_count_prob: bool = False):
    mem_aug_pomdp = memory_cross_product(mem_params, pomdp)
    loss, _, _ = mstd_err(pi,
                          mem_aug_pomdp,
                          # value_type,
                          error_type,
                          # alpha,
                          lambda_=lambda_0,
                          residual=residual,
                          # flip_count_prob=flip_count_prob
                          )
    return loss

@partial(jit, static_argnames=['error_type', 'residual'])
def mstd_err(
        pi: jnp.ndarray,
        pomdp: POMDP, # non-state args
        error_type: str = 'l2',
        lambda_: float = 0.,
        residual: bool = False): # initialize static args
    # First, calculate our TD(0) Q-values
    v_vals, q_vals, info = lstdq_lambda(pi, pomdp, lambda_=lambda_)
    vals = {'v': v_vals, 'q': q_vals}
    # assert value_type == 'q'

    n_states = pomdp.base_mdp.state_space.n
    n_obs = pomdp.observation_space.n
    n_actions = pomdp.action_space.n

    # Project Q-values from observations to states
    q_soa = q_vals.T[None, :, :].repeat(n_states, axis=0)
    # q_sa = pomdp.phi @ q_vals.T

    # Calculate all "potential" next q-values by repeating at the front.
    qp_soasoa = (
        q_soa[None, None, None, ...]
        .repeat(n_states, axis=0)
        .repeat(n_obs, axis=1)
        .repeat(n_actions, axis=2)
    )

    # Calculate all current q-values, repeating at the back.
    q_soasoa = (
        q_soa[..., None, None, None]
        .repeat(n_states, axis=-3)
        .repeat(n_obs, axis=-2)
        .repeat(n_actions, axis=-1)
    )

    # Expanded and repeated reward tensor
    R_sas = jnp.swapaxes(pomdp.base_mdp.R, 0, 1)
    R_soasoa = (
        R_sas[:, None, :, :, None, None]
        .repeat(n_obs, axis=1)
        .repeat(n_obs, axis=-2)
        .repeat(n_actions, axis=-1)
    )

    # Calculate targets (R + gamma * Q') and stop_grad it.
    targets = R_soasoa + pomdp.gamma * qp_soasoa
    targets = lax.cond(residual, lambda x: x, lambda x: lax.stop_gradient(x), targets)

    # Compute errors
    tde_soasoa = (targets - q_soasoa)

    if error_type == 'l2':
        mag_tde_soasoa = (tde_soasoa**2)
    elif error_type == 'abs':
        mag_tde_soasoa = jnp.abs(tde_soasoa)
    else:
        raise NotImplementedError(f"Error {error_type} not implemented yet in mem_loss fn.")

    # set terminal count to 0 and compute Pr(s)
    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    pr_s = c_s / c_s.sum()

    # Retrieve Pr(o|s), Pr(s'|s,a)
    phi_so = pomdp.phi
    T_sas = jnp.swapaxes(pomdp.base_mdp.T, 0, 1)

    # Compute Pr(s,o,a,s',o',a')
    pr_s_soasoa =       pr_s[   :, None, None, None, None, None] # Pr(s)
    phi_soasoa =      phi_so[   :,    :, None, None, None, None] # Pr(o|s)
    pi_soasoa =           pi[None,    :,    :, None, None, None] # Pr(a|o)
    T_soasoa =         T_sas[   :, None,    :,    :, None, None] # Pr(s'|s,a)
    next_phi_soasoa = phi_so[None, None, None,    :,    :, None] # Pr(o'|s')
    next_pi_soasoa =      pi[None, None, None, None,    :,    :] # Pr(a'|o')
    # Pr(s,o,a,s',o',a') = Pr(s) * Pr(o|s) * Pr(a|o) * Pr(s'|s,a) * Pr(o'|s') * Pr(a'|o')
    pr_soasoa = (
        pr_s_soasoa * phi_soasoa * pi_soasoa * T_soasoa * next_phi_soasoa * next_pi_soasoa
    )

    # Reweight squared errors according to Pr(s,o,a,s',o',a')
    weighted_sq_tde_soasoa = pr_soasoa * mag_tde_soasoa

    # Sum over all dimensions
    loss = weighted_sq_tde_soasoa.sum()
    return loss, vals, vals

@partial(jit, static_argnames=['value_type', 'error_type', 'lambda_'])
def value_error(pi: jnp.ndarray,
                pomdp: POMDP,
                value_type: str = 'q',
                error_type: str = 'l2',
                lambda_: float = 1.0):
    state_vals, mc_vals, td_vals, info = analytical_pe(pi, pomdp)
    if lambda_ == 0.0:
        obs_vals = td_vals
    elif lambda_ == 1.0:
        obs_vals = mc_vals
    else:
        v_vals, q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_)
        obs_vals = {'v': v_vals, 'q': q_vals}

    # Expand observation (q-)value function to state (q-)value function
    # (a,o) @ (o,s) => (a,s);  (o,) @ (o,s) => (s,)
    expanded_obs_vals = obs_vals[value_type] @ pomdp.phi.T
    diff = state_vals[value_type] - expanded_obs_vals

    c_s = info['occupancy'] * (1 - pomdp.terminal_mask)
    p_s = c_s / c_s.sum()

    pi_s = pomdp.phi @ pi
    weight = (pi_s * p_s[:, None]).T
    if value_type == 'v':
        weight = weight.sum(axis=0)
    weight = lax.stop_gradient(weight)

    if error_type == 'l2':
        unweighted_err = (diff**2)
    elif error_type == 'abs':
        unweighted_err = jnp.abs(diff)
    else:
        raise NotImplementedError(
            f"error_type {error_type} not implemented yet in value_error fn.")

    weighted_err = weight * unweighted_err
    if value_type == 'q':
        weighted_err = weighted_err.sum(axis=0)

    loss = weighted_err.sum()

    return loss, state_vals, expanded_obs_vals
