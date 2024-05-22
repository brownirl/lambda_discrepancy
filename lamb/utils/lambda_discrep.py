import jax.numpy as jnp
from typing import Callable
from functools import partial

from jax.nn import softmax

from lamb.mdp import POMDP
from lamb.memory.analytical import memory_cross_product
from lamb.utils.loss import discrep_loss, mstd_err, value_error
from lamb.utils.policy_eval import analytical_pe

def lambda_discrep_measures(pomdp: POMDP, pi: jnp.ndarray, discrep_loss_fn: Callable = None):
    if discrep_loss_fn is None:
        discrep_loss_fn = partial(discrep_loss, value_type='q', error_type='l2', alpha=1.)

    state_vals, _, _, _ = analytical_pe(pi, pomdp)
    discrep, mc_vals, td_vals = discrep_loss_fn(pi, pomdp)

    measures = {
        'discrep': discrep,
        'mc_vals_q': mc_vals['q'],
        'td_vals_q': td_vals['q'],
        'mc_vals_v': mc_vals['v'],
        'td_vals_v': td_vals['v'],
        'state_vals_v': state_vals['v'],
        'state_vals_q': state_vals['q'],
        'p0': pomdp.p0.copy()
    }
    return measures

def augment_and_log_all_measures(mem_params: jnp.ndarray, pomdp: POMDP, mem_aug_pi_params: jnp.ndarray) -> dict:
    mem_pomdp = memory_cross_product(mem_params, pomdp)
    return log_all_measures(mem_pomdp, mem_aug_pi_params)

def log_all_measures(pomdp: POMDP, pi_params: jnp.ndarray) -> dict:
    """
    Logs a few things:
    1. values
    2. lambda discrep
    3. mstde
    4. value error
    Note: we assume pi_params is already augmented.
    """
    # Lambda discrepancy
    pi = softmax(pi_params, axis=-1)
    discrep, mc_vals, td_vals = discrep_loss(pi, pomdp, error_type='l2', value_type='q', alpha=1.)

    # MSTDE
    mstde_loss, vals, _ = mstd_err(pi, pomdp, error_type='l2', residual=False)
    mstde_res_loss, vals, _ = mstd_err(pi, pomdp, error_type='l2', residual=True)

    # Value error
    value_err, state_vals, expanded_obs_vals = value_error(pi, pomdp, value_type='q', error_type='l2',
                                                           lambda_=0.)

    value_dict = {
        'mc_vals': mc_vals,
        'td_vals': td_vals,
        'state_vals': state_vals,
        'p0': pomdp.p0.copy()
    }

    return {'errors':
                {'ld': discrep, 'mstde': mstde_loss, 'mstde_residual': mstde_res_loss, 'value': value_err},
            'values': value_dict
            }
