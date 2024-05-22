from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.nn import softmax
import numpy as np
import optax

from lamb.mdp import POMDP
from lamb.utils.policy import construct_aug_policy
from lamb.utils.loss import policy_discrep_loss, pg_objective_func, \
    mem_pg_objective_func, unrolled_mem_pg_objective_func
from lamb.utils.loss import mem_discrep_loss, mem_bellman_loss, mem_tde_loss, obs_space_mem_discrep_loss
from lamb.utils.math import glorot_init, reverse_softmax
from lamb.utils.optimizer import get_optimizer
from lamb.vi import policy_iteration_step

def new_pi_over_mem(pi_params: jnp.ndarray, add_n_mem_states: int,
                    new_mem_pi: str = 'repeat'):
    old_pi_params_shape = pi_params.shape

    pi_params = pi_params.repeat(add_n_mem_states, axis=0)

    if new_mem_pi == 'random':
        # randomly init policy for new memory state
        new_mem_params = glorot_init(old_pi_params_shape)
        pi_params = pi_params.at[1::2].set(new_mem_params)

    return pi_params


class AnalyticalAgent:
    """
    Analytical agent that learns optimal policy params based on an
    analytic policy gradient.
    """
    def __init__(self,
                 pi_params: jnp.ndarray,
                 rand_key: random.PRNGKey,
                 optim_str: str = 'adam',
                 pi_lr: float = 1.,
                 mi_lr: float = 1.,
                 mem_params: jnp.ndarray = None,
                 value_type: str = 'v',
                 error_type: str = 'l2',
                 objective: str = 'discrep',
                 residual: bool = False,
                 lambda_0: float = 0.,
                 lambda_1: float = 1.,
                 alpha: float = 1.,
                 pi_softmax_temp: float = 1,
                 policy_optim_alg: str = 'policy_iter',
                 new_mem_pi: str = 'copy',
                 epsilon: float = 0.1,
                 flip_count_prob: bool = False):
        """
        :param pi_params: Policy parameters
        :param rand_key: Initialized jax PRNG key
        :param mem_params: Memory parameters (optional)
        :param value_type: If we optimize lambda discrepancy, what type of lambda discrepancy do we optimize? (v | q)
        :param error_type: lambda discrepancy error type (l2 | abs)
        :param objective: What objective are we trying to minimize? (discrep | bellman | tde)
        :param pi_softmax_temp: When we take the softmax over pi_params, what is the softmax temperature?
        :param policy_optim_alg: What type of policy optimization do we do? (pi | pg)
            (discrep_max: discrepancy maximization | discrep_min: discrepancy minimization
            | policy_iter: policy iteration | policy_grad: policy gradient)
        :param new_mem_pi: When we do memory iteration and add memory states, how do we initialize the new policy params
                           over the new memory states? (copy | random)
        :param epsilon: (POLICY ITERATION ONLY) When we perform policy iteration, what epsilon do we use?
        :param flip_count_prob: For our memory loss, do we flip our count probabilities??
        """
        self.policy_optim_alg = policy_optim_alg
        self.pi_params = pi_params
        self.og_n_obs = self.pi_params.shape[0]

        self.pg_objective_func = jit(pg_objective_func)
        if self.policy_optim_alg == 'policy_mem_grad':
            self.pg_objective_func = jit(mem_pg_objective_func)
        elif self.policy_optim_alg == 'policy_mem_grad_unrolled':
            self.pg_objective_func = jit(unrolled_mem_pg_objective_func)

        self.policy_iteration_update = jit(policy_iteration_step, static_argnames=['eps'])
        self.epsilon = epsilon

        self.val_type = value_type
        self.error_type = error_type
        self.objective = objective
        self.residual = residual
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.flip_count_prob = flip_count_prob

        self.policy_discrep_objective_func = None
        self.memory_objective_func = None

        self.init_and_jit_objectives()

        self.new_mem_pi = new_mem_pi

        self.mem_params = None
        if mem_params is not None:
            self.mem_params = mem_params

            if self.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
                mem_probs, pi_probs = softmax(self.mem_params, -1), softmax(self.pi_params, -1)
                aug_policy = construct_aug_policy(mem_probs, pi_probs)
                self.pi_aug_params = reverse_softmax(aug_policy)

            self.mi_lr = mi_lr
            self.mem_optim = get_optimizer(optim_str, self.mi_lr)
            self.mem_optim_state = self.mem_optim.init(self.mem_params)

        # initialize optimizers
        self.optim_str = optim_str
        self.pi_lr = pi_lr
        self.pi_optim = get_optimizer(optim_str, self.pi_lr)

        pi_params_to_optimize = self.pi_params
        if self.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
            pi_params_to_optimize = self.pi_aug_params
        self.pi_optim_state = self.pi_optim.init(pi_params_to_optimize)

        self.pi_softmax_temp = pi_softmax_temp

        self.rand_key = rand_key

    def init_and_jit_objectives(self):
        # TODO: delete this?
        if hasattr(self, 'weight_discrep'):
            if self.weight_discrep:
                self.alpha = 0.
            else:
                self.alpha = 1.
            self.flip_count_prob = False
            del self.weight_discrep

        # TODO: set objective switch here as well?
        partial_policy_discrep_loss = partial(policy_discrep_loss,
                                              value_type=self.val_type,
                                              error_type=self.error_type,
                                              alpha=self.alpha,
                                              flip_count_prob=self.flip_count_prob)
        self.policy_discrep_objective_func = jit(partial_policy_discrep_loss)

        mem_loss_fn = mem_discrep_loss
        partial_kwargs = {
            'value_type': self.val_type,
            'error_type': self.error_type,
            'lambda_0': self.lambda_0,
            'lambda_1': self.lambda_1,
            'alpha': self.alpha,
            'flip_count_prob': self.flip_count_prob
        }
        if hasattr(self, 'objective'):
            if self.objective == 'bellman':
                mem_loss_fn = mem_bellman_loss
                partial_kwargs['residual'] = self.residual
            elif self.objective == 'tde':
                mem_loss_fn = mem_tde_loss
                partial_kwargs['residual'] = self.residual
            elif self.objective == 'obs_space':
                mem_loss_fn = obs_space_mem_discrep_loss

        partial_mem_discrep_loss = partial(mem_loss_fn, **partial_kwargs)
        self.memory_objective_func = jit(partial_mem_discrep_loss)

    @property
    def policy(self) -> jnp.ndarray:
        # return the learnt policy
        return softmax(self.pi_params, axis=-1)

    @property
    def memory(self) -> jnp.ndarray:
        return softmax(self.mem_params, axis=-1)

    def reset_pi_params(self, pi_shape: Sequence[int] = None):
        self.rand_key, pi_reset_key = random.split(self.rand_key)

        if pi_shape is None:
            pi_shape = self.pi_params.shape
        self.pi_params = glorot_init(pi_shape)
        self.pi_optim_state = self.pi_optim.init(self.pi_params)

    def new_pi_over_mem(self):
        if self.pi_params.shape[0] != self.og_n_obs:
            raise NotImplementedError(
                "Have not implemented adding bits to already existing memory.")

        add_n_mem_states = self.mem_params.shape[-1]
        self.pi_params = new_pi_over_mem(self.pi_params,
                                         add_n_mem_states=add_n_mem_states,
                                         new_mem_pi=self.new_mem_pi)

    @partial(jit, static_argnames=['self'])
    def policy_gradient_update(self, params: jnp.ndarray, optim_state: jnp.ndarray, pomdp: POMDP):
        outs, params_grad = value_and_grad(self.pg_objective_func, has_aux=True)(params, pomdp)
        v_0, (td_v_vals, td_q_vals) = outs

        # We add a negative here to params_grad b/c we're trying to
        # maximize the PG objective (value of start state).
        params_grad = -params_grad
        updates, optimizer_state = self.pi_optim.update(params_grad, optim_state, params)
        params = optax.apply_updates(params, updates)
        return v_0, td_v_vals, td_q_vals, params, optimizer_state

    @partial(jit, static_argnames=['self', 'sign'])
    def policy_discrep_update(self,
                              params: jnp.ndarray,
                              optim_state: jnp.ndarray,
                              pomdp: POMDP,
                              sign: bool = True):
        outs, params_grad = value_and_grad(self.policy_discrep_objective_func,
                                           has_aux=True)(params, pomdp)
        loss, (mc_vals, td_vals) = outs

        # it's the flip of sign b/c the optimizer already applies the negative sign
        params_grad *= (1 - float(sign))

        updates, optimizer_state = self.pi_optim.update(params_grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return loss, mc_vals, td_vals, params, optimizer_state

    def policy_improvement(self, pomdp: POMDP):
        if self.policy_optim_alg in ['policy_grad', 'policy_mem_grad', 'policy_mem_grad_unrolled']:
            policy_params = self.pi_params
            if self.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
                policy_params = self.pi_aug_params
            v_0, prev_td_v_vals, prev_td_q_vals, new_pi_params, new_optim_state= \
                self.policy_gradient_update(policy_params, self.pi_optim_state, pomdp)
            output = {
                'v_0': v_0,
                'prev_td_q_vals': prev_td_q_vals,
                'prev_td_v_vals': prev_td_v_vals
            }
        elif self.policy_optim_alg == 'policy_iter':
            new_pi_params, prev_td_v_vals, prev_td_q_vals = self.policy_iteration_update(
                self.pi_params, pomdp, eps=self.epsilon)
            new_optim_state = self.pi_optim_state
            output = {'prev_td_q_vals': prev_td_q_vals, 'prev_td_v_vals': prev_td_v_vals}
        elif self.policy_optim_alg == 'discrep_max' or self.policy_optim_alg == 'discrep_min':
            loss, mc_vals, td_vals, new_pi_params, new_optim_state = self.policy_discrep_update(
                self.pi_params,
                self.pi_optim_state,
                pomdp,
                sign=(self.policy_optim_alg == 'discrep_max'))
            output = {'loss': loss, 'mc_vals': mc_vals, 'td_vals': td_vals}
        else:
            raise NotImplementedError

        if self.policy_optim_alg in ['policy_mem_grad', 'policy_mem_grad_unrolled']:
            self.pi_aug_params = new_pi_params
        else:
            self.pi_params = new_pi_params
        self.pi_optim_state = new_optim_state
        return output

    @partial(jit, static_argnames=['self'])
    def memory_update(self, params: jnp.ndarray, optim_state: jnp.ndarray, pi_params: jnp.ndarray,
                      pomdp: POMDP):
        pi = softmax(pi_params / self.pi_softmax_temp, axis=-1)
        loss, params_grad = value_and_grad(self.memory_objective_func, argnums=0)(params, pi,
                                                                                  pomdp)

        updates, optimizer_state = self.mem_optim.update(params_grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, optimizer_state

    def memory_improvement(self, pomdp: POMDP):
        assert self.mem_params is not None, 'I have no memory params'
        loss, new_mem_params, new_mem_optim_state = self.memory_update(self.mem_params, self.mem_optim_state,
                                                  self.pi_params, pomdp)
        self.mem_params = new_mem_params
        self.mem_optim_state = new_mem_optim_state
        return loss

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()

        # delete unpickleable jitted functions
        del state['pg_objective_func']
        del state['policy_iteration_update']
        del state['policy_discrep_objective_func']
        del state['memory_objective_func']
        del state['pi_optim']
        state['pi_params'] = np.array(state['pi_params'])

        if state['mem_params'] is not None:
            del state['mem_optim']
            state['mem_params'] = np.array(state['mem_params'])
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        # restore jitted functions
        self.pg_objective_func = jit(pg_objective_func)
        if self.policy_optim_alg == 'policy_mem_grad':
            self.pg_objective_func = jit(mem_pg_objective_func)
        elif self.policy_optim_alg == 'policy_mem_grad_unrolled':
            self.pg_objective_func = jit(unrolled_mem_pg_objective_func)
        self.policy_iteration_update = jit(policy_iteration_step, static_argnames=['eps'])

        if 'optim_str' not in state:
            state['optim_str'] = 'sgd'
        self.pi_optim = get_optimizer(state['optim_str'], state['pi_lr'])
        if hasattr(self, 'mem_params'):
            self.mi_optim = get_optimizer(state['optim_str'], state['mi_lr'])

        if not hasattr(self, 'val_type'):
            self.val_type = 'v'
            self.error_type = 'l2'
        if not hasattr(self, 'lambda_0'):
            self.lambda_0 = 0.
            self.lambda_1 = 1.

        self.init_and_jit_objectives()
