"""
This file runs a memory iteration with a batch of randomized initial policies,
as well as the TD optimal policy, on a list of different measures.

"""
import argparse
from functools import partial
from time import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, nn
from jax.config import config
from jax.debug import print
from jax_tqdm import scan_tqdm
import optax

from lamb.agents.analytical import new_pi_over_mem
from lamb.envs.classic import load_pomdp
from lamb.mdp import POMDP
from lamb.utils.lambda_discrep import log_all_measures, augment_and_log_all_measures
from lamb.memory import memory_cross_product
from lamb.utils.file_system import results_path, numpyify_and_save
from lamb.utils.math import reverse_softmax
from lamb.utils.loss import (
    pg_objective_func,
    discrep_loss,
    mem_tde_loss,
    mem_discrep_loss,
    mem_bellman_loss,
    obs_space_mem_discrep_loss
)
from lamb.utils.policy import get_unif_policies


def get_args():
    # Args
    parser = argparse.ArgumentParser()
    # yapf:disable

    # hyperparams for tmaze_hperparams
    parser.add_argument('--tmaze_corridor_length',
                        default=None,
                        type=int,
                        help='Length of corridor for tmaze_hyperparams')
    parser.add_argument('--tmaze_discount',
                        default=None,
                        type=float,
                        help='Discount rate for tmaze_hyperparams')
    parser.add_argument('--tmaze_junction_up_pi',
                        default=None,
                        type=float,
                        help='probability of traversing up at junction for tmaze_hyperparams')

    parser.add_argument('--spec', default='tmaze_5_two_thirds_up', type=str,
                        help='name of POMDP spec; evals Pi_phi policies by default')
    parser.add_argument('--mi_iterations', type=int, default=1,
                        help='For memory iteration, how many iterations of memory iterations do we do?')
    parser.add_argument('--mi_steps', type=int, default=20000,
                        help='For memory iteration, how many steps of memory improvement do we do per iteration?')
    parser.add_argument('--pi_steps', type=int, default=10000,
                        help='For memory iteration, how many steps of policy improvement do we do per iteration?')


    parser.add_argument('--policy_optim_alg', type=str, default='policy_iter',
                        help='policy improvement algorithm to use. "policy_iter" - policy iteration, "policy_grad" - policy gradient, '
                             '"discrep_max" - discrepancy maximization, "discrep_min" - discrepancy minimization')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='What optimizer do we use? (sgd | adam | rmsprop)')

    parser.add_argument('--random_policies', default=100, type=int,
                        help='How many random policies do we use for random kitchen sinks??')
    parser.add_argument('--leave_out_optimal', action='store_true',
                        help="Do we include the optimal policy when we select the initial policy")
    parser.add_argument('--n_mem_states', default=2, type=int,
                        help='for memory_id = 0, how many memory states do we have?')

    parser.add_argument('--lambda_0', default=0., type=float,
                        help='First lambda parameter for lambda-discrep')
    parser.add_argument('--lambda_1', default=1., type=float,
                        help='Second lambda parameter for lambda-discrep')

    parser.add_argument('--alpha', default=1., type=float,
                        help='Temperature parameter, for how uniform our lambda-discrep weighting is')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--value_type', default='q', type=str,
                        help='Do we use (v | q) for our discrepancies?')
    parser.add_argument('--error_type', default='l2', type=str,
                        help='Do we use (l2 | abs) for our discrepancies?')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='(POLICY ITERATION AND TMAZE_EPS_HYPERPARAMS ONLY) What epsilon do we use?')

    # CURRENTLY NOT USED
    parser.add_argument('--objectives', default=['discrep'])

    parser.add_argument('--study_name', default=None, type=str,
                        help='name of the experiment. Results saved to results/{experiment_name} directory if not None. Else, save to results directory directly.')
    parser.add_argument('--platform', default='cpu', type=str,
                        help='What platform do we run things on? (cpu | gpu)')
    parser.add_argument('--seed', default=2024, type=int,
                        help='What is the overall seed we use?')
    parser.add_argument('--n_seeds', default=1, type=int,
                        help='How many seeds do we run?')

    args = parser.parse_args()
    return args


sweep_hparams = {
    'alpha': 1,
    'lr': 0.01,
}


def get_kitchen_sink_policy(policies: jnp.ndarray, pomdp: POMDP, measure: Callable):
    batch_measures = jax.vmap(measure, in_axes=(0, None))
    all_policy_measures, _, _ = batch_measures(policies, pomdp)
    return policies[jnp.argmax(all_policy_measures)]


def make_experiment(args):

    # Get POMDP definition
    pomdp, pi_dict = load_pomdp(args.spec,
                                memory_id=0,
                                n_mem_states=args.n_mem_states,
                                corridor_length=args.tmaze_corridor_length,
                                discount=args.tmaze_discount,
                                junction_up_pi=args.tmaze_junction_up_pi)


    def experiment(rng: random.PRNGKey):
        info = {}

        batch_log_all_measures = jax.vmap(log_all_measures, in_axes=(None, 0))

        rng, mem_rng = random.split(rng)


        beginning_info = {}
        rng, pi_rng = random.split(rng)
        pi_shape = (pomdp.observation_space.n, pomdp.action_space.n)
        pi_paramses = reverse_softmax(get_unif_policies(pi_rng, pi_shape, args.random_policies + 1))
        updateable_pi_params = pi_paramses[-1]

        beginning_info['pi_params'] = pi_paramses.copy()
        beginning_info['measures'] = batch_log_all_measures(pomdp, pi_paramses)
        info['beginning'] = beginning_info

        # mem_aug_pi_paramses =
        # beginning_info['all_init_mem_measures'] = jax.vmap(augment_and_log_all_measures, in_axes=(0, None, 0))(mem_params, pomdp, mem_aug_pi_paramses)

        optim = get_optimizer(args.optimizer, args.lr)

        pi_tx_params = optim.init(updateable_pi_params)

        print("Running initial policy improvement")
        @scan_tqdm(args.pi_steps)
        def update_pg_step(inps, i):
            params, tx_params, pomdp = inps
            outs, params_grad = value_and_grad(pg_objective_func, has_aux=True)(params, pomdp)
            v_0, (td_v_vals, td_q_vals) = outs

            # We add a negative here to params_grad b/c we're trying to
            # maximize the PG objective (value of start state).
            params_grad = -params_grad
            updates, tx_params = optim.update(params_grad, tx_params, params)
            params = optax.apply_updates(params, updates)
            outs = (params, tx_params, pomdp)
            return outs, {'v0': v_0, 'v': td_v_vals, 'q': td_q_vals}

        output_pi_tuple, init_pi_optim_info = jax.lax.scan(update_pg_step,
                                                           (updateable_pi_params, pi_tx_params, pomdp),
                                                           jnp.arange(args.pi_steps),
                                                           length=args.pi_steps)

        memoryless_optimal_pi_params, _, _ = output_pi_tuple

        after_pi_op_info = {}
        after_pi_op_info['initial_improvement_pi_params'] = memoryless_optimal_pi_params
        after_pi_op_info['initial_improvement_measures'] = log_all_measures(pomdp, memoryless_optimal_pi_params)
        print("Learnt initial improvement policy:\n{}", nn.softmax(memoryless_optimal_pi_params, axis=-1))

        pis_with_memoryless_optimal = pi_paramses.at[-1].set(memoryless_optimal_pi_params)

        after_pi_op_info['all_tested_pi_params'] = pis_with_memoryless_optimal
        info['after_pi_op'] = after_pi_op_info

        if args.leave_out_optimal:
            pis_with_memoryless_optimal = pi_paramses[:-1]

        # now we get our kitchen sink policies
        kitchen_sinks_info = {}
        ld_pi_params = get_kitchen_sink_policy(pis_with_memoryless_optimal, pomdp, discrep_loss)
        pis_to_learn_mem = jnp.stack([ld_pi_params])

        kitchen_sinks_info['ld'] = ld_pi_params.copy()

        # We initialize 3 mem params: 1 for LD
        mem_shape = (pis_to_learn_mem.shape[0], pomdp.action_space.n, pomdp.observation_space.n, args.n_mem_states, args.n_mem_states)
        mem_params = random.normal(mem_rng, shape=mem_shape) * 0.5

        mem_tx_params = jax.vmap(optim.init, in_axes=0)(mem_params)

        info['beginning']['init_mem_params'] = mem_params.copy()
        info['after_kitchen_sinks'] = kitchen_sinks_info

        # Set up for batch memory iteration
        def update_mem_step(mem_params: jnp.ndarray,
                            pi_params: jnp.ndarray,
                            mem_tx_params: jnp.ndarray,
                            objective: str = 'discrep',
                            residual: bool = False):
            partial_kwargs = {
                'value_type': args.value_type,
                'error_type': args.error_type,
                'lambda_0': args.lambda_0,
                'lambda_1': args.lambda_1,
                'alpha': args.alpha,
            }
            mem_loss_fn = mem_discrep_loss
            if objective == 'bellman':
                mem_loss_fn = mem_bellman_loss
                partial_kwargs['residual'] = residual
            elif objective == 'tde':
                mem_loss_fn = mem_tde_loss
                partial_kwargs['residual'] = residual
            elif objective == 'obs_space':
                mem_loss_fn = obs_space_mem_discrep_loss

            pi = jax.nn.softmax(pi_params, axis=-1)
            loss, params_grad = value_and_grad(mem_loss_fn, argnums=0)(mem_params, pi, pomdp)

            updates, mem_tx_params = optim.update(params_grad, mem_tx_params, mem_params)
            new_mem_params = optax.apply_updates(mem_params, updates)

            return new_mem_params, pi_params, mem_tx_params, loss

        # Make our vmapped memory function
        update_ld_step = jax.vmap(partial(update_mem_step, objective='discrep', residual=True), in_axes=0)

        def scan_wrapper(inps, i, f: Callable):
            mem_params, pi_params, mem_tx_params, = inps
            new_mem_params, pi_params, mem_tx_params, loss = f(mem_params, pi_params, mem_tx_params)
            return (new_mem_params, pi_params, mem_tx_params), loss

        scan_tqdm_dec = scan_tqdm(args.mi_steps)
        update_ld_step = scan_tqdm_dec(partial(scan_wrapper, f=update_ld_step))

        mem_aug_pi_paramses = jax.vmap(new_pi_over_mem, in_axes=(0, None))(pis_to_learn_mem, args.n_mem_states)
        batch_mem_log_all_measures = jax.vmap(augment_and_log_all_measures, in_axes=(0, None, 0))
        mem_input_tuple = (mem_params, mem_aug_pi_paramses, mem_tx_params)

        # Memory iteration for all of our measures
        print("Starting {} iterations of λ-discrepancy minimization", args.mi_steps)
        after_mem_op_info = {}
        ld_mem_out, losses = jax.lax.scan(update_ld_step, mem_input_tuple, jnp.arange(args.mi_steps), length=args.mi_steps)
        ld_mem_paramses, ld_pi_paramses, _ = ld_mem_out
        ld_mem_info = {'mems': ld_mem_paramses,
                       'measures': batch_mem_log_all_measures(ld_mem_paramses, pomdp, ld_pi_paramses)}
        after_mem_op_info['ld'] = ld_mem_info

        info['after_mem_op'] = after_mem_op_info

        def cross_and_improve_pi(mem_params: jnp.ndarray,
                                 pi_params: jnp.ndarray,
                                 pi_tx_params: dict):
            mem_pomdp = memory_cross_product(mem_params, pomdp)

            output_pi_tuple, pi_optim_info = jax.lax.scan(update_pg_step,
                                                               (pi_params, pi_tx_params, mem_pomdp),
                                                               jnp.arange(args.pi_steps),
                                                               length=args.pi_steps)
            return output_pi_tuple, pi_optim_info

        # Get our parameters ready for batch policy improvement
        all_mem_paramses = ld_mem_paramses

        # now we do policy improvement over the learnt memory
        # reset pi indices, and mem_augment
        rng, pi_rng = random.split(rng)
        new_pi_paramses = reverse_softmax(get_unif_policies(pi_rng, pi_shape, mem_params.shape[0]))
        mem_aug_pi_paramses = new_pi_paramses.repeat(mem_params.shape[-1], axis=1)

        # Use the same initial random pi params across all final policy improvements.
        all_mem_aug_pi_params = mem_aug_pi_paramses
        all_mem_pi_tx_paramses = jax.vmap(optim.init, in_axes=0)(all_mem_aug_pi_params)

        # Batch policy improvement with PG
        all_improved_pi_tuple, all_improved_pi_info = jax.vmap(cross_and_improve_pi, in_axes=0)(all_mem_paramses,
                                                                                                all_mem_aug_pi_params,
                                                                                                all_mem_pi_tx_paramses)

        # Retrieve each set of learned pi params
        all_improved_pi_params, _, _ = all_improved_pi_tuple
        n = 1
        ld_improved_pi_params = all_improved_pi_params[:n]

        final_info = {
            'ld': {'pi_params': ld_improved_pi_params,
                   'measures': batch_mem_log_all_measures(ld_mem_paramses, pomdp, ld_improved_pi_params)},
        }

        info['final'] = final_info

        return info

    return experiment


if __name__ == "__main__":
    start_time = time()
    # jax.disable_jit(True)

    args = get_args()

    np.set_printoptions(precision=4, suppress=True)
    config.update('jax_platform_name', args.platform)
    config.update("jax_enable_x64", True)

    rng = random.PRNGKey(seed=args.seed)
    rngs = random.split(rng, args.n_seeds + 1)
    rng, exp_rngs = rngs[-1], rngs[:-1]

    t0 = time()
    experiment_vjit_fn = jax.jit(jax.vmap(make_experiment(args)))

    # Run the experiment!
    # results will be batched over (n_seeds, random_policies + 1).
    # The + 1 is for the TD optimal policy.
    outs = jax.block_until_ready(experiment_vjit_fn(exp_rngs))

    time_finish = time()

    results_path = results_path(args, entry_point='batch_run')
    info = {'logs': outs, 'args': args.__dict__}

    end_time = time()
    run_stats = {'start_time': start_time, 'end_time': end_time}
    info['run_stats'] = run_stats

    print(f"Saving results to {results_path}")
    numpyify_and_save(results_path, info)
