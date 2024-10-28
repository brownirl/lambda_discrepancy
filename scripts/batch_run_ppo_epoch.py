from collections import deque
from dataclasses import replace
from functools import partial
import inspect
from typing import Literal

from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from tap import Tap

from lamb.agents.ppo import Transition, env_step
from lamb.envs import get_gymnax_env
from lamb.envs.jax_wrappers import LogEnvState
from lamb.models import get_network_fn, ScannedRNN
from lamb.utils.file_system import get_results_path


class BatchPPOHyperparams(Tap):
    env: str = 'tmaze_5'
    num_envs: int = 4
    default_max_steps_in_episode: int = 1000
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_steps: int = 128
    num_epochs: int = 50
    update_epochs: int = 4
    num_minibatches: int = 4

    memoryless: bool = False
    double_critic: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    alpha: list[float] = [1.]  # adv = alpha * adv_lambda_0 + (1 - alpha) * adv_lambda_1
    ld_weight: list[float] = [0.0]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.
    vf_coeff: list[float] = [0.5]

    hidden_size: int = 128
    total_steps: int = int(1.5e6)
    entropy_coeff: float = 0.01
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'batch_ppo_test'

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.lambda1 = jnp.array(self.lambda1)
        self.alpha = jnp.array(self.alpha)
        self.ld_weight = jnp.array(self.ld_weight)

def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(args: BatchPPOHyperparams, rand_key: jax.random.PRNGKey):
    num_updates = (
        args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_gymnax_env(args.env, env_key,
                                     gamma=args.gamma,
                                     action_concat=args.action_concat)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    double_critic = args.double_critic
    memoryless = args.memoryless

    network_fn, action_size = get_network_fn(env, env_params, memoryless=memoryless)

    network = network_fn(action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size)

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    _env_step = partial(env_step, network=network, env=env, env_params=env_params)

    def train(vf_coeff, ld_weight, alpha, lambda1, lambda0, lr, rng):
        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac


        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, args.num_envs, *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, args.num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
        network_params = network.init(_rng, init_hstate, init_x)
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        # We first need to populate our LogEnvState stats.
        rng, _rng = jax.random.split(rng)
        init_rng = jax.random.split(_rng, args.num_envs)
        init_obsv, init_env_state = env.reset(init_rng, env_params)
        init_init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        init_runner_state = (
            train_state,
            env_state,
            init_obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_init_hstate,
            _rng,
        )

        starting_runner_state, _ = jax.lax.scan(
            _env_step, init_runner_state, None, env_params.max_steps_in_episode
        )

        def recursive_replace(env_state, new_env_state, names):
            if not isinstance(env_state, LogEnvState):
                return replace(env_state, env_state=recursive_replace(env_state.env_state, new_env_state.env_state, names))
            new_log_vals = {name: getattr(new_env_state, name) for name in names}
            return replace(env_state, **new_log_vals)

        replace_field_names = ['returned_episode_returns', 'returned_discounted_episode_returns', 'returned_episode_lengths']
        env_state = recursive_replace(env_state, starting_runner_state[1], replace_field_names)

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done, gae_lambda):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done, gae_lambda = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + args.gamma * next_value * (1 - next_done) - value
                    gae = delta + args.gamma * gae_lambda * (1 - next_done) * gae
                    return (gae, value, done, gae_lambda), gae
                _, advantages = jax.lax.scan(_get_advantages,
                                             (jnp.zeros_like(last_val), last_val, last_done, gae_lambda),
                                             traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            gae_lambda = jnp.array(lambda0)
            if double_critic:
                # last_val is index 1 here b/c we squeezed earlier.
                _calculate_gae = jax.vmap(_calculate_gae,
                                          in_axes=[transition_axes_map, 1, None, 0],
                                          out_axes=2)
                gae_lambda = jnp.array([lambda0, lambda1])
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-args.clip_eps, args.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        # Lambda discrepancy loss
                        if double_critic:
                            value_loss = ld_weight * (jnp.square(value[..., 0] - value[..., 1])).mean() + \
                                         (1 - ld_weight) * value_loss

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)

                        # which advantage do we use to update our policy?
                        if double_critic:
                            gae = (alpha * gae[..., 0] +
                                   (1 - alpha) * gae[..., 1])
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - args.clip_eps,
                                1.0 + args.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + vf_coeff * value_loss
                            - args.entropy_coeff * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], args.num_minibatches, -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            train_state = update_state[0]

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            rng = update_state[-1]
            if args.debug:

                def callback(info):
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
                    if len(timesteps) > 0:
                        print(
                            f"timesteps={timesteps[0]} - {timesteps[-1]}, avg episodic return={avg_return_values:.2f}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

            return runner_state, metric

        def _epoch(runner_state, _):
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, jnp.arange(round(num_updates / args.num_epochs)), round(num_updates / args.num_epochs)
            )
            # save metrics only every update_log_freq
            metric = jax.tree.map(update_filter, metric)

            res = {'metric': metric}
            if args.save_checkpoints:
                res['checkpoint'] = runner_state[0].params

            return runner_state, res

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            init_hstate,
            _rng,
        )

        runner_state, metric = jax.lax.scan(
            _epoch, runner_state, jnp.arange(args.num_epochs), args.num_epochs
        )
        # combine epochs with time steps
        metric['metric'] = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), metric['metric'])

        # returned metric has an extra dimension.
        # runner_state, metric = jax.lax.scan(
        #     _update_step, runner_state, jnp.arange(num_updates), num_updates
        # )
        #
        # # save metrics only every update_log_freq
        # metric = jax.tree.map(update_filter, metric)

        # TODO: offline eval here.
        final_train_state = runner_state[0]

        reset_rng = jax.random.split(_rng, args.num_eval_envs)
        eval_obsv, eval_env_state = env.reset(reset_rng, env_params)

        eval_init_hstate = ScannedRNN.initialize_carry(args.num_eval_envs, args.hidden_size)

        eval_runner_state = (
            final_train_state,
            eval_env_state,
            eval_obsv,
            jnp.zeros((args.num_eval_envs), dtype=bool),
            eval_init_hstate,
            _rng,
        )

        # COLLECT EVAL TRAJECTORIES
        eval_runner_state, eval_traj_batch = jax.lax.scan(
            _env_step, eval_runner_state, None, env_params.max_steps_in_episode
        )

        res = {"runner_state": runner_state, "metric": metric['metric'], 'final_eval_metric': eval_traj_batch.info}

        if args.save_checkpoints:
            res['checkpoint'] = metric['checkpoint']
        return res

    return train


if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = BatchPPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)

    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # we need to go backwards, since JAX returns indices
    # in the order in which they're vmapped.
    for i, arg in reversed(list(enumerate(train_args))):
        dims = [None] * len(train_args)
        dims[i] = 0
        vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
        if arg == 'rng':
            swept_args.appendleft(rngs)
        else:
            assert hasattr(args, arg)
            swept_args.appendleft(getattr(args, arg))

    train_jit = jax.jit(vmaps_train)
    out = train_jit(*swept_args)

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.
    final_eval = out['final_eval_metric']

    # the +1 at the end is to include the done step
    largest_episode = final_eval['returned_episode'].argmax(axis=-2).max() + 1

    def get_first_n_filter(x):
        return x[..., :largest_episode, :]
    out['final_eval_metric'] = jax.tree.map(get_first_n_filter, final_eval)

    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict()
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    print("Done.")
