from functools import partial
from pathlib import Path
from typing import Union, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
from flax.training import orbax_utils
import orbax.checkpoint

from lamb.envs.pocman import State
from lamb.models import ScannedRNN
from lamb.utils.file_system import load_train_state, make_hash_md5


class CollectHyperparams(Tap):
    rnn_path_0: Union[str, Path]
    rnn_path_1: Union[str, Path]
    behavior_path: Union[str, Path]

    update_idx_to_take: int = None

    num_envs: int = 4
    n_samples: int = int(1e6)

    seed: int = 2024
    platform: str = 'cpu'

    def configure(self) -> None:
        self.add_argument('--rnn_path_0', type=Path)
        self.add_argument('--rnn_path_1', type=Path)
        self.add_argument('--behavior_path', type=Path)


def ppo_pocman_step(runner_state, unused,
                    behavior_network, rnn_network_0, rnn_network_1,
                    env, env_params):
    def get_pocman_state(s) -> State:
        if isinstance(s, State):
            return s
        if hasattr(s, 'env_state'):
            return get_pocman_state(s.env_state)
        else:
            raise TypeError('No Pocman env_state found.')

    (behavior_ts, rnn_ts_0, rnn_ts_1, env_state, last_obs, last_done,
        behavior_hstate, rnn_hstate_0, rnn_hstate_1, rng) = runner_state
    rng, _rng = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    next_behavior_hstate, pi, value = behavior_network.apply(behavior_ts.params, behavior_hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    value, action, log_prob = (
        value.squeeze(0),
        action.squeeze(0),
        log_prob.squeeze(0),
    )

    # get our RNN hidden states that we're sampling
    next_rnn_hstate_0, _, _ = rnn_network_0.apply(rnn_ts_0.params, rnn_hstate_0, ac_in)
    next_rnn_hstate_1, _, _ = rnn_network_1.apply(rnn_ts_1.params, rnn_hstate_1, ac_in)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, next_behavior_hstate.shape[0])
    obsv, next_env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

    # transition = Transition(
    #     last_done, action, value, reward, log_prob, last_obs, info
    # )
    pocman_state = get_pocman_state(env_state)
    possible_locations = jnp.array(env._unwrapped.generator.reachable_spaces)

    def get_single_occupancy(loc: jnp.ndarray):
        return jnp.all(possible_locations == loc[None, ...], axis=-1)

    # We vmap twice, once for the batch dimension in VecEnv,
    # the second time for the 4 ghosts
    ghost_occupancy = jax.vmap(jax.vmap(get_single_occupancy))(pocman_state.ghost_locations).sum(axis=-2)
    datum = {
        'x_0': rnn_hstate_0,
        'x_1': rnn_hstate_1,
        'pellet_occupancy': jnp.all(pocman_state.pellet_locations != 0, axis=-1),
        'ghost_occupancy': jnp.clip(ghost_occupancy, a_max=1),
        # 'state': pocman_state
    }
    runner_state = (behavior_ts, rnn_ts_0, rnn_ts_1, next_env_state, obsv, done,
                    next_behavior_hstate, next_rnn_hstate_0, next_rnn_hstate_1, rng)
    return runner_state, datum


def make_collect(args: CollectHyperparams, key: chex.PRNGKey):
    steps_to_collect = args.n_samples // args.num_envs

    behavior_key, rnn_key_0, rnn_key_1, key = jax.random.split(key, 4)

    env, env_params, behavior_args, behavior_network, behavior_ts = load_train_state(behavior_key, args.behavior_path,
                                                                                     update_idx_to_take=args.update_idx_to_take,
                                                                                     best_over_rng=True)
    _, _, rnn_args_0, rnn_network_0, rnn_ts_0 = load_train_state(rnn_key_0, args.rnn_path_0,
                                                                 update_idx_to_take=args.update_idx_to_take,
                                                                 best_over_rng=True)
    _, _, rnn_args_1, rnn_network_1, rnn_ts_1 = load_train_state(rnn_key_1, args.rnn_path_1,
                                                                 update_idx_to_take=args.update_idx_to_take,
                                                                 best_over_rng=True)

    _env_step = partial(ppo_pocman_step, behavior_network=behavior_network,
                        rnn_network_0=rnn_network_0, rnn_network_1=rnn_network_1,
                        env=env, env_params=env_params)
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    ckpts = {
        'behavior': {'args': behavior_args, 'ts': behavior_ts, 'path': args.behavior_path},
        'rnn_0': {'args': rnn_args_0, 'ts': rnn_ts_0, 'path': args.rnn_path_0},
        'rnn_1': {'args': rnn_args_1, 'ts': rnn_ts_1, 'path': args.rnn_path_1}
    }

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        # init hidden state
        init_behavior_hstate = ScannedRNN.initialize_carry(args.num_envs, behavior_args['hidden_size'])
        init_rnn_hstate_0 = ScannedRNN.initialize_carry(args.num_envs, rnn_args_0['hidden_size'])
        init_rnn_hstate_1 = ScannedRNN.initialize_carry(args.num_envs, rnn_args_1['hidden_size'])
        init_runner_state = (
            behavior_ts,
            rnn_ts_0,
            rnn_ts_1,
            env_state,
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_behavior_hstate,
            init_rnn_hstate_0,
            init_rnn_hstate_1,
            _rng,
        )

        runner_state, dataset = jax.lax.scan(
            _env_step, init_runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        # Now we flatten back down
        flat_dataset = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)

        return flat_dataset

    return collect, ckpts


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = CollectHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    make_key, collect_key, key = jax.random.split(key, 3)

    collect_fn, ckpt_info = make_collect(args, make_key)
    collect_fn = jax.jit(collect_fn)

    dataset = collect_fn(collect_key)

    def path_to_str(d: dict):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                path_to_str(v)


    to_save = {
        'dataset': dataset,
        'args': args.as_dict(),
        'ckpt': ckpt_info,
    }
    path_to_str(to_save)

    save_path = args.behavior_path.parent / \
                f'buffer_{args.n_samples}_timestep_{args.update_idx_to_take}_seed_{args.seed}_{make_hash_md5(args.as_dict())}'

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(to_save)

    print(f"Saving results to {save_path}")
    orbax_checkpointer.save(save_path, to_save, save_args=save_args)

    print("Done.")

