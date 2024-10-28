from functools import partial
from pathlib import Path
from typing import Union, Literal

from chex import dataclass
from jumanji.environments.routing.pac_man import State
from jumanji.environments.routing.pac_man.types import Position
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from tap import Tap

from porl.agents.ppo import env_step
from porl.envs.pocman import PocMan
from porl.models.actor_critic import ScannedRNN, PelletPredictorNN
from porl.utils.file_system import load_train_state, numpyify_and_save

from definitions import ROOT_DIR


class PocmanProbeCollectHyperparams(Tap):
    probe_path_0: Union[str, Path]
    probe_path_1: Union[str, Path]
    rnn_path_0: Union[str, Path]
    rnn_path_1: Union[str, Path]

    behavior_policy_idx: Literal[0, 1] = 1
    seed: int = 2024

    def configure(self) -> None:
        self.add_argument('--probe_path_0', type=Path)
        self.add_argument('--probe_path_1', type=Path)
        self.add_argument('--rnn_path_0', type=Path)
        self.add_argument('--rnn_path_1', type=Path)


def state_to_dict(state: State):
    state_dict = {}
    for k, v in state.items():
        if isinstance(v, Position):
            state_dict[k] = {'x': v.x, 'y': v.y}
        else:
            state_dict[k] = v
    return state_dict


def unpack_and_flatten_state(state) -> dict:
    while (not isinstance(state, State)):
        state = state.env_state

    flattened_state = jax.tree.map(lambda x: x[0], state)
    return state_to_dict(flattened_state)


def load_probe(fpath: Path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    unpacked_ts = restored['final_train_state']

    # TODO: refactor this
    n_pellet_predictions = unpacked_ts['params']['params']['Dense_3']['bias'].shape[0]

    network = PelletPredictorNN(hidden_size=args['hidden_size'],
                                n_outs=n_pellet_predictions,
                                n_hidden_layers=args['n_hidden_layers'])
    return network, unpacked_ts

def predictions_to_map(predictions: jnp.ndarray, env: PocMan):
    predictions = predictions.squeeze()
    env_generator = env._unwrapped.generator

    # we first subtract by 1, so that all walls are -1, and
    # empty spaces are 0.
    preds_map = env_generator.numpy_maze - 1

    preds_map = preds_map.at[env_generator.pellet_spaces[:, 1], env_generator.pellet_spaces[:, 0]].set(predictions)
    return preds_map


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = PocmanProbeCollectHyperparams().parse_args()

    key = jax.random.PRNGKey(args.seed)
    load_key_0, load_key_1, key = jax.random.split(key, 3)

    probe_network_0, probe_ts_0 = load_probe(args.probe_path_0)
    probe_network_1, probe_ts_1 = load_probe(args.probe_path_1)

    # TODO: This is kind of sketch. We've refactored this now, so change this when done retraining.
    env, env_params, rnn_args0, rnn_network0, rnn_ts0 = load_train_state(load_key_0, args.rnn_path_0,
                                                                         update_idx_to_take=2,
                                                                         best_over_rng=True)
    _, _, rnn_args1, rnn_network1, rnn_ts1 = load_train_state(load_key_1, args.rnn_path_1,
                                                              update_idx_to_take=2,
                                                              best_over_rng=True)

    predictions_to_map = jax.jit(partial(predictions_to_map, env=env))

    networks = [rnn_network0, rnn_network1]
    tses = [rnn_ts0, rnn_ts1]
    ts = tses[args.behavior_policy_idx]
    _env_step = jax.jit(partial(env_step, network=networks[args.behavior_policy_idx], env=env, env_params=env_params))

    @jax.jit
    def predict_probes(obs, done, hs0, hs1):
        ac_in = (obs[jnp.newaxis, :], done[jnp.newaxis, :])
        hs0, _, _ = rnn_network0.apply(rnn_ts0.params, hs0, ac_in)
        hs1, _, _ = rnn_network1.apply(rnn_ts1.params, hs1, ac_in)

        predictions0, _ = probe_network_0.apply(probe_ts_0['params'], hs0)
        predictions1, _ = probe_network_1.apply(probe_ts_1['params'], hs1)
        return (predictions0, predictions1), (hs0, hs1)

    key, reset_key = jax.random.split(key)
    reset_key = reset_key[None, ...]
    obsv, state = env.reset(reset_key, env_params)
    states = [unpack_and_flatten_state(state)]
    predictions, pred_maps = [], []

    assert rnn_args1['hidden_size'] == rnn_args0['hidden_size']
    hstate = ScannedRNN.initialize_carry(1, rnn_args1['hidden_size'])
    hs0 = ScannedRNN.initialize_carry(1, rnn_args0['hidden_size'])
    hs1 = ScannedRNN.initialize_carry(1, rnn_args1['hidden_size'])

    done = jnp.array([False])
    rs = (ts, state, obsv, done, hstate, key)
    while not jnp.any(done):
        preds, (hs0, hs1) = predict_probes(obsv, done, hs0, hs1)
        predictions.append(preds)
        pred_maps.append((predictions_to_map(preds[0]), predictions_to_map(preds[1])))
        rs, transition = _env_step(rs, None)
        ts, state, obsv, done, hstate, key = rs
        states.append(unpack_and_flatten_state(state))

    preds, (hs0, hs1) = predict_probes(obsv, done, hs0, hs1)
    pred_maps.append((predictions_to_map(preds[0]), predictions_to_map(preds[1])))

    res = {
        'states': states,
        'predictions': pred_maps
    }

    res_path = Path(ROOT_DIR, 'results', f'pocman_pellet_probe_trajectory_bidx_{args.behavior_policy_idx}.npy')

    print(f"Saving Pocman probe trajectory to {res_path}")
    numpyify_and_save(res_path, res)
    print("Done.")
