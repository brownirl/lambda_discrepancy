from pathlib import Path

import numpy as np

from lamb.utils.math import normalize

from .pomdp import POMDPFile, POMDP
from .jax_pomdp import POMDP as JAXPOMDP

from definitions import ROOT_DIR


def load_spec(name: str, **kwargs):
    """
    Loads a pre-defined POMDP specification, as well as policies.
    :param name:            The name of the function or .POMDP file defining the POMDP.
    :param memory_id:       ID of memory function to use.
    :param n_mem_states:    Number of memory states allowed.
    :param mem_leakiness:   for memory_id="f" - how leaky do is out leaky identity function.

    The following **kwargs are specified for the following specs:
    tmaze_hyperparams:
        :param corridor_length:     Length of the maze corridor.
        :param discount:            Discount factor gamma to use.
        :param junction_up_pi:      If we specify a policy for the tmaze spec, what is the probability
                                    of traversing UP at the tmaze junction?
    """

    # Try to load from examples_lib first
    # then from pomdp_files
    try:
        pomdp_dir = Path(ROOT_DIR) / 'lamb' / 'envs' / 'classic' / 'POMDP'

        potential_pomdp_paths = [pomdp_dir / f'{name}.POMDP', pomdp_dir / f'{name}.pomdp']

        file_path = next(fpath for fpath in potential_pomdp_paths if fpath.exists())
        spec = POMDPFile(file_path).get_spec()
        if spec is None:
            raise ValueError(f"{name} POMDP doesn't exist in {pomdp_dir}.")
    except FileNotFoundError as _:
        raise FileNotFoundError(f"Couldn't find POMDP file {file_path}")

    # Check sizes and types
    if len(spec.keys()) < 6:
        raise ValueError("POMDP specification must contain at least: T, R, gamma, p0, phi, Pi_phi")
    if len(spec['T'].shape) != 3:
        raise ValueError("T tensor must be 3d")
    if len(spec['R'].shape) != 3:
        raise ValueError("R tensor must be 3d")

    if spec['pi_phi'] is not None:
        spec['pi_phi'] = np.array(spec['pi_phi']).astype('float')
        spec['pi_phi'] = normalize(spec['pi_phi'])
        if not np.all(len(spec['T']) == np.array([len(spec['R']), len(spec['pi_phi'][0][0])])):
            raise ValueError("T, R, and pi_phi must contain the same number of actions")

    # Make sure probs sum to 1
    # e.g. if they are [0.333, 0.333, 0.333], normalizing will do so
    spec['T'] = normalize(spec['T']) # terminal states had all zeros -> nan
    spec['p0'] = normalize(spec['p0'])
    spec['phi'] = normalize(spec['phi'])

    return spec


def load_pomdp(name: str, rand_key: np.random.RandomState = None, **kwargs) -> tuple[POMDP, dict]:
    """
    Wraps a MDP/POMDP specification in a POMDP
    """
    spec = load_spec(name, rand_key=rand_key, **kwargs)
    pomdp = POMDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], phi=spec['phi'], rand_key=rand_key)
    return pomdp, {'Pi_phi': spec['pi_phi']}


def load_jax_pomdp(name: str, fully_observable: bool = False):
    spec = load_spec(name)
    pomdp = JAXPOMDP(spec['T'], spec['R'], spec['p0'], spec['gamma'], spec['phi'],
                     fully_observable=fully_observable)
    return pomdp
