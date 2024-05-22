from functools import partial
import json
from typing import Tuple
from pathlib import Path

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random, lax

from lamb.utils.math import euclidian_dist

from definitions import ROOT_DIR


def half_dist_prob(dist: float, max_dist: float, lb: float = 0.5):
    prob = (1 + jnp.power(2.0, -dist / max_dist)) * lb
    return prob

@chex.dataclass
class RockSampleState:
    position: chex.Array
    rock_morality: chex.Array


class RockSample(Environment):
    direction_mapping = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)

    def __init__(self,
                 key: chex.PRNGKey,
                 config_path: Path = Path(ROOT_DIR, 'porl', 'envs',
                                          'configs', 'rocksample_7_8_config.json')):
        """
        RockSample environment in Gymnax.
        Observations: position (2) and rock goodness/badness
        Actions: k + 5 actions. Actions are as follows:
        0: North, 1: East, 2: South, 3: West,
        4: Sample,
        5: Check rock 1, ..., k + 5: Check rock k
        :param config_file: config file for RockSample (located in unc/envs/configs)
        :param seed: random seed for the environment.
        :param rock_obs_init: What do we initialize our rock observations with?
        """
        self.config_path = config_path
        with open(self.config_path) as f:
            config = json.load(f)

        self.size = config['size']
        self.k = config['rocks']

        self.rock_positions = self.generate_map(self.size, self.k, key)

        self.half_efficiency_distance = config['half_efficiency_distance']
        self.bad_rock_reward = config['bad_rock_reward']
        self.good_rock_reward = config['good_rock_reward']
        self.exit_reward = config['exit_reward']

    def observation_space(self, env_params: EnvParams):
        # return gymnax.environments.spaces.Box(0, 1, (2 * self.size + 2,))
        return gymnax.environments.spaces.Box(-1, 1, (2 * self.size + self.k,))

    def action_space(self, env_params: EnvParams):
        return gymnax.environments.spaces.Discrete(self.k + 5)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    @staticmethod
    def generate_map(size: int, k: int,
                     rand_key: random.PRNGKey) -> jnp.ndarray:
        rows_range = jnp.arange(size)
        cols_range = rows_range[:-1]
        possible_rock_positions = jnp.dstack(jnp.meshgrid(rows_range, cols_range)).reshape(-1, 2)
        all_positions_idx = random.choice(rand_key,
                                          possible_rock_positions.shape[0], (k, ),
                                          replace=False)
        all_positions = possible_rock_positions[all_positions_idx]
        return all_positions

    @staticmethod
    @partial(jax.jit, static_argnums=(0, -1))
    def sample_positions(size: int, rand_key: random.PRNGKey, n: int = 1):
        rows_range = jnp.arange(size)
        cols_range = rows_range[:-1]
        possible_positions = jnp.dstack(jnp.meshgrid(rows_range, cols_range)).reshape(-1, 2)
        sample_idx = random.choice(rand_key, (size - 1) * size, (n, ), replace=True)
        sample_position = jnp.array(possible_positions[sample_idx], dtype=int)
        return sample_position

    @partial(jax.jit, static_argnums=(0,))
    def sample_morality(self, rand_key: random.PRNGKey) -> jnp.ndarray:
        k = self.rock_positions.shape[0]
        return random.bernoulli(rand_key, shape=(k, )).astype(int)

    def get_obs(self, state: RockSampleState, action: int, key: chex.PRNGKey) -> jnp.ndarray:
        """
        Observation is dependent on action here.
        Observation is a size ** 2 + k vector:
        first two features are the position.
        last k features are the current_rock_obs.
        :param state:
        :param action:
        :return:
        """
        position_obs_y = jnp.zeros(self.size)
        position_obs_x = jnp.zeros(self.size)
        position = state.position.astype(int)
        position_obs_y = position_obs_y.at[position[0]].set(1)
        position_obs_x = position_obs_x.at[position[1]].set(1)

        # CHECK
        check_inputs = (state, action, key)

        current_rocks_obs = jax.lax.cond(action > 4,
                                         self._check_transition,
                                         lambda x: jnp.zeros(self.k, dtype=int),
                                         check_inputs)

        return jnp.concatenate([position_obs_y, position_obs_x, current_rocks_obs])

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, RockSampleState]:

        morality_key, position_key = random.split(key)
        rock_morality = self.sample_morality(morality_key)
        agent_position = self.sample_positions(self.size, position_key, 1)[0]

        state = RockSampleState(position=agent_position,
                                rock_morality=rock_morality)

        # we use the 0 action here, since it's < 4
        return self.get_obs(state, 0, key), state

    def _check_transition(self, check_inps: tuple[RockSampleState, int, chex.PRNGKey]):
        state, action, rand_key = check_inps

        rock_idx = action - 5
        dist = euclidian_dist(state.position, self.rock_positions[rock_idx])
        prob = half_dist_prob(dist, self.half_efficiency_distance)

        # w.p. prob we return correct rock observation.
        rock_obs = state.rock_morality[rock_idx]
        choices = jnp.array([rock_obs, 1 - rock_obs])
        probs = jnp.array([prob, 1 - prob])
        rock_obs = random.choice(rand_key, choices, (1, ), p=probs)[0]

        obs = jnp.zeros(self.k, dtype=rock_obs.dtype).at[rock_idx].set((rock_obs == 0) * -1 + rock_obs)
        return obs

    def _move_transition(self, move_inps: tuple[RockSampleState, EnvParams, int]):
        state, params, action = move_inps

        new_pos = state.position + self.direction_mapping[action % self.direction_mapping.shape[0]]
        pos_max = jnp.array([self.size - 1, self.size - 1])
        new_position = jnp.maximum(jnp.minimum(new_pos, pos_max), 0)

        return new_position

    def _sample_transition(self, sample_inps: tuple[RockSampleState, EnvParams]):
        state, params = sample_inps

        ele = (self.rock_positions == state.position)
        bool_pos = jnp.all(ele, axis=-1).astype(int)

        zero_arr = jnp.zeros_like(state.rock_morality)
        new_rock_morality = jnp.maximum(state.rock_morality - bool_pos, zero_arr)

        all_rock_rews = self.good_rock_reward * state.rock_morality \
                        + self.bad_rock_reward * (1 - state.rock_morality)
        rew = jnp.dot(bool_pos, all_rock_rews)
        sample_outs = (new_rock_morality, rew)

        return sample_outs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: RockSampleState,
                 action: int,
                 params: EnvParams):

        # Note: we can pass in unmodified state to each branch b/c they never happen simultaneously.

        # SAMPLING
        sampling_inputs = (state, params)

        no_change_outs = (state.rock_morality, 0.)
        sample_outs = jax.lax.cond(action == 4,
                                   self._sample_transition,
                                   lambda x: no_change_outs,
                                   sampling_inputs)
        rock_morality, reward = sample_outs

        # MOVING
        move_inputs = (state, params, action)

        position = jax.lax.cond(action < 4,
                                self._move_transition,
                                lambda x: state.position,
                                move_inputs)

        terminal = position[1] == (self.size - 1)
        reward += terminal * self.exit_reward

        info = {
            # 'rock_morality': rock_morality,
            # 'agent_position': position,
            # 'rock_positions': self.rock_positions
        }
        next_state = RockSampleState(position=position,
                                     rock_morality=rock_morality)

        # check is dealt with in get_obs
        obs = self.get_obs(next_state, action, key)

        return lax.stop_gradient(obs), lax.stop_gradient(next_state), reward, terminal, info

    # def generate_array(self) -> jnp.ndarray:
    #     """
    #     Generate numpy array representing state.
    #     Mappings are as follows:
    #     0 = white space
    #     1 = agent
    #     2 = rock
    #     3 = agent + rock
    #     4 = goal
    #     :return:
    #     """
    #     viz_array = np.zeros((self.size, self.size))
    #
    #     viz_array[self.rock_positions[:, 0], self.rock_positions[:, 1]] = 2
    #
    #     viz_array[int(self.agent_position[0]), int(self.agent_position[1])] += 1
    #
    #     viz_array[:, self.size - 1] = 4
    #     return jnp.array(viz_array)


