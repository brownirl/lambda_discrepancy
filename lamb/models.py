import functools
import math

import chex
import distrax
from gymnax.environments import environment, spaces
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant

from lamb.envs.battleship import Battleship

class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

class SimpleNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out


class DiscreteActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)
        return pi

class Critic(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return critic


class DiscreteActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SimpleNN(hidden_size=self.hidden_size)(obs)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return _, pi, jnp.squeeze(v, axis=-1)


class DiscreteActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)

class ContinuousActor(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi

class ContinuousActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SimpleNN(hidden_size=self.hidden_size)(obs)

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return _, pi, jnp.squeeze(v, axis=-1)


class ContinuousActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class SquareDeconv(nn.Module):
    hidden_size: int
    final_size: int
    final_n_channels: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        # add dimensions here to
        x = x[..., None, None, :]
        # 5x5
        if self.final_size == 5:
            out1 = nn.ConvTranspose(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding='VALID')(x)
            out1 = nn.relu(out1)
            tconv_out = nn.ConvTranspose(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding='VALID')(out1)

        # 3x3
        elif self.final_size == 3:
            out1 = nn.ConvTranspose(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding='VALID')(x)
            out1 = nn.relu(out1)
            tconv_out = nn.ConvTranspose(features=self.final_n_channels, kernel_size=(2, 2), strides=1, padding='VALID')(out1)

        # 10x10
        elif self.final_size == 10:
            out1 = nn.ConvTranspose(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding='VALID')(x)
            out1 = nn.relu(out1)
            out2 = nn.ConvTranspose(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding='VALID')(out1)
            out2 = nn.relu(out2)
            tconv_out = nn.ConvTranspose(features=self.hidden_size, kernel_size=(5, 5), strides=1, padding='VALID')(out2)

        else:
            raise NotImplementedError
        return tconv_out


class SmallImageCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # 5x5
        if x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=1)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        # 3x3
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 3:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out1)

        # 10x10
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(5, 5), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out2)

        else:
            raise NotImplementedError

        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last three dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-3], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out


class ImageDiscreteActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return _, pi, jnp.squeeze(v, axis=-1)


class ImageDiscreteActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class BattleShipActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # Obs is a t x b x obs_size array.
        obs, dones = x
        hit = obs[..., 0:1]
        valid_action_mask = obs[..., 1:self.action_dim + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)
        grid_size = int(math.sqrt(self.action_dim))

        # # here we check if we've concatenated action
        # # if we do, use a CNN to embed the action.
        # if obs.shape[-1] > 1:
        #     flat_action = obs[..., 1:]
        #     chex.assert_axis_dimension(flat_action, -1, self.action_dim)
        #
        #     # Make our one-hot action image
        #     flat_action_idx = jnp.argmax(flat_action, axis=-1)
        #
        #     def idx_to_img(idx):
        #         # converts an idx into an image, with order='C'
        #         idx = jnp.unravel_index(idx, (grid_size, grid_size))
        #         img = jnp.zeros((grid_size, grid_size, 2))
        #         return img.at[idx].set(1)
        #
        #     vmap_first_two_dims_img = jax.vmap(jax.vmap(idx_to_img))
        #     # add last dimension for channels
        #     action = vmap_first_two_dims_img(flat_action_idx)
        #
        #     action_embedding = SmallImageCNN(self.hidden_size)(action)
        #     obs = jnp.concatenate((hit, action_embedding), axis=-1)

        embedding = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        embedding = jnp.concatenate((hit, embedding), axis=-1)
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        # MLP actor
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class BattleShipActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # Obs is a t x b x obs_size array.
        obs, dones = x
        hit = obs[..., 0:1]
        valid_action_mask = obs[..., 1:self.action_dim + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)
        grid_size = int(math.sqrt(self.action_dim))

        # # here we check if we've concatenated action
        # # if we do, use a CNN to embed the action.
        # if obs.shape[-1] > 1:
        #     flat_action = obs[..., 1:]
        #     chex.assert_axis_dimension(flat_action, -1, self.action_dim)
        #
        #     # Make our one-hot action image
        #     flat_action_idx = jnp.argmax(flat_action, axis=-1)
        #
        #     def idx_to_img(idx):
        #         # converts an idx into an image, with order='C'
        #         idx = jnp.unravel_index(idx, (grid_size, grid_size))
        #         img = jnp.zeros((grid_size, grid_size, 2))
        #         return img.at[idx].set(1)
        #
        #     vmap_first_two_dims_img = jax.vmap(jax.vmap(idx_to_img))
        #     # add last dimension for channels
        #     action = vmap_first_two_dims_img(flat_action_idx)
        #
        #     action_embedding = SmallImageCNN(self.hidden_size)(action)
        #     obs = jnp.concatenate((hit, action_embedding), axis=-1)

        embedding = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        embedding = jnp.concatenate((hit, embedding), axis=-1)
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        # MLP actor
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)

def get_network_fn(env: environment.Environment, env_params: environment.EnvParams,
                   memoryless: bool = False):
    if isinstance(env, Battleship) or (hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship)):
        network_fn = BattleShipActorCriticRNN
        action_size = env.action_space(env_params).n
        if memoryless:
            network_fn = BattleShipActorCritic
    elif isinstance(env.action_space(env_params), spaces.Discrete):
        action_size = env.action_space(env_params).n

        # Check whether we use image observations
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageDiscreteActorCriticRNN
            if memoryless:
                network_fn = ImageDiscreteActorCritic

        else:
            network_fn = DiscreteActorCriticRNN
            if memoryless:
                network_fn = DiscreteActorCritic
    elif isinstance(env.action_space(env_params), spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        network_fn = ContinuousActorCriticRNN
        if memoryless:
            network_fn = ContinuousActorCritic
    else:
        raise NotImplementedError
    return network_fn, action_size
