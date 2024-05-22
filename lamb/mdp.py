import copy
import gymnasium as gym
from gymnasium import spaces
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from lamb.utils.data import one_hot

def normalize(M, axis=-1):
    M = M.astype(float)
    if M.ndim > 1:
        denoms = M.sum(axis=axis, keepdims=True)
    else:
        denoms = M.sum()
    M = np.divide(M, denoms.astype(float), out=np.zeros_like(M), where=(denoms != 0))
    return M

def is_stochastic(M):
    return jnp.allclose(M, normalize(M))

def random_sparse_mask(size, sparsity):
    n_rows, n_cols = size
    p = (1 - sparsity) # probability of 1
    q = (n_cols * p - 1) / (n_cols - 1) # get remaining probability after mandatory 1s
    if 0 < q <= 1:
        some_ones = np.random.choice([0, 1], size=(n_rows, n_cols - 1), p=[1 - q, q])
        mask = np.concatenate([np.ones((n_rows, 1)), some_ones], axis=1)
    else:
        mask = np.concatenate([np.ones((n_rows, 1)), np.zeros((n_rows, n_cols - 1))], axis=1)
    for row in mask:
        np.random.shuffle(row)
    return mask

def random_stochastic_matrix(size):
    alpha_size = size[-1]
    out_size = size[:-1] if len(size) > 1 else None
    return np.random.dirichlet(np.ones(alpha_size), out_size)

def random_reward_matrix(Rmin, Rmax, size):
    R = np.random.uniform(Rmin, Rmax, size)
    R = np.round(R, 2)
    return R

def random_observation_fn(n_states, n_obs_per_block):
    all_state_splits = [
        random_stochastic_matrix(size=(1, n_obs_per_block)) for _ in range(n_states)
    ]
    all_state_splits = jnp.stack(all_state_splits).squeeze()
    #e.g.[[p, 1-p],
    #     [q, 1-q],
    #     ...]

    obs_fn_mask = jnp.kron(jnp.eye(n_states), jnp.ones((1, n_obs_per_block)))
    #e.g.[[1, 1, 0, 0, 0, 0, ...],
    #     [0, 0, 1, 1, 0, 0, ...],
    #     ...]

    tiled_split_probs = jnp.kron(jnp.ones((1, n_states)), all_state_splits)
    #e.g.[[p, 1-p, p, 1-p, p, 1-p, ...],
    #     [q, 1-q, q, 1-q, q, 1-q, ...],
    #     ...]

    observation_fn = obs_fn_mask * tiled_split_probs
    return observation_fn

@register_pytree_node_class
class MDP(gym.Env):
    def __init__(self, T, R, p0, gamma=0.9, terminal_mask: np.ndarray = None,
                 rand_key: np.random.RandomState = None):
        self.gamma = gamma
        self.T = T
        self.R = R
        # if isinstance(T, np.ndarray):
        #     self.T = np.stack(T).copy().astype(float)
        #     self.R = np.stack(R).copy().astype(float)
        # else:
        #     self.T = jnp.stack(T).copy().astype(float)
        #     self.R = jnp.stack(R).copy().astype(float)

        self.R_min = np.min(self.R)
        self.R_max = np.max(self.R)
        self.p0 = p0
        self.current_state = None
        self.rand_key = rand_key

        # Terminal mask is a boolean mask across all states that indicates
        # whether the state is a terminal(/absorbing) state.
        if terminal_mask is not None:
           self.terminal_mask = terminal_mask
        else:
            self.terminal_mask = jnp.array([jnp.all(self.T[:, i, i] == 1.) for i in range(self.state_space.n)])

    def tree_flatten(self):
        children = (self.T, self.R, self.p0, self.gamma, self.terminal_mask)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        return repr(self.T) + '\n' + repr(self.R)

    def stationary_distribution(self, pi=None, p0=None, max_steps=200):
        if p0 is None:
            state_distr = np.ones(self.state_space.n) / self.state_space.n
        else:
            state_distr = p0
        old_distr = state_distr

        if pi is None:
            T_pi = np.mean(self.T, axis=0)
        else:
            T_pi = self.T[pi, np.arange(self.state_space.n), :]

        for t in range(max_steps):
            state_distr = state_distr @ T_pi
            if np.allclose(state_distr, old_distr):
                break
            old_distr = state_distr
        return state_distr

    def reset(self, state=None, **kwargs):
        if state is None:
            if self.rand_key is not None:
                state = self.rand_key.choice(self.state_space.n, p=self.p0)
            else:
                state = np.random.choice(self.state_space.n, p=self.p0)
        self.current_state = state
        info = {'state': self.current_state}
        return self.observe(self.current_state), info

    def step(self, action: int, gamma_terminal: bool = True):
        pr_next_s = self.T[action, self.current_state, :]
        if self.rand_key is not None:
            next_state = self.rand_key.choice(self.state_space.n, p=pr_next_s)
        else:
            next_state = np.random.choice(self.state_space.n, p=pr_next_s)
        reward = self.R[action][self.current_state][next_state]
        # Check if next_state is absorbing state
        is_absorbing = (self.T[:, next_state, next_state] == 1)
        terminal = is_absorbing.all() # absorbing for all actions
        truncated = False
        observation = self.observe(next_state)
        info = {'state': next_state}
        self.current_state = next_state
        # Conform to new-style Gym API
        return observation, reward, terminal, truncated, info

    def observe(self, s):
        return s

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Discrete(self.T.shape[-1])

    @property
    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.T.shape[-1])

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.T.shape[0])

    @classmethod
    def generate(cls, n_states, n_actions, sparsity=0, gamma=0.9, Rmin=-1, Rmax=1):
        T = [] # List of s -> s transition matrices, one for each action
        R = [] # List of s -> s reward matrices, one for each action
        for a in range(n_actions):
            T_a = random_stochastic_matrix(size=(n_states, n_states))
            R_a = random_reward_matrix(Rmin, Rmax, (n_states, n_states))
            if sparsity > 0:
                mask = random_sparse_mask((n_states, n_states), sparsity)
                T_a = normalize(T_a * mask)
                R_a = R_a * mask
            T.append(T_a)
            R.append(R_a)

        T = np.array(T)
        R = np.array(R)
        p0 = random_stochastic_matrix(size=[n_states])
        mdp = cls(T, R, p0, gamma)
        return mdp

class BlockMDP(MDP):
    def __init__(self, base_mdp, n_obs_per_block=2, obs_fn=None):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)

        if obs_fn is None:
            obs_fn = random_observation_fn(base_mdp.state_space.n, n_obs_per_block)
        else:
            n_obs_per_block = obs_fn.shape[1]

        obs_mask = (obs_fn > 0).astype(int)

        self.T = [] # List of x -> x transition matrices, one for each action
        self.R = [] # List of x -> x reward matrices, one for each action
        for a in range(self.action_space.n):
            Ta, Ra = base_mdp.T[a], base_mdp.R[a]
            Tx_a = obs_mask.transpose() @ Ta @ obs_fn
            Rx_a = obs_mask.transpose() @ Ra @ obs_mask
            self.T.append(Tx_a)
            self.R.append(Rx_a)
        self.T = jnp.stack(self.T)
        self.R = jnp.stack(self.R)
        self.obs_fn = obs_fn

@register_pytree_node_class
class POMDP(MDP):
    def __init__(self, base_mdp: MDP, phi):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.p0, base_mdp.gamma,
                         base_mdp.terminal_mask)
        self.base_mdp = copy.deepcopy(base_mdp)
        self.phi = phi # array: base_mdp.state_space.n, n_abstract_states

    def tree_flatten(self):
        children = (self.T, self.R, self.p0, self.gamma, self.terminal_mask,
                    self.phi)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mdp = MDP(*children[:-1])
        return cls(mdp, children[-1])

    def __repr__(self):
        base_str = super().__repr__()
        return base_str + '\n' + repr(self.phi)

    def observe(self, s):
        assert self.phi[s].sum() == 1
        if self.rand_key is not None:
            return self.rand_key.choice(self.observation_space.n, p=self.phi[s])

        return np.random.choice(self.observation_space.n, p=self.phi[s])

    @property
    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.phi.shape[-1])

    # def B(self, pi, t=200):
    #     p = self.base_mdp.stationary_distribution(pi=pi, p0=self.p0, max_steps=t)
    #     return normalize(p*self.phi.transpose())
    #
    # def compute_Tz(self, belief, Tx):
    #     return belief @ Tx @ self.phi
    #
    # def compute_Rz(self, belief, Rx, Tx, Tz):
    #     return jnp.divide( (belief@(Rx*Tx)@self.phi), Tz,
    #                      out=jnp.zeros_like(Tz), where=(Tz!=0) )

    def is_abstract_policy(self, pi):
        agg_states = (self.phi.sum(axis=0) > 1)
        for idx, is_agg in enumerate(agg_states):
            agg_cluster = (
                one_hot(idx, self.observation_space.n) @ self.phi.transpose()).astype(bool)
            if not np.all(pi[agg_cluster] == pi[agg_cluster][0]):
                return False
        return True

    def get_abstract_policy(self, pi):
        assert self.is_abstract_policy(pi)
        mask = self.phi.transpose()
        obs_fn = normalize(mask)
        return (pi @ obs_fn.transpose()).astype(int)

    def get_ground_policy(self, pi):
        return self.phi @ pi

    def generate_random_policies(self, n):
        policies = []
        for _ in range(n):
            policies.append(
                random_stochastic_matrix((self.observation_space.n, self.action_space.n)))

        return policies

    @classmethod
    def generate(cls, n_states, n_actions, n_obs, sparsity=0, gamma=0.9, Rmin=-1, Rmax=1):
        mdp = MDP.generate(n_states, n_actions, sparsity, gamma, Rmin, Rmax)
        phi = random_stochastic_matrix(size=(n_states, n_obs))
        return cls(mdp, phi)

class UniformPOMDP(POMDP):
    def __init__(self, base_mdp, phi, pi=None, p0=None):
        super().__init__(base_mdp, phi, pi, p0)

    def B(self, pi, t=200):
        p = self._replace_stationary_distribution(pi=pi, p0=self.p0, max_steps=t)
        return normalize(p * self.phi.transpose())

    def _replace_stationary_distribution(self, pi=None, p0=None, max_steps=200):
        return jnp.ones(self.base_mdp.state_space.n) / self.base_mdp.state_space.n

def test():
    # Generate a random base MDP
    mdp1 = MDP.generate(n_states=5, n_actions=3, sparsity=0.5)
    assert all([is_stochastic(mdp1.T[a]) for a in range(mdp1.action_space.n)])

    # Add block structure to the base MDP
    mdp2 = BlockMDP(mdp1, n_obs_per_block=2)
    assert all([np.allclose(mdp2.base_mdp.T[a], mdp1.T[a]) for a in range(mdp1.action_space.n)])
    assert all([np.allclose(mdp2.base_mdp.R[a], mdp1.R[a]) for a in range(mdp1.action_space.n)])

    # Construct abstract MDP of the block MDP using perfect abstraction function
    phi = (mdp2.obs_fn.transpose() > 0).astype(int)
    mdp3 = POMDP(mdp2, phi)
    assert np.allclose(mdp1.T, mdp3.T)
    assert np.allclose(mdp1.R, mdp3.R)
    print('All tests passed.')


if __name__ == '__main__':
    test()
