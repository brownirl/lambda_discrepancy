from typing import Literal

from jax import numpy as jnp
from tap import Tap

class BatchHyperparams(Tap):
    env: str = 'CartPole-P-v0'
    # env: str = 'Hopper-P-v0'
    # env: str = 'MemoryChain-bsuite'
    default_max_steps_in_episode: int = 1000
    memoryless: bool = False
    double_critic: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    alpha: list[float] = [1.]  # adv = alpha * adv_lambda_0 + (1 - alpha) * adv_lambda_1
    ld_weight: list[float] = [0.5]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.

    hidden_size: int = 128
    total_steps: int = int(1.5e6)
    entropy_coeff: float = 0.01

    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'batch_ppo_test'

    def process_args(self) -> None:
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.lambda1 = jnp.array(self.lambda1)
        self.alpha = jnp.array(self.alpha)
        self.ld_weight = jnp.array(self.ld_weight)
