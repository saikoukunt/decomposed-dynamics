from math import floor
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array

from .rnn import RNN, pos_tanh


class bistable_RNN(RNN):
    def __init__(
        self,
        W_exc: float,
        W_inhib: float,
        W_input: float = 0.01,
        dt: float = 0.05,
        activation_fn: Callable = pos_tanh,
    ):
        W_rec = np.zeros((2, 2))
        W_rec[0, 0] = W_exc
        W_rec[1, 1] = W_exc
        W_rec[0, 1] = -W_inhib
        W_rec[1, 0] = -W_inhib

        W_in = np.zeros((2, 2))
        W_in[0, 0] = W_input
        W_in[1, 0] = -W_input
        W_in[:, 1] = 1

        super().__init__(2, 2, dt, jnp.array(W_rec), jnp.array(W_in), activation_fn)

    def compute_input_activations(self, coherence: float, mu_0: float) -> Array:
        inputs = jnp.zeros((2))
        inputs = inputs.at[0].set(coherence)
        inputs = inputs.at[1].set(mu_0)

        return inputs
