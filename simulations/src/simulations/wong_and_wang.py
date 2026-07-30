from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, jit

from .rnn import RNN


@jit
def wong_and_wang_activation(
    x: Array, a: float = 270, b: float = 108, d: float = 0.154
):
    return (a * x - b) / (1 - jnp.exp(-d * (a * x - b)))


class wong_and_wang_RNN(RNN):
    tau_s: float
    gamma: float

    @override
    def __init__(
        self,
        W_exc: float = 0.2609,
        W_inhib: float = 0.0497,
        W_a_ext: float = 5.2e-4,
        dt: float = 0.1,
        gamma: float = 0.641,
        tau_s: float = 0.1,
    ):
        W_rec = np.zeros((2, 2))
        W_rec[0, 0] = W_exc
        W_rec[1, 1] = W_exc
        W_rec[0, 1] = -W_inhib
        W_rec[1, 0] = -W_inhib

        W_in = np.zeros((2, 3))
        W_in[:, 0] = 1
        W_in[0, 1] = W_a_ext
        W_in[1, 2] = W_a_ext

        super().__init__(
            2, 3, dt, jnp.array(W_rec), jnp.array(W_in), wong_and_wang_activation
        )

        self.tau_s = tau_s
        self.gamma = gamma

    @override
    @eqx.filter_jit
    def compute_input_activations(
        self, coherence: float, mu_0: float = 30, I_0: float = 0.3255
    ) -> Array:
        inputs = jnp.zeros((3))
        inputs = inputs.at[0].set(I_0)
        inputs = inputs.at[1].set(mu_0 * (1 + coherence / 100))
        inputs = inputs.at[2].set(mu_0 * (1 - coherence / 100))

        return inputs

    @override
    @eqx.filter_jit
    def compute_xdot_from_inputs(self, x: Array, input: Array) -> Array:
        rec_current = jnp.einsum("ij, ...j -> ...i", self.W_rec, x)
        input_current = jnp.einsum("im, ...m -> ...i", self.W_in, input)
        current = rec_current + input_current

        return -x / self.tau_s + (1 - x) * self.gamma * self.activation_fn(current)
