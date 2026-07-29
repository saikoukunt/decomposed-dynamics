from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, jit
from jax.nn import tanh


@jit
def pos_tanh(input: Array):
    return 0.5 + 0.5 * tanh(input)


class RNN(eqx.Module):
    state_dim: int
    input_dim: int
    dt: float
    W_rec: Array
    W_in: Array
    activation_fn: Callable

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        dt: float = 0.05,
        W_rec: Array | None = None,
        W_in: Array | None = None,
        activation_fn: Callable = pos_tanh,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt
        self.W_rec = jnp.zeros((state_dim, state_dim)) if W_rec is None else W_rec
        self.W_in = jnp.zeros((state_dim, input_dim)) if W_in is None else W_in
        self.activation_fn = activation_fn

    @eqx.filter_jit
    def compute_xdot_from_inputs(self, x: Array, input: Array) -> Array:
        rec_current = jnp.einsum("ij, ...j -> ...i", self.W_rec, x)
        input_current = jnp.einsum("im, ...m -> ...i", self.W_in, input)
        xdot = -x + self.activation_fn(rec_current + input_current)

        return xdot

    def compute_input_activations(self, inputs: Array) -> Array:
        """Subclasses should override this function with their own kwargs"""
        return inputs

    def compute_xdot(self, x: Array, **input_kwargs) -> Array:
        inputs = self.compute_input_activations(**input_kwargs)

        return self.compute_xdot_from_inputs(x, inputs)

    @eqx.filter_jit
    def euler_step(
        self,
        x: Array,
        sigma: float,
        key: Array,
        **input_kwargs,
    ) -> Array:
        flow = self.dt * self.compute_xdot(x, **input_kwargs)
        noise = sigma * jnp.sqrt(self.dt) * jr.normal(key, x.shape)

        return x + flow + noise

    @eqx.filter_jit
    def euler_trajectory(
        self, x_0: Array, sigma: float, num_iter: int, seed: int, **input_kwargs
    ) -> Array:

        key = jr.key(seed)
        x = jnp.zeros((num_iter, self.state_dim))
        x[0, :] = x_0

        for i in range(num_iter):
            key, subkey = jr.split(key)
            x[i, :] = self.euler_step(x[i - 1, :], sigma, subkey, **input_kwargs)

        return x
