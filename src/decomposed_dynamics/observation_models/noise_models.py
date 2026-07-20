import equinox as eqx
import jax.numpy as jnp
from jax import Array
from optax import l2_loss

from decomposed_dynamics.observation_models.base import NoiseModel


class GaussianNoiseModel(NoiseModel):
    @eqx.filter_jit
    def neg_log_likelihood(self, rates: Array, observations: Array):
        return l2_loss(rates, observations).sum(axis=-1).mean()


class PoissonNoiseModel(NoiseModel):
    @eqx.filter_jit
    def neg_log_likelihood(
        self, rates: Array, observations: Array, epsilon: Array = 1e-8
    ):
        return rates - observations * jnp.log(rates + epsilon)
