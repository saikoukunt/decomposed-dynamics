import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.nn import softplus

from decomposed_dynamics.observation_models.base import RateModel


class LinearRateModel(RateModel):
    D: Array

    def initialize_params(self, key: Array):
        self.D = jr.normal(key, (self.num_observations, self.num_latents))

    @eqx.filter_jit
    def predict_rates(self, x: Array) -> Array:
        return jnp.einsum("nd, ...d -> ...n", self.D, x)

    @eqx.filter_jit
    def apply_prox(self) -> RateModel:
        D = self.D / jnp.linalg.norm(self.D, axis=1, keepdims=True)
        updated_model = eqx.tree_at(lambda model: model.D, self, D)

        return updated_model


class SoftplusLinearRateModel(LinearRateModel):
    @eqx.filter_jit
    def predict_rates(self, x: Array) -> Array:
        return softplus(super().predict_rates(x))
