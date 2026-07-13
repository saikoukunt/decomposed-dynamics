from dataclasses import dataclass, replace
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import Array, jit
from jax.nn import softplus
from jax.tree_util import register_dataclass

from decomposed_dynamics.observation_models.base import ObservationParams, RateModel


@register_dataclass
@dataclass(frozen=True)
class LinearRateParams(ObservationParams):
    D: Array


@partial(
    register_dataclass, data_fields=[], meta_fields=["num_observations", "num_latents"]
)
class LinearRateModel(RateModel):
    def initialize_params(self, key: Array) -> LinearRateParams:
        D = jr.normal(key, (self.num_observations, self.num_latents))

        return LinearRateParams(D)

    @jit
    def predict_rates(self, params: LinearRateParams, x: Array) -> Array:
        return jnp.einsum("nd, ...d -> ...n", params.D, x)

    @jit
    def apply_prox(self, params: LinearRateParams) -> LinearRateParams:
        D = params.D / jnp.linalg.norm(params.D, axis=1, keepdims=True)

        return replace(params, D=D)


@partial(
    register_dataclass, data_fields=[], meta_fields=["num_observations", "num_latents"]
)
class SoftplusLinearRateModel(LinearRateModel):
    @jit
    def predict_rates(self, params: LinearRateParams, x: Array) -> Array:
        return softplus(super().predict_rates(x))
