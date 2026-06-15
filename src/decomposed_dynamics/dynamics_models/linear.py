from dataclasses import dataclass, replace
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad, jit
from jax.tree_util import register_dataclass

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorParams,
)
from decomposed_dynamics.utils import (
    operator_correlation,
    reweighted_l1_prox,
    spectral_normalize,
)


@register_dataclass
@dataclass(frozen=True)
class LinearOperatorParams(OperatorParams):
    F: Array


@partial(
    register_dataclass, data_fields=[], meta_fields=["num_latents", "num_operators"]
)
class DecomposedLinearDynamics(DecomposedDynamicsModel):
    def initialize_params(self, key: Array) -> LinearOperatorParams:
        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        F = spectral_normalize(F)

        return LinearOperatorParams(F)

    @jit
    def compute_operator_flows(
        self, operators: LinearOperatorParams, x: Array
    ) -> Array:
        return jnp.einsum("kij, ...j -> ...ki", operators.F, x)

    @jit
    def apply_prox(
        self,
        operators: LinearOperatorParams,
        l1_coeff: float,
        l1_reweight_coeff: float = 200,
    ) -> LinearOperatorParams:
        F = spectral_normalize(operators.F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        return replace(operators, F=F)

    @jit
    def decorrelate_operators(
        self,
        operators: LinearOperatorParams,
        operator_decorr_coeff: float,
    ) -> LinearOperatorParams:
        decorr_gradient = grad(operator_correlation)(operators.F)
        F = operators.F - operator_decorr_coeff * decorr_gradient

        return replace(operators, F=F)
