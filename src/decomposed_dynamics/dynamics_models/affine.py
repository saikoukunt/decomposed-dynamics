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
class AffineOperatorParams(OperatorParams):
    F: Array
    b: Array


@partial(
    register_dataclass, data_fields=[], meta_fields=["num_latents", "num_operators"]
)
class DecomposedAffineDynamics(DecomposedDynamicsModel):
    def initialize_params(self, key: Array) -> AffineOperatorParams:
        key, subkey = jr.split(key)

        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        F = spectral_normalize(F)

        b = jr.normal(subkey, (self.num_operators, self.num_latents))

        return AffineOperatorParams(F, b)

    @jit
    def compute_operator_flows(
        self, operators: AffineOperatorParams, x: Array
    ) -> Array:
        offsets = x - operators.b
        return jnp.einsum("kij, ...j -> ...ki", operators.F, offsets) + operators.b

    @jit
    def apply_prox(
        self,
        operators: AffineOperatorParams,
        l1_coeff: float,
        l1_reweight_coeff: float = 200,
    ) -> AffineOperatorParams:
        F = spectral_normalize(operators.F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        return replace(operators, F=F)

    @jit
    def decorrelate_operators(
        self, operators: AffineOperatorParams, operator_decorr_coeff: float
    ) -> AffineOperatorParams:
        decorr_gradient = grad(operator_correlation)(operators.F)
        F = operators.F - operator_decorr_coeff * decorr_gradient

        return replace(operators, F=F)
