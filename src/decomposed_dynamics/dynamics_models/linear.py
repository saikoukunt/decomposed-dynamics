from dataclasses import dataclass, replace
from functools import partial
from typing import override

import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad, jit
from jax.tree_util import register_dataclass

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorHyperparams,
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


@dataclass(frozen=True)
class LinearOperatorHyperparams(OperatorHyperparams):
    l1_coeff: float = 0.03
    l1_reweight_coeff: float = 200
    decorr_coeff: float = 0.01


@partial(
    register_dataclass, data_fields=[], meta_fields=["num_latents", "num_operators"]
)
class DecomposedLinearDynamics(DecomposedDynamicsModel):
    def initialize_params(self, key: Array) -> LinearOperatorParams:
        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        F = spectral_normalize(F)

        return LinearOperatorParams(F)

    def initialize_hyperparams(self, **kwargs) -> LinearOperatorHyperparams:
        return LinearOperatorHyperparams(**kwargs)

    @jit
    def compute_operator_flows(
        self, operators: LinearOperatorParams, x: Array
    ) -> Array:
        return jnp.einsum("kij, ...j -> ...ki", operators.F, x)

    @jit(static_argnames=["hyperparams"])
    def regularize_operators(
        self, operators: LinearOperatorParams, hyperparams: LinearOperatorHyperparams
    ):
        F = self.apply_prox(
            operators.F, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff
        )
        F = self.decorrelate_operators(F, hyperparams.decorr_coeff)

        return replace(operators, F=F)

    @jit
    def apply_prox(
        self,
        F: Array,
        l1_coeff: float,
        l1_reweight_coeff: float = 200,
    ) -> LinearOperatorParams:
        F = spectral_normalize(F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        return F

    @jit
    def decorrelate_operators(
        self,
        F: Array,
        operator_decorr_coeff: float,
    ) -> LinearOperatorParams:
        decorr_gradient = grad(operator_correlation)(F)
        F = F - operator_decorr_coeff * decorr_gradient

        return F
