from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorHyperparams,
)
from decomposed_dynamics.utils import (
    operator_correlation,
    reweighted_l1_prox,
    spectral_normalize,
)


@dataclass(frozen=True)
class LinearOperatorHyperparams(OperatorHyperparams):
    l1_coeff: float = 0.03
    l1_reweight_coeff: float = 200
    decorr_coeff: float = 0.01


class DecomposedLinearDynamics(DecomposedDynamicsModel):
    F: Array

    def initialize_params(self, key: Array):
        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        self.F = spectral_normalize(F)

    @staticmethod
    def initialize_hyperparams(**kwargs) -> LinearOperatorHyperparams:
        return LinearOperatorHyperparams(**kwargs)

    @eqx.filter_jit
    def compute_operator_flows(self, x: Array) -> Array:
        return jnp.einsum("kij, ...j -> ...ki", self.F, x)

    @eqx.filter_jit
    def regularize_operators(self, hyperparams: LinearOperatorHyperparams) -> Self:
        F = self.apply_prox(self.F, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff)
        F = self.decorrelate_operators(F, hyperparams.decorr_coeff)
        updated_model = eqx.tree_at(lambda model: model.F, self, F)

        return updated_model

    @eqx.filter_jit
    def apply_prox(
        self,
        F: Array,
        l1_coeff: float,
        l1_reweight_coeff: float,
    ) -> Array:
        F = spectral_normalize(F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        return F

    @eqx.filter_jit
    def decorrelate_operators(self, F: Array, operator_decorr_coeff: float) -> Array:
        decorr_gradient = grad(operator_correlation)(F)
        F = F - operator_decorr_coeff * decorr_gradient

        return F
