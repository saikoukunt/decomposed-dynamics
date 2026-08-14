from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad
from jaxopt.prox import prox_lasso

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
)
from decomposed_dynamics.dynamics_models.linear import LinearOperatorHyperparams
from decomposed_dynamics.utils import (
    operator_correlation,
    reweighted_l1_prox,
    spectral_normalize,
)


@dataclass(frozen=True)
class AffineOperatorHyperparams(LinearOperatorHyperparams):
    b_l1_coeff: float = 0.4


class DecomposedAffineDynamics(DecomposedDynamicsModel):
    F: Array
    b: Array

    def initialize_params(self, key: Array):
        key, subkey = jr.split(key)

        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        self.F = spectral_normalize(F)

        self.b = jr.normal(subkey, (self.num_operators, self.num_latents))

    @staticmethod
    def initialize_hyperparams(**kwargs) -> AffineOperatorHyperparams:
        return AffineOperatorHyperparams(**kwargs)

    @eqx.filter_jit
    def compute_operator_flows(self, x: Array) -> Array:
        offsets = x[..., None, :] - self.b
        return jnp.einsum("kij, ...kj -> ...ki", self.F, offsets) + self.b

    @eqx.filter_jit
    def regularize_operators(self, hyperparams: AffineOperatorHyperparams) -> Self:
        F, b = self.apply_prox(
            self.F,
            self.b,
            hyperparams.l1_coeff,
            hyperparams.l1_reweight_coeff,
            hyperparams.b_l1_coeff,
        )
        F = self.decorrelate_operators(F, hyperparams.decorr_coeff)
        updated_model = eqx.tree_at(lambda model: (model.F, model.b), self, (F, b))

        return updated_model

    @eqx.filter_jit
    def apply_prox(
        self,
        F: Array,
        b: Array,
        l1_coeff: float,
        l1_reweight_coeff: float,
        b_l1_coeff: float,
    ) -> tuple[Array, Array]:
        F = spectral_normalize(F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        b = prox_lasso(b, l1reg=b_l1_coeff)

        return F, b

    @eqx.filter_jit
    def decorrelate_operators(self, F: Array, operator_decorr_coeff: float) -> Array:
        decorr_gradient = grad(operator_correlation)(F)
        F = F - operator_decorr_coeff * decorr_gradient

        return F
