from dataclasses import dataclass, replace
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad, jit
from jax.tree_util import register_dataclass
from jaxopt.prox import prox_lasso

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorParams,
)
from decomposed_dynamics.dynamics_models.linear import LinearOperatorHyperparams
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


@dataclass(frozen=True)
class AffineOperatorHyperparams(LinearOperatorHyperparams):
    b_l1_coeff: float = 0.4


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

    def initialize_hyperparams(self, **kwargs) -> AffineOperatorHyperparams:
        return AffineOperatorHyperparams(**kwargs)

    @jit
    def compute_operator_flows(
        self, operators: AffineOperatorParams, x: Array
    ) -> Array:
        offsets = x[..., None, :] - operators.b
        return jnp.einsum("kij, ...kj -> ...ki", operators.F, offsets) + operators.b

    @jit(static_argnames=["hyperparams"])
    def regularize_operators(
        self, operators: AffineOperatorParams, hyperparams: AffineOperatorHyperparams
    ):
        F, b = self.apply_prox(
            operators.F,
            operators.b,
            hyperparams.l1_coeff,
            hyperparams.l1_reweight_coeff,
            hyperparams.b_l1_coeff,
        )
        F = self.decorrelate_operators(F, hyperparams.decorr_coeff)

        return replace(operators, F=F, b=b)

    @jit
    def apply_prox(
        self,
        F: Array,
        b: Array,
        l1_coeff: float,
        l1_reweight_coeff: float,
        b_l1_coeff: float,
    ) -> AffineOperatorParams:
        F = spectral_normalize(F)
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        b = prox_lasso(b, l1reg=b_l1_coeff)

        return F, b

    @jit
    def decorrelate_operators(
        self, F: Array, operator_decorr_coeff: float
    ) -> AffineOperatorParams:
        decorr_gradient = grad(operator_correlation)(F)
        F = F - operator_decorr_coeff * decorr_gradient

        return F
