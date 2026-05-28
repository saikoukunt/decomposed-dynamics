import functools
from dataclasses import dataclass, replace

import jax.numpy as jnp
import jax.random as jr
from jax import Array, jit
from jax.tree_util import register_dataclass

from decomposed_dynamics.operators.base import OperatorGroup, OperatorParams
from decomposed_dynamics.utils import reweighted_l1_prox, spectral_normalize


@functools.partial(register_dataclass, data_fields=["F"])
@dataclass(frozen=True)
class LinearOperatorParams(OperatorParams):
    F: Array


@dataclass(frozen=True)
class LinearOperatorGroup(OperatorGroup):
    l1_coeff: float
    l1_reweight_coeff: float = 200

    def initialize_params(
        self, key: Array, num_operators: int, input_dim: int, output_dim: int
    ) -> LinearOperatorParams:
        F = jr.normal(key, (num_operators, output_dim, input_dim))
        F = spectral_normalize(F)

        return LinearOperatorParams(F)

    @jit
    def compute_operator_flows(
        self, operators: LinearOperatorParams, x: Array
    ) -> Array:
        return jnp.einsum("kij, ...j -> ...ki", operators.F, x)

    @jit
    def apply_prox(self, operators: LinearOperatorParams) -> LinearOperatorParams:
        F = spectral_normalize(operators.F)
        F = reweighted_l1_prox(operators.F, self.l1_coeff, self.l1_reweight_coeff)

        return replace(operators, F=F)
