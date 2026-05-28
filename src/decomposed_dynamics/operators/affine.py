import functools
from dataclasses import dataclass, replace

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.tree_util import register_dataclass

from decomposed_dynamics.operators.base import OperatorGroup, OperatorParams
from decomposed_dynamics.utils import reweighted_l1_prox, spectral_normalize


@functools.partial(register_dataclass, data_fields=["F", "b"])
@dataclass(frozen=True)
class AffineOperatorParams(OperatorParams):
    F: Array
    b: Array


@dataclass(frozen=True)
class AffineOperatorGroup(OperatorGroup):
    l1_coeff: float
    l1_reweight_coeff: float = 200

    def initialize_params(
        self, key: Array, num_operators: int, input_dim: int, output_dim: int
    ) -> AffineOperatorParams:
        key, subkey = jr.split(key)

        F = jr.normal(key, (num_operators, output_dim, input_dim))
        F = spectral_normalize(F)

        b = jr.normal(subkey, (num_operators, output_dim))

        return AffineOperatorParams(F, b)

    def compute_operator_flows(
        self, operators: AffineOperatorParams, x: Array
    ) -> Array:
        return jnp.einsum("kij, ...j -> ...ki", operators.F, x) + operators.b

    def apply_prox(self, operators: AffineOperatorParams) -> AffineOperatorParams:
        F = spectral_normalize(operators.F)
        F = reweighted_l1_prox(operators.F, self.l1_coeff, self.l1_reweight_coeff)

        return replace(operators, F=F)
