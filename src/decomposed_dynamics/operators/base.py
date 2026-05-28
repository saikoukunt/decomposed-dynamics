from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, jit


class OperatorParams(ABC):
    pass


class OperatorGroup(ABC):
    @abstractmethod
    def initialize_operators(
        self, key: Array, num_operators: int, input_dim: int, output_dim: int, **kwargs
    ) -> OperatorParams:
        raise NotImplementedError

    @abstractmethod
    def compute_operator_flows(self, operators: OperatorParams, x: Array) -> Array:
        raise NotImplementedError

    @jit
    def compute_weighted_flow(self, coeffs: Array, flows: Array) -> Array:
        return jnp.einsum("...k, ...ki -> i")

    def apply_prox(self, operators: OperatorParams) -> OperatorParams:
        return operators
