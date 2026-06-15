from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, jit


class OperatorParams(ABC):
    pass


class DecomposedDynamicsModel(ABC):
    def __init__(self, num_operators, num_latents):
        self.num_operators = num_operators
        self.num_latents = num_latents

    @abstractmethod
    def initialize_params(self, key: Array, **kwargs) -> OperatorParams:
        raise NotImplementedError

    @abstractmethod
    def compute_operator_flows(self, operators: OperatorParams, x: Array) -> Array:
        raise NotImplementedError

    @jit
    def predict_next_state(self, c: Array, flows: Array) -> Array:
        return jnp.einsum("...k, ...ki -> ...i", c, flows)

    def apply_prox(self, operators: OperatorParams, **kwargs) -> OperatorParams:
        return operators

    @abstractmethod
    def decorrelate_operators(
        self, operators: OperatorParams, **kwargs
    ) -> OperatorParams:
        pass
