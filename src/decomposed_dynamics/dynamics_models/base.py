from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax import Array


def linear_combination(c: Array, preds: Array) -> Array:
    return jnp.einsum("...k, ...ki -> ...i", c, preds)


class OperatorHyperparams(ABC):
    pass


class DecomposedDynamicsModel(eqx.Module):
    num_operators: int
    state_dim: int

    def __init__(self, num_operators: int, state_dim: int, key: Array, **init_kwargs):
        self.num_operators = num_operators
        self.state_dim = state_dim
        self.initialize_params(key, **init_kwargs)

    @abstractmethod
    def initialize_params(self, key: Array, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def initialize_hyperparams(**kwargs) -> OperatorHyperparams:
        raise NotImplementedError

    @abstractmethod
    def compute_operator_predictions(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def regularize_operators(
        self, latents: Array, hyperparams: OperatorHyperparams, **kwargs
    ) -> Self:
        raise NotImplementedError

    @eqx.filter_jit
    def combine_operator_predictions(self, x: Array, c: Array, preds: Array) -> Array:
        return linear_combination(c, preds)

    def compute_operator_flows(self, x: Array) -> Array:
        return self.compute_operator_predictions(x) - x[..., None, :]


class DeltaDynamics(DecomposedDynamicsModel):
    """Operators predict velocities, stepped forward with an Euler step of size `dt`."""

    dt: float = eqx.field(static=True)

    def __init__(self, *args, dt: float, **kwargs):
        self.dt = dt
        super().__init__(*args, **kwargs)

    @eqx.filter_jit
    def combine_operator_predictions(self, x: Array, c: Array, preds: Array) -> Array:
        return x + self.dt * linear_combination(c, preds)

    def compute_operator_flows(self, x: Array) -> Array:
        return self.compute_operator_predictions(x)
