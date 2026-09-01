from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit


class OperatorHyperparams(ABC):
    pass


class DecomposedDynamicsModel(eqx.Module):
    num_operators: int
    num_latents: int

    def __init__(self, num_operators: int, num_latents: int, key: Array, **init_kwargs):
        self.num_operators = num_operators
        self.num_latents = num_latents
        self.initialize_params(key, **init_kwargs)

    @abstractmethod
    def initialize_params(self, key: Array, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def initialize_hyperparams(**kwargs) -> OperatorHyperparams:
        raise NotImplementedError

    @abstractmethod
    def compute_operator_flows(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def regularize_operators(self, hyperparams: OperatorHyperparams, **kwargs) -> Self:
        raise NotImplementedError

    @eqx.filter_jit
    def predict_next_state(self, x: Array, c: Array, flows: Array) -> Array:
        return jnp.einsum("...k, ...ki -> ...i", c, flows)


class DeltaDynamics(eqx.Module):
    dt: float

    def __init__(self, dt):
        self.dt = dt

    @jit
    def predict_next_state(self, x: Array, c: Array, flows: Array) -> Array:
        return x + self.dt * jnp.einsum("...k, ...ki -> ...i", c, flows)
