import functools
from abc import ABC, abstractmethod
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, vmap

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel
from decomposed_dynamics.dynamics_models.regularization import (
    reweight_prox_hyperparams,
)
from decomposed_dynamics.observation_models.base import ObservationModel


class Penalty(eqx.Module):
    """Penalty coefficient; acts as its value `_lambda` in arithmetic and jnp functions."""

    _lambda: Array = eqx.field(converter=jnp.asarray)
    reweight: Array = eqx.field(default=0.0, converter=jnp.asarray)

    def reweighted(self, x: Array) -> Self:
        return Penalty(
            reweight_prox_hyperparams(x, self._lambda, self.reweight), self.reweight
        )

    def __jax_array__(self) -> Array:
        return self._lambda

    # python scalars don't go through __jax_array__
    def __mul__(self, other: Any) -> Array:
        return self._lambda * other

    def __rmul__(self, other: Any) -> Array:
        return other * self._lambda


def is_penalty(node: Any) -> bool:
    return isinstance(node, Penalty)


def interpolate_penalties[T](start: T, end: T, frac: float) -> T:
    """Interpolate the value of every `Penalty` in `start` toward `end`; other leaves stay at `start`."""
    return jax.tree.map(
        lambda a, b: (
            Penalty(a._lambda + frac * (b._lambda - a._lambda), a.reweight)
            if is_penalty(a)
            else a
        ),
        start,
        end,
        is_leaf=is_penalty,
    )


class InferenceHyperparams(eqx.Module):
    max_iter: int = 1000
    tol: float = 1e-4
    dynamics_loss_coeff: Array = eqx.field(default=1.0, converter=jnp.array)

    @classmethod
    def get_backend(
        cls, observation_model: ObservationModel | None = None
    ) -> "type[InferenceBackend | NoObsInferenceBackend]":
        raise NotImplementedError


class NoObsInferenceBackend(ABC):
    @staticmethod
    def initialize_hyperparams(**kwargs) -> InferenceHyperparams:
        return InferenceHyperparams(**kwargs)

    @classmethod
    @eqx.filter_jit
    def infer_batch(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: InferenceHyperparams,
    ) -> Array:

        infer_trial = functools.partial(
            cls.infer_trial,
            dynamics_model,
            hyperparams=hyperparams,
        )
        return vmap(infer_trial)(states, targets)

    @staticmethod
    @abstractmethod
    def infer_trial(
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: InferenceHyperparams,
    ) -> Array:
        raise NotImplementedError


class InferenceBackend(ABC):
    @staticmethod
    def initialize_hyperparams(**kwargs) -> InferenceHyperparams:
        return InferenceHyperparams(**kwargs)

    @classmethod
    @eqx.filter_jit
    def infer_batch(
        cls,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        observations: Array,
        hyperparams: InferenceHyperparams,
    ) -> tuple[Array, Array]:

        infer_trial = functools.partial(
            cls.infer_trial,
            observation_model,
            dynamics_model,
            hyperparams=hyperparams,
        )
        state = vmap(infer_trial)(observations)

        return (
            state[..., : dynamics_model.state_dim],
            state[..., 1:, dynamics_model.state_dim :],
        )

    @staticmethod
    @abstractmethod
    def infer_trial(
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        observations: Array,
        hyperparams: InferenceHyperparams,
    ) -> Array:
        raise NotImplementedError
