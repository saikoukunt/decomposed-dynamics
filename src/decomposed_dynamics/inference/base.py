import functools
from abc import ABC, abstractmethod
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from jaxopt import ProximalGradient

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel
from decomposed_dynamics.observation_models.base import ObservationModel
from decomposed_dynamics.proximal_operators import ProximalOperator


class InferenceHyperparams(eqx.Module):
    max_iter: int = 1000
    tol: float = 1e-4
    dynamics_loss_coeff: Array = eqx.field(default=1.0, converter=jnp.array)

    @classmethod
    def get_backend(
        cls, observation_model: ObservationModel | None = None
    ) -> "type[InferenceBackend | NoObsInferenceBackend]":
        raise NotImplementedError


def _prox_coeffs_only(num_latents: int) -> Callable:
    """Apply the prox to the operator coefficients of a `[latents, coeffs]` state only."""

    def wrapped(x: Array, prox: ProximalOperator, scaling: float = 1.0) -> Array:
        latents, coeffs = x[..., :num_latents], x[..., num_latents:]
        return jnp.concatenate([latents, prox(coeffs, scaling)], axis=-1)

    return wrapped


def solve_reweighted(
    solver: ProximalGradient,
    init: Array,
    prox: ProximalOperator,
    num_latents: int = 0,
    **loss_kwargs,
) -> Array:
    solution, _ = solver.run(init, hyperparams_prox=prox, **loss_kwargs)
    solution, _ = solver.run(
        solution,
        hyperparams_prox=prox.reweight(solution[..., num_latents:]),
        **loss_kwargs,
    )

    return jnp.where(jnp.any(jnp.isnan(solution)), jnp.zeros_like(solution), solution)


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
