import functools
from abc import ABC, abstractmethod
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from jaxopt import ProximalGradient

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel
from decomposed_dynamics.observation_models.base import ObservationModel
from decomposed_dynamics.proximal_operators import _reweight_prox_hyperparams


class InferenceHyperparams(eqx.Module):
    max_iter: int = 1000
    tol: float = 1e-4
    dynamics_loss_coeff: Array = eqx.field(default=0.0, converter=jnp.array)


def _prox_coeffs_only(prox: Callable, num_latents: int) -> Callable:
    """Apply `prox` to the operator coefficients of a `[latents, coeffs]` state only."""

    def wrapped(x: Array, hyperparams_prox: Array, scaling: float = 1.0) -> Array:
        latents, coeffs = x[..., :num_latents], x[..., num_latents:]
        return jnp.concatenate(
            [latents, prox(coeffs, hyperparams_prox, scaling)], axis=-1
        )

    return wrapped


def solve_reweighted(
    solver: ProximalGradient,
    init: Array,
    prox_hyperparams: Array,
    prox_reweight_coeff: Array,
    reweight: Callable = _reweight_prox_hyperparams,
    **loss_kwargs,
) -> Array:
    solution, _ = solver.run(init, hyperparams_prox=prox_hyperparams, **loss_kwargs)
    solution, _ = solver.run(
        solution,
        hyperparams_prox=reweight(solution, prox_hyperparams, prox_reweight_coeff),
        **loss_kwargs,
    )

    return jnp.where(jnp.any(jnp.isnan(solution)), jnp.zeros_like(solution), solution)


class NoObsInferenceBackend(ABC):
    @staticmethod
    def initialize_hyperparams(**kwargs) -> InferenceHyperparams:
        return InferenceHyperparams(**kwargs)

    @eqx.filter_jit
    def infer_batch(
        self,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: InferenceHyperparams,
        compute_per_operator_predictions: Callable,
    ) -> Array:

        infer_trial = functools.partial(
            self.infer_trial,
            dynamics_model,
            compute_per_operator_predictions,
            hyperparams=hyperparams,
        )
        return vmap(infer_trial)(states, targets)

    @abstractmethod
    def infer_trial(
        self,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        states: Array,
        targets: Array,
        hyperparams: InferenceHyperparams,
    ) -> Array:
        raise NotImplementedError


class InferenceBackend(ABC):
    @staticmethod
    def initialize_hyperparams(**kwargs) -> InferenceHyperparams:
        return InferenceHyperparams(**kwargs)

    @eqx.filter_jit
    def infer_batch(
        self,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        observations: Array,
        hyperparams: InferenceHyperparams,
        compute_per_operator_predictions: Callable,
    ) -> tuple[Array, Array]:

        infer_trial = functools.partial(
            self.infer_trial,
            observation_model,
            dynamics_model,
            compute_per_operator_predictions,
            hyperparams=hyperparams,
        )
        state = vmap(infer_trial)(observations)

        return (
            state[..., : dynamics_model.state_dim],
            state[..., dynamics_model.state_dim :],
        )

    @abstractmethod
    def infer_trial(
        self,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        observations: Array,
        hyperparams: InferenceHyperparams,
    ) -> Array:
        raise NotImplementedError
