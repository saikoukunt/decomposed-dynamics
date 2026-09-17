import functools
from dataclasses import dataclass
from typing import Callable, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxopt import ProximalGradient

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import (
    InferenceHyperparams,
    NoObsInferenceBackend,
)
from decomposed_dynamics.loss_functions import normalized_dynamics_reconstruction_loss
from decomposed_dynamics.proximal_operators import prox_l1_unit_tv


class FusedLassoHyperparams(InferenceHyperparams):
    prox_hyperparams: Array = eqx.field(default=(0.25, 0.25), converter=jnp.array)


@dataclass(frozen=True)
class FusedLassoNoObsInference(NoObsInferenceBackend):
    prox: Callable = prox_l1_unit_tv

    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> FusedLassoHyperparams:
        return FusedLassoHyperparams(**kwargs)

    @override
    @eqx.filter_jit
    def infer_trial(
        self,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        states: Array,
        targets: Array,
        hyperparams: FusedLassoHyperparams,
    ) -> Array:

        per_operator_predictions = compute_per_operator_predictions(states)
        solver = ProximalGradient(
            functools.partial(
                self.least_squares_sequence,
                dynamics_model=dynamics_model,
                per_operator_predictions=per_operator_predictions,
                targets=targets,
            ),
            self.prox,
            maxiter=hyperparams.max_iter,
            tol=hyperparams.tol,
        )

        coeffs, _ = solver.run(
            jnp.zeros((states.shape[0], dynamics_model.num_operators)),
            hyperparams_prox=hyperparams.prox_hyperparams,
        )

        return jnp.where(jnp.any(jnp.isnan(coeffs)), jnp.zeros_like(coeffs), coeffs)

    @staticmethod
    @eqx.filter_jit
    def least_squares_sequence(
        coeffs: Array,
        dynamics_model: DecomposedDynamicsModel,
        per_operator_predictions: Array,
        targets: Array,
    ) -> Array:
        return normalized_dynamics_reconstruction_loss(
            dynamics_model, coeffs, targets, per_operator_predictions
        ).sum()
