import functools
from dataclasses import dataclass
from typing import Callable, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax, vmap
from jaxopt import ProximalGradient

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import (
    InferenceHyperparams,
    NoObsInferenceBackend,
    solve_reweighted,
)
from decomposed_dynamics.loss_functions import normalized_dynamics_reconstruction_loss
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.proximal_operators import (
    _reweight_prox_hyperparams,
    prox_l1_unit_tv,
)


class FusedLassoHyperparams(InferenceHyperparams):
    prox_hyperparams: Array = eqx.field(default=(0.25, 0.25), converter=jnp.array)
    prox_reweight_coeff: Array = eqx.field(default=(200.0, 200.0), converter=jnp.array)

    @override
    @classmethod
    def get_backend(
        cls, observation_model: ObservationModel | None = None
    ) -> "FusedLassoNoObsInference":
        if observation_model is not None:
            raise NotImplementedError("fused lasso has no observation model backend")
        return FusedLassoNoObsInference()


def _lipschitz_constant(per_operator_predictions: Array, targets: Array) -> Array:
    variance = jnp.maximum(0.5 * (targets**2).sum(-1), 1e-4)
    gram = jnp.einsum(
        "tki, tli -> tkl", per_operator_predictions, per_operator_predictions
    )

    return jnp.linalg.eigvalsh(gram / variance[:, None, None])[:, -1].max()


def _reweight_l1_and_tv(
    coeffs: Array, prox_hyperparams: Array, prox_reweight_coeff: Array
) -> tuple[Array, Array]:
    l1_coeff, tv_coeff = prox_hyperparams
    l1_reweight_coeff, tv_reweight_coeff = prox_reweight_coeff

    return (
        _reweight_prox_hyperparams(coeffs, l1_coeff, l1_reweight_coeff),
        _reweight_prox_hyperparams(
            jnp.diff(coeffs, axis=0), tv_coeff, tv_reweight_coeff
        ),
    )


@dataclass(frozen=True)
class FusedLassoNoObsInference(NoObsInferenceBackend):
    prox: Callable = prox_l1_unit_tv

    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> FusedLassoHyperparams:
        return FusedLassoHyperparams(**kwargs)

    @override
    @eqx.filter_jit
    def infer_batch(
        self,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: FusedLassoHyperparams,
        compute_per_operator_predictions: Callable,
    ) -> Array:

        infer_trial = functools.partial(
            self.infer_trial,
            dynamics_model,
            compute_per_operator_predictions,
            hyperparams=hyperparams,
        )
        return lax.map(lambda trial: infer_trial(*trial), (states, targets))

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

        per_operator_predictions = vmap(compute_per_operator_predictions)(states)
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
            stepsize=lambda _, L=_lipschitz_constant(per_operator_predictions, targets): (
                1 / L
            ),
        )

        return solve_reweighted(
            solver,
            jnp.zeros((states.shape[0], dynamics_model.num_operators)),
            hyperparams.prox_hyperparams,
            hyperparams.prox_reweight_coeff,
            reweight=_reweight_l1_and_tv,
        )

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
