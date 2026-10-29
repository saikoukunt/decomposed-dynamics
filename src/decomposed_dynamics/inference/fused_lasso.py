import functools
from typing import override

import equinox as eqx
import jax
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
    L1UnitTVProx,
    Penalty,
    ProximalOperator,
    apply_prox,
)


class FusedLassoHyperparams(InferenceHyperparams):
    prox: ProximalOperator = L1UnitTVProx(
        Penalty(0.25, reweight=200.0), Penalty(0.25, reweight=200.0)
    )

    @override
    @classmethod
    def get_backend(
        cls, observation_model: ObservationModel | None = None
    ) -> "type[FusedLassoNoObsInference]":
        if observation_model is not None:
            raise NotImplementedError("fused lasso has no observation model backend")
        return FusedLassoNoObsInference


def _lipschitz_constant(
    dynamics_model: DecomposedDynamicsModel,
    states: Array,
    per_operator_predictions: Array,
    targets: Array,
) -> Array:
    zero_coeffs = jnp.zeros(per_operator_predictions.shape[:-1])
    null_predictions = dynamics_model.combine_operator_predictions(
        states, zero_coeffs, per_operator_predictions
    )
    variance = jnp.maximum(0.5 * ((targets - null_predictions) ** 2).sum(-1), 1e-4)
    # combine is affine in the coefficients, so its jacobian gives the effective atoms
    atoms = vmap(jax.jacfwd(dynamics_model.combine_operator_predictions, argnums=1))(
        states, zero_coeffs, per_operator_predictions
    )
    gram = jnp.einsum("tik, til -> tkl", atoms, atoms)

    return jnp.linalg.eigvalsh(gram / variance[:, None, None])[:, -1].max()


class FusedLassoNoObsInference(NoObsInferenceBackend):
    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> FusedLassoHyperparams:
        return FusedLassoHyperparams(**kwargs)

    @override
    @classmethod
    @eqx.filter_jit
    def infer_batch(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: FusedLassoHyperparams,
    ) -> Array:

        infer_trial = functools.partial(
            cls.infer_trial,
            dynamics_model,
            hyperparams=hyperparams,
        )
        return lax.map(lambda trial: infer_trial(*trial), (states, targets))

    @override
    @classmethod
    @eqx.filter_jit
    def infer_trial(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: FusedLassoHyperparams,
    ) -> Array:

        per_operator_predictions = vmap(dynamics_model.compute_operator_predictions)(
            states
        )
        solver = ProximalGradient(
            functools.partial(
                cls.least_squares_sequence,
                dynamics_model=dynamics_model,
                per_operator_predictions=per_operator_predictions,
                states=states,
                targets=targets,
            ),
            apply_prox,
            maxiter=hyperparams.max_iter,
            tol=hyperparams.tol,
            stepsize=lambda _, L=_lipschitz_constant(dynamics_model, states, per_operator_predictions, targets): (
                1 / L
            ),
        )

        return solve_reweighted(
            solver,
            jnp.zeros((states.shape[0], dynamics_model.num_operators)),
            hyperparams.prox,
        )

    @staticmethod
    @eqx.filter_jit
    def least_squares_sequence(
        coeffs: Array,
        dynamics_model: DecomposedDynamicsModel,
        per_operator_predictions: Array,
        states: Array,
        targets: Array,
    ) -> Array:
        return normalized_dynamics_reconstruction_loss(
            dynamics_model, states, coeffs, targets, per_operator_predictions
        ).sum()
