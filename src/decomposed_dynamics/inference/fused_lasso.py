import functools
from dataclasses import dataclass
from typing import Callable, override

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
    as_prox_hyperparams,
    reweight_prox_hyperparams,
    prox_l1_unit_tv,
)


class FusedLassoHyperparams(InferenceHyperparams):
    prox_hyperparams: tuple[Array, Array] = eqx.field(
        default=(0.25, 0.25), converter=as_prox_hyperparams
    )
    prox_reweight_coeff: tuple[Array, Array] = eqx.field(
        default=(200.0, 200.0), converter=as_prox_hyperparams
    )

    @override
    @classmethod
    def get_backend(
        cls, observation_model: ObservationModel | None = None
    ) -> "FusedLassoNoObsInference":
        if observation_model is not None:
            raise NotImplementedError("fused lasso has no observation model backend")
        return FusedLassoNoObsInference()


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


def _reweight_l1_and_tv(
    coeffs: Array,
    prox_hyperparams: tuple[Array, Array],
    prox_reweight_coeff: tuple[Array, Array],
) -> tuple[Array, Array]:
    l1_coeff, tv_coeff = prox_hyperparams
    l1_reweight_coeff, tv_reweight_coeff = prox_reweight_coeff

    return (
        reweight_prox_hyperparams(coeffs, l1_coeff, l1_reweight_coeff),
        reweight_prox_hyperparams(
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
    ) -> Array:

        infer_trial = functools.partial(
            self.infer_trial,
            dynamics_model,
            hyperparams=hyperparams,
        )
        return lax.map(lambda trial: infer_trial(*trial), (states, targets))

    @override
    @eqx.filter_jit
    def infer_trial(
        self,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: FusedLassoHyperparams,
    ) -> Array:

        per_operator_predictions = vmap(dynamics_model.compute_operator_predictions)(states)
        solver = ProximalGradient(
            functools.partial(
                self.least_squares_sequence,
                dynamics_model=dynamics_model,
                per_operator_predictions=per_operator_predictions,
                states=states,
                targets=targets,
            ),
            self.prox,
            maxiter=hyperparams.max_iter,
            tol=hyperparams.tol,
            stepsize=lambda _, L=_lipschitz_constant(
                dynamics_model, states, per_operator_predictions, targets
            ): (1 / L),
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
        states: Array,
        targets: Array,
    ) -> Array:
        return normalized_dynamics_reconstruction_loss(
            dynamics_model, states, coeffs, targets, per_operator_predictions
        ).sum()
