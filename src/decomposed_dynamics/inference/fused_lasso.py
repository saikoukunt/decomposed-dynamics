import functools
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap
from jaxopt import ProximalGradient
from optax import l2_loss

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import NoObsInferenceHyperparams
from decomposed_dynamics.utils import _reweight_l1, prox_l1_unit_tv


class FusedLassoNoObsInferenceHyperparams(NoObsInferenceHyperparams):
    hyperparams_prox: Array
    reweight_coeffs_prox: Array
    prox: Callable = prox_l1_unit_tv


@eqx.filter_jit
def fused_lasso_infer_no_obs(
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    latents: Array,
    targets: Array,
    hyperparams: FusedLassoNoObsInferenceHyperparams,
) -> Array:
    infer_one_trial = functools.partial(
        _fused_lasso_infer_one_no_obs_trial,
        dynamics_model,
        compute_per_operator_predictions,
        hyperparams=hyperparams,
    )
    return vmap(infer_one_trial)(latents, targets)


@eqx.filter_jit
def _fused_lasso_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    latents: Array,
    targets: Array,
    hyperparams: FusedLassoNoObsInferenceHyperparams,
) -> Array:

    per_operator_predictions = compute_per_operator_predictions(latents)

    solver = ProximalGradient(
        functools.partial(
            _least_squares_sequence,
            dynamics_model=dynamics_model,
            per_operator_predictions=per_operator_predictions,
            targets=targets,
            latents=latents,
        ),
        hyperparams.prox,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )

    coeffs, _ = solver.run(
        jnp.zeros(latents.shape[0], dynamics_model.num_operators),
        hyperparams_prox=hyperparams.hyperparams_prox,
    )

    coeffs, _ = solver.run(
        coeffs,
        hyperparams_prox=_reweight_l1(
            coeffs, hyperparams.hyperparams_prox, hyperparams.reweight_coeffs_prox
        ),
    )

    coeffs = jnp.where(jnp.any(jnp.isnan(coeffs)), jnp.zeros_like(coeffs), coeffs)
    return coeffs


@eqx.filter_jit
def _least_squares_sequence(
    coeffs: Array,
    dynamics_model: DecomposedDynamicsModel,
    per_operator_predictions: Array,
    targets: Array,
    latents: Array,
) -> Array:
    prediction = dynamics_model.combine_operator_predictions(
        latents, coeffs, per_operator_predictions
    )
    reconstruction_loss = l2_loss(prediction, targets).sum(axis=-1)

    null_prediction = dynamics_model.combine_operator_predictions(
        latents, jnp.zeros_like(coeffs), per_operator_predictions
    )
    variance = jnp.maximum(l2_loss(null_prediction, targets).sum(axis=-1), 1e-2)

    return (reconstruction_loss / variance).sum()
