import functools
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso
from optax import l2_loss

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference import (
    InferenceHyperparams,
    NoObsInferenceHyperparams,
)
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.utils import _reweight_l1


@eqx.filter_jit
def bpdn_df_inference(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    observations: Array,
    hyperparams: InferenceHyperparams,
):
    infer_one_trial = functools.partial(
        _bpdn_df_infer_one_trial,
        observation_model,
        dynamics_model,
        compute_per_operator_predictions,
        hyperparams=hyperparams,
    )
    state = vmap(infer_one_trial)(observations)

    return (
        state[..., : dynamics_model.num_latents],
        state[..., dynamics_model.num_latents :],
    )


@eqx.filter_jit
def _bpdn_df_infer_one_trial(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    observations: Array,
    hyperparams: InferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        _bpdn_df_least_squares,
        prox_non_negative_lasso,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )

    infer_one_timestep = functools.partial(
        _bpdn_df_infer_one_timestep,
        observation_model=observation_model,
        dynamics_model=dynamics_model,
        compute_per_operator_predictions=compute_per_operator_predictions,
        solver=solver,
        hyperparams=hyperparams,
    )
    _, state = lax.scan(
        infer_one_timestep,
        (
            jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
            jnp.bool_(True),
        ),
        observations,
    )

    return state


@eqx.filter_jit
def _bpdn_df_infer_one_timestep(
    carry: tuple[Array, Array],
    observations: Array,
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    solver: ProximalGradient,
    hyperparams: InferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    state, is_first = carry
    prev_latents = state[: dynamics_model.num_latents]
    prev_coeffs = state[dynamics_model.num_latents :]

    per_operator_predictions = compute_per_operator_predictions(prev_latents)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)
    l1_coeff = hyperparams.l1_coeff * jnp.ones(
        dynamics_model.num_latents + dynamics_model.num_operators
    )
    l1_coeff[: dynamics_model.num_latents] = 0

    # solve vanilla L1
    state, _ = solver.run(
        jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        observation_model=observation_model,
        dynamics_model=dynamics_model,
        per_operator_predictions=per_operator_predictions,
        observations=observations,
        prev_coeffs=prev_coeffs,
        prev_latents=prev_latents,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    state, _ = solver.run(
        state,
        hyperparams_prox=_reweight_l1(state, l1_coeff, hyperparams.l1_reweight_coeff),
        obs_model=observation_model,
        dynamics_model=dynamics_model,
        flows=per_operator_predictions,
        y_t=observations,
        c_tminus1=prev_coeffs,
        x_tminus1=prev_latents,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    state = jnp.where(jnp.any(jnp.isnan(state)), jnp.zeros_like(state), state)
    return (
        state,
        jnp.bool_(False),
    ), state


@eqx.filter_jit
def _bpdn_df_least_squares(
    state: Array,
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    per_operator_predictions: Array,
    observations: Array,
    prev_coeffs: Array,
    prev_latents: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    latents = state[: dynamics_model.num_latents]
    coeffs = state[dynamics_model.num_latents :]

    rates = observation_model.predict_rates(latents)
    data_nll = observation_model.neg_log_likelihood(rates, observations)

    predicted_state = dynamics_model.combine_operator_predictions(
        prev_latents, coeffs, per_operator_predictions
    )
    dynamics_recon_loss = dynamics_loss_coeff * l2_loss(predicted_state, latents).sum()

    smooth_loss = smooth_coeff * l2_loss(coeffs, prev_coeffs).sum()

    return data_nll + dynamics_recon_loss + smooth_loss


@eqx.filter_jit
def bpdn_df_inference_no_obs(
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    latents: Array,
    targets: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:

    infer_one_trial = functools.partial(
        _bpdn_df_infer_one_no_obs_trial,
        dynamics_model,
        compute_per_operator_predictions,
        hyperparams=hyperparams,
    )
    return vmap(infer_one_trial)(latents, targets)


@eqx.filter_jit
def _bpdn_df_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    latents: Array,
    targets: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        functools.partial(_bpdn_df_no_obs_least_squares, dynamics_model=dynamics_model),
        hyperparams.prox,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )
    infer_one_timestep = functools.partial(
        _bpdn_df_infer_one_no_obs_timestep,
        dynamics_model=dynamics_model,
        compute_per_operator_predictions=compute_per_operator_predictions,
        solver=solver,
        hyperparams=hyperparams,
    )
    _, C = lax.scan(
        infer_one_timestep,
        (jnp.zeros(dynamics_model.num_operators), jnp.bool_(True)),
        (latents, targets),
    )

    return C


@eqx.filter_jit
def _bpdn_df_infer_one_no_obs_timestep(
    carry: tuple[Array, Array],
    xs: tuple[Array, Array],
    dynamics_model: DecomposedDynamicsModel,
    compute_per_operator_predictions: Callable,
    solver: ProximalGradient,
    hyperparams: NoObsInferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    prev_coeffs, is_first = carry
    latents, targets = xs
    per_operator_predictions = compute_per_operator_predictions(latents)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)

    # solve vanilla L1
    coeffs, _ = solver.run(
        jnp.zeros(dynamics_model.num_operators),
        hyperparams_prox=hyperparams.l1_coeff,
        per_operator_predictions=per_operator_predictions,
        targets=targets,
        latents=latents,
        prev_coeffs=prev_coeffs,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    coeffs, _ = solver.run(
        coeffs,
        hyperparams_prox=_reweight_l1(
            coeffs, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff
        ),
        per_operator_predictions=per_operator_predictions,
        targets=targets,
        latents=latents,
        prev_coeffs=prev_coeffs,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    coeffs = jnp.where(jnp.any(jnp.isnan(coeffs)), jnp.zeros_like(coeffs), coeffs)
    return (coeffs, jnp.bool_(False)), coeffs


@eqx.filter_jit
def _bpdn_df_no_obs_least_squares(
    coeffs: Array,
    dynamics_model: DecomposedDynamicsModel,
    per_operator_predictions: Array,
    targets: Array,
    latents: Array,
    prev_coeffs: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    prediction = dynamics_model.combine_operator_predictions(
        latents, coeffs, per_operator_predictions
    )
    reconstruction_loss = l2_loss(prediction, targets).sum()

    null_prediction = dynamics_model.combine_operator_predictions(
        latents, jnp.zeros_like(coeffs), per_operator_predictions
    )
    variance = jnp.maximum(l2_loss(null_prediction, targets).sum(axis=-1), 1e-2)

    smooth_loss = smooth_coeff * l2_loss(coeffs, prev_coeffs).sum()

    return reconstruction_loss / variance + smooth_loss
