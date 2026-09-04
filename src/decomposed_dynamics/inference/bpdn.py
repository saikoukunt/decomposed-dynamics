import functools

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
def bpdn_inference(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    observations: Array,
    hyperparams: InferenceHyperparams,
):
    infer_one_trial = functools.partial(
        _bpdn_infer_one_trial,
        observation_model,
        dynamics_model,
        hyperparams=hyperparams,
    )
    state = vmap(infer_one_trial)(observations)

    return (
        state[..., : dynamics_model.num_latents],
        state[..., dynamics_model.num_latents :],
    )


@eqx.filter_jit
def _bpdn_infer_one_trial(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    observations: Array,
    hyperparams: InferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        _bpdn_least_squares,
        prox_non_negative_lasso,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )

    infer_one_timestep = functools.partial(
        _bpdn_infer_one_timestep,
        observation_model=observation_model,
        dynamics_model=dynamics_model,
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
def _bpdn_infer_one_timestep(
    carry: tuple[Array, Array],
    y_t: Array,
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    solver: ProximalGradient,
    hyperparams: InferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    state, is_first = carry
    x_tminus1 = state[: dynamics_model.num_latents]
    c_tminus1 = state[dynamics_model.num_latents :]

    flows = dynamics_model.compute_operator_flows(x_tminus1)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)
    l1_coeff = hyperparams.l1_coeff * jnp.ones(
        dynamics_model.num_latents + dynamics_model.num_operators
    )
    l1_coeff[: dynamics_model.num_latents] = 0

    # solve vanilla L1
    state, _ = solver.run(
        jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        obs_model=observation_model,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        c_tminus1=c_tminus1,
        x_tminus1=x_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    state, _ = solver.run(
        state,
        hyperparams_prox=_reweight_l1(state, l1_coeff, hyperparams.l1_reweight_coeff),
        obs_model=observation_model,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        c_tminus1=c_tminus1,
        x_tminus1=x_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    state = jnp.where(jnp.any(jnp.isnan(state)), jnp.zeros_like(state), state)
    return (
        state,
        jnp.bool_(False),
    ), state


@eqx.filter_jit
def _bpdn_least_squares(
    state: Array,
    obs_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    y_t: Array,
    c_tminus1: Array,
    x_tminus1: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    x_t = state[: dynamics_model.num_latents]
    c_t = state[dynamics_model.num_latents :]

    rates = obs_model.predict_rates(x_t)
    data_nll = obs_model.neg_log_likelihood(rates, y_t)

    predicted_state = dynamics_model.predict_next_state(x_tminus1, c_t, flows)
    dynamics_recon_loss = dynamics_loss_coeff * l2_loss(predicted_state, x_t).sum()

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return data_nll + dynamics_recon_loss + smooth_loss


@eqx.filter_jit
def bpdn_inference_no_obs(
    dynamics_model: DecomposedDynamicsModel,
    X: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:

    infer_one_trial = functools.partial(
        _bpdn_infer_one_no_obs_trial,
        dynamics_model,
        hyperparams=hyperparams,
    )
    return vmap(infer_one_trial)(X)

@eqx.filter_jit
def _bpdn_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    latents: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        _bpdn_no_obs_least_squares,
        prox_non_negative_lasso,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )
    infer_one_timestep = functools.partial(
        _bpdn_infer_one_no_obs_timestep,
        dynamics_model=dynamics_model,
        solver=solver,
        hyperparams=hyperparams,
    )
    _, C = lax.scan(
        infer_one_timestep,
        (jnp.zeros(dynamics_model.num_operators), jnp.bool_(True)),
        (latents[:-1, :], latents[1:, :]),
    )

    return C


@eqx.filter_jit
def _bpdn_infer_one_no_obs_timestep(
    carry: tuple[Array, Array],
    xs: tuple[Array, Array],
    dynamics_model: DecomposedDynamicsModel,
    solver: ProximalGradient,
    hyperparams: NoObsInferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    c_tminus1, is_first = carry
    x_tminus1, x_t = xs
    flows = dynamics_model.compute_operator_flows(x_tminus1)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)

    # solve vanilla L1
    c_t, _ = solver.run(
        jnp.zeros(dynamics_model.num_operators),
        hyperparams_prox=hyperparams.l1_coeff,
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    c_t, _ = solver.run(
        c_t,
        hyperparams_prox=_reweight_l1(
            c_t, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff
        ),
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@eqx.filter_jit
def _bpdn_no_obs_least_squares(
    c_t: Array,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    x_t: Array,
    x_tminus1: Array,
    c_tminus1: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    predicted_state = dynamics_model.predict_next_state(x_tminus1, c_t, flows)
    reconstruction_loss = l2_loss(predicted_state, x_t).sum()
    null_predictions = dynamics_model.predict_next_state(
        x_tminus1, jnp.zeros_like(c_t), flows
    )
    variance = jnp.maximum(l2_loss(null_predictions, x_t).sum(axis=-1), 1e-3)

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return dynamics_loss_coeff * reconstruction_loss / variance + smooth_loss
