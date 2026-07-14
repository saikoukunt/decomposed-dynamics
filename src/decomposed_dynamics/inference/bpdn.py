import functools

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso
from optax import l2_loss

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel, OperatorParams
from decomposed_dynamics.observation_models import ObservationModel, ObservationParams
from decomposed_dynamics.utils import _reweight_l1


def bpdn_inference(
    obs_model: ObservationModel,
    obs_params: ObservationParams,
    dynamics_model,
    operators,
    Y,
    **kwargs,
):
    infer_one_trial = functools.partial(
        _bpdn_infer_one_trial,
        obs_model,
        obs_params,
        dynamics_model,
        operators,
        **kwargs,
    )
    state = vmap(infer_one_trial)(Y)

    return (
        state[..., : dynamics_model.num_latents],
        state[..., dynamics_model.num_latents :],
    )


@jit
def bpdn_inference_no_obs(
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    X: Array,
    **kwargs,
) -> Array:

    infer_one_trial = functools.partial(
        _bpdn_infer_one_no_obs_trial,
        dynamics_model,
        operators,
        **kwargs,
    )
    return vmap(infer_one_trial)(X)


@jit
def _bpdn_infer_one_trial(
    obs_model: ObservationModel,
    obs_params: ObservationParams,
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    Y: Array,
    l1_coeff: float = 0.4,
    dynamics_loss_coeff: float = 0.5,
    smooth_coeff: float = 0.4,
    max_iter: int = 3000,
    tol: float = 1e-4,
) -> Array:
    solver = ProximalGradient(
        _bpdn_least_squares, prox_non_negative_lasso, maxiter=max_iter, tol=tol
    )
    l1_coeff = l1_coeff * jnp.ones(
        dynamics_model.num_latents + dynamics_model.num_operators
    )
    l1_coeff[: dynamics_model.num_latents] = 0

    infer_one_timestep = functools.partial(
        _bpdn_infer_one_timestep,
        dynamics_model=dynamics_model,
        operators=operators,
        solver=solver,
        l1_coeff=l1_coeff,
        dynamics_loss_coeff=dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )
    _, state = lax.scan(
        infer_one_timestep,
        (
            jnp.zeros(dynamics_model.num_latents),
            jnp.zeros(dynamics_model.num_operators),
            jnp.bool_(True),
        ),
        Y,
    )

    return state


@jit
def _bpdn_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    X: Array,
    l1_coeff: float = 0.4,
    smooth_coeff: float = 0.4,
    max_iter: int = 3000,
    tol: float = 1e-4,
) -> Array:
    solver = ProximalGradient(
        _bpdn_no_obs_least_squares, prox_non_negative_lasso, maxiter=max_iter, tol=tol
    )
    infer_one_timestep = functools.partial(
        _bpdn_infer_one_no_obs_timestep,
        dynamics_model=dynamics_model,
        operators=operators,
        solver=solver,
        l1_coeff=l1_coeff,
        smooth_coeff=smooth_coeff,
    )
    _, C = lax.scan(
        infer_one_timestep,
        (jnp.zeros(dynamics_model.num_operators), jnp.bool_(True)),
        (X[:-1, :], X[1:, :]),
    )

    return C


def _bpdn_infer_one_timestep(
    carry: tuple[Array, Array],
    y_t: Array,
    obs_model: ObservationModel,
    obs_params: ObservationParams,
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    solver: ProximalGradient,
    l1_coeff: float,
    dynamics_loss_coeff: float,
    smooth_coeff: float | Array,
    reweight_coeff: float = 200.0,
) -> tuple[tuple[Array, Array, Array], Array]:

    x_tminus1, c_tminus1, is_first = carry
    flows = dynamics_model.compute_operator_flows(operators, x_tminus1)
    smooth_coeff = jnp.where(is_first, 0.0, smooth_coeff)

    # solve vanilla L1
    state = solver.run(
        jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        obs_model=obs_model,
        obs_params=obs_params,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    state = solver.run(
        state,
        hyperparams_prox=_reweight_l1(state, l1_coeff),
        obs_model=obs_model,
        obs_params=obs_params,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    state = jnp.where(jnp.any(jnp.isnan(state)), jnp.zeros_like(state), state)
    return (
        state[: dynamics_model.num_latents],
        state[dynamics_model.num_latents :],
        jnp.bool_(False),
    ), state


def _bpdn_infer_one_no_obs_timestep(
    carry: tuple[Array, Array],
    xs: tuple[Array, Array],
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    solver: ProximalGradient,
    l1_coeff: float,
    smooth_coeff: float | Array,
    reweight_coeff: float = 200.0,
) -> tuple[tuple[Array, Array], Array]:

    c_tminus1, is_first = carry
    x_tminus1, x_t = xs
    flows = dynamics_model.compute_operator_flows(operators, x_tminus1)
    smooth_coeff = jnp.where(is_first, 0.0, smooth_coeff)

    # solve vanilla L1
    c_t, _ = solver.run(
        jnp.zeros(dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    c_t, _ = solver.run(
        c_t,
        hyperparams_prox=_reweight_l1(c_t, l1_coeff),
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@jit
def _bpdn_least_squares(
    state: Array,
    obs_model: ObservationModel,
    obs_params: ObservationParams,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    y_t: Array,
    x_tminus1: Array,
    c_tminus1: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    x_t = state[: dynamics_model.num_latents]
    c_t = state[dynamics_model.num_latents :]

    rates = obs_model.predict_rates(obs_params, x_t)
    data_nll = obs_model.neg_log_likelihood(rates, y_t)

    predicted_state = dynamics_model.predict_next_state(c_t, flows)
    dynamics_recon_loss = dynamics_loss_coeff * l2_loss(predicted_state, x_t).sum()

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return data_nll + dynamics_recon_loss + smooth_loss


@jit
def _bpdn_no_obs_least_squares(
    c_t: Array,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    x_t: Array,
    c_tminus1: Array,
    smooth_coeff: Array,
) -> Array:
    predicted_state = dynamics_model.predict_next_state(c_t, flows)
    reconstruction_loss = l2_loss(predicted_state, x_t).sum()

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return reconstruction_loss + smooth_loss
