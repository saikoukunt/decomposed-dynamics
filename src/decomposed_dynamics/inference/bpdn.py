import functools

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso
from optax import l2_loss

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel, OperatorParams
from decomposed_dynamics.utils import _reweight_l1


def bpdn_inference(Y: Array, D: Array, operators: OperatorParams, **kwargs):
    pass


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
def _bpdn_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    X: Array,
    smooth_coeff: float = 0.1,
    l1_coeff: float = 0.4,
    max_iter: int = 3000,
    tol: float = 1e-4,
) -> Array:
    solver = ProximalGradient(
        _bpdn_least_squares, prox_non_negative_lasso, maxiter=max_iter, tol=tol
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


@jit(static_argnames=["solver"])
def _bpdn_infer_one_no_obs_timestep(
    carry: tuple[Array, Array],
    Xs: tuple[Array, Array],
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    solver: ProximalGradient,
    l1_coeff: float,
    smooth_coeff: float | Array,
    reweight_coeff: float = 200.0,
) -> tuple[tuple[Array, Array], Array]:

    c_tminus1, is_first = carry
    X_t, X_tplus1 = Xs
    flows = dynamics_model.compute_operator_flows(operators, X_t)
    smooth_coeff = jnp.where(is_first, 0.0, smooth_coeff)

    # solve vanilla L1
    c_t, _ = solver.run(
        jnp.zeros(dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        dynamics_model=dynamics_model,
        flows=flows,
        X_tplus1=X_tplus1,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    c_t, _ = solver.run(
        c_t,
        hyperparams_prox=_reweight_l1(c_t, l1_coeff),
        dynamics_model=dynamics_model,
        flows=flows,
        X_tplus1=X_tplus1,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@jit
def _bpdn_least_squares(
    c_t: Array,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    X_tplus1: Array,
    c_tminus1: Array,
    smooth_coeff: Array,
) -> Array:
    predicted_next_state = dynamics_model.predict_next_state(c_t, flows)
    reconstruction_loss = l2_loss(predicted_next_state, X_tplus1).sum()
    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return reconstruction_loss + smooth_loss
