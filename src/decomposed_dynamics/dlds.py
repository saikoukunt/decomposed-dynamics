import functools

import jax.numpy as jnp
from jax import Array, grad, jit, lax
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso

from .loss_functions import (
    _bpdn_least_squares,
    _dynamics_recon_loss_all,
    _operator_decorr_loss,
)


@jit
def update_c(
    C: Array,
    X: Array,
    F: Array,
    smooth_coeff: float = 0,
    l1_coeff: float = 0,
    max_iter: int = 3000,
    tol: float = 1e-4,
) -> Array:
    solver = ProximalGradient(
        _bpdn_least_squares, prox_non_negative_lasso, maxiter=max_iter, tol=tol
    )
    update_c_t = functools.partial(
        _update_c_t,
        F=F,
        solver=solver,
        l1_coeff=l1_coeff,
        smooth_coeff=smooth_coeff,
    )
    _, C = lax.scan(update_c_t, (C[:, 0], jnp.bool_(True)), (X[:, :-1].T, X[:, 1:].T))

    return C


def _update_c_t(
    carry: tuple[Array, Array],
    Xs: tuple[Array, Array],
    F: Array,
    solver: ProximalGradient,
    l1_coeff: float,
    smooth_coeff: float | Array,
    reweight_coeff: float = 200.0,
) -> tuple[tuple[Array, Array], Array]:

    c_tminus1, is_first = carry
    X_t, X_tplus1 = Xs
    FX_t = F @ X_t
    smooth_coeff = jnp.where(is_first, 0.0, smooth_coeff)

    # solve vanilla L1
    c_t, _ = solver.run(
        jnp.zeros_like(c_tminus1),
        hyperparams_prox=l1_coeff,
        FX_t=FX_t,
        X_tplus1=X_tplus1,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    c_t, _ = solver.run(
        c_t,
        hyperparams_prox=l1_coeff / (1 + reweight_coeff * jnp.abs(c_t)),
        FX_t=FX_t,
        X_tplus1=X_tplus1,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@jit(static_argnames=["normalize_F"])
def update_F(
    C: Array,
    X: Array,
    F: Array,
    lr_f: float,
    decorr_coeff: float,
    normalize_F: bool = True,
):
    dynamics_recon_gradient = grad(_dynamics_recon_loss_all, argnums=2)(C, X, F)
    F = F - lr_f * dynamics_recon_gradient

    decorr_gradient = grad(_operator_decorr_loss)(F)
    F = F - decorr_coeff * decorr_gradient

    if normalize_F:
        F = F / jnp.linalg.matrix_norm(F, keepdims=True, ord=2)

    # TODO: Add soft-thresholding for L1 regularization

    return F
