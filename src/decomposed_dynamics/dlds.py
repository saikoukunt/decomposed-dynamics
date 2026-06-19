import functools

import jax
import jax.numpy as jnp
from jax import Array, grad, jit, lax, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso
from tqdm import trange

from .extract_snippets import extract_snippets
from .extract_snippets import extract_snippets_dict
from .extract_snippets import min_second_dim_size
from .loss_functions import (
    _bpdn_least_squares,
    _dynamics_recon_loss_all,
    _operator_decorr_loss,
)


def fit_no_obs(
    data: dict,
    num_motifs: int,
    samples_per_snippet: int,
    num_snippets: int,
    max_iter: int = 200,
    c_l1_coeff: float = 0.4,
    c_smooth_coeff: float = 0.4,
    c_fista_tol: float = 1e-4,
    c_fista_max_iter: int = 1000,
    F_lr_init: float = 10.0,
    F_lr_decay: float = 0.99995,
    F_decorr_coeff: float = 0.05,
    F_l1_coeff: float = 0.03,
):
    trial_keys = list(data.keys())
    num_latents = data[trial_keys[0]].shape[0]
    min_trial_length = min_second_dim_size(data)
    num_timepoints = min(min_trial_length, samples_per_snippet)

    key = jax.random.key(42)
    F = jax.random.normal(key, (num_motifs, num_latents, num_latents))
    F /= jnp.linalg.norm(F, axis=(1, 2), keepdims=True)

    F_lr = F_lr_init
    pbar = trange(max_iter)
    for i in pbar:
        X, _ = extract_snippets_dict(data, num_snippets, num_timepoints, seed=i)

        C = infer_no_obs_state(
            X,
            F,
            c_smooth_coeff=c_smooth_coeff,
            c_l1_coeff=c_l1_coeff,
            c_fista_max_iter=c_fista_max_iter,
            c_fista_tol=c_fista_tol,
        )
        F_new = update_F(
            C, X, F, lr_F=F_lr, decorr_coeff=F_decorr_coeff, l1_coeff=F_l1_coeff
        )

        reconstruction_error = float(_dynamics_recon_loss_all(C, X, F_new))
        delta_F = float(calculate_delta_F(F_new, F))
        pbar.set_postfix(
            recon_err=f"{reconstruction_error:.4f}", delta_F=f"{delta_F:.6f}"
        )

        F = F_new
        F_lr *= F_lr_decay
    return F

def final_c_fit(data: dict,
    F: Array,
    c_l1_coeff: float = 0.2,
    c_smooth_coeff: float = 0.4,
    c_fista_tol: float = 1e-4,
    c_fista_max_iter: int = 1000,
):
    print(f"Final loop to recompute dynamics coeffients")
    trial_keys = list(data.keys())
    C_final = {}
    pbar = trange(len(trial_keys))
    for i in pbar:
        trial_key = trial_keys[i]
        C_final[trial_key] = infer_no_obs_state(
                jnp.expand_dims(data[trial_key], 0),
                F,
                c_smooth_coeff=c_smooth_coeff,
                c_l1_coeff=c_l1_coeff,
                c_fista_max_iter=c_fista_max_iter,
                c_fista_tol=c_fista_tol,
            )
        reconstruction_error = float(_dynamics_recon_loss_all(C_final[trial_key], jnp.expand_dims(data[trial_key], 0), F))
        pbar.set_postfix(
            recon_err=f"{reconstruction_error:.4f}"
        )
    return C_final


def infer_no_obs_state(
    X: Array,
    F: Array,
    c_l1_coeff: float = 0.2,
    c_smooth_coeff: float = 0.4,
    c_fista_tol: float = 1e-4,
    c_fista_max_iter: int = 1000,
):

    C = update_c_all(
        X,
        F,
        l1_coeff=c_l1_coeff,
        smooth_coeff=c_smooth_coeff,
        max_iter=c_fista_max_iter,
        tol=c_fista_tol,
    )

    return C


@jit
def calculate_delta_F(F_new, F_old):
    return vmap(_calculate_one_delta_F)(F_new, F_old).mean()


@jit
def _calculate_one_delta_F(F_new, F_old):
    delta_F = F_new - F_old
    #delta_F = F_new # - F_old
    delta_F = jnp.einsum("ij, ij", delta_F, delta_F)
    delta_F /= (F_old**2).sum()

    return delta_F


@jit
def update_c_all(X: Array, F: Array, **kwargs):
    update_one_trial = functools.partial(update_c, F=F, **kwargs)
    return vmap(update_one_trial)(X)


@jit
def update_c(
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
    _, C = lax.scan(
        update_c_t,
        (jnp.zeros((F.shape[0])), jnp.bool_(True)),
        (X[:, :-1].T, X[:, 1:].T),
    )

    return C.T


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
        hyperparams_prox=_reweight_l1(c_t, l1_coeff),
        FX_t=FX_t,
        X_tplus1=X_tplus1,
        c_tminus1=c_tminus1,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@jit
def update_F(
    C: Array, X: Array, F: Array, lr_F: float, decorr_coeff: float, l1_coeff: float
):
    dynamics_recon_gradient = grad(_dynamics_recon_loss_all, argnums=2)(C, X, F)
    F = F - lr_F * dynamics_recon_gradient

    decorr_gradient = grad(_operator_decorr_loss)(F)
    F = F - decorr_coeff * decorr_gradient

    # normalize by operator norm
    F = F / jnp.linalg.matrix_norm(F, keepdims=True, ord=2)

    # soft threshold to encourage sparsity TODO: compare this to unweighted
    reweighted_l1 = _reweight_l1(F, l1_coeff)
    F = jnp.sign(F) * jnp.maximum(jnp.abs(F) - reweighted_l1, 0)

    return F


@jit
def _reweight_l1(x: Array, l1_coeff: Array, reweight_coeff: float = 200) -> Array:
    return l1_coeff / (1 + reweight_coeff * jnp.abs(x))
