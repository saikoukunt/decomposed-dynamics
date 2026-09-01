import functools
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, vmap
from jaxopt import ProximalGradient
from optax import l2_loss
from tqdm import trange

from decomposed_dynamics.dynamics_models import HierarchicalDecomposedDynamics
from decomposed_dynamics.inference import (
    NoObsInferenceHyperparams,
    bpdn_inference_no_obs,
)
from decomposed_dynamics.utils import eqx_module_to_string, extract_snippets


def fit_hierarchical_mlps(
    data: dict,
    C: dict,
    dynamics_model: HierarchicalDecomposedDynamics,
    samples_per_snippet: int,
    num_snippets: int,
    lr_init: float,
    filter_spec: HierarchicalDecomposedDynamics,
    max_iter: int = 200,
    inference_hyperparams: dict | NoObsInferenceHyperparams = {},
) -> HierarchicalDecomposedDynamics:

    progress_bar = trange(max_iter)

    lr = jnp.linspace(1e-5, lr_init, max_iter)[::-1]

    for i in progress_bar:
        latents, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)
        C_batch, _ = extract_snippets(C, num_snippets, samples_per_snippet, seed=i)

        latents = latents.reshape(-1, dynamics_model.num_latents)
        C_batch = C_batch.reshape(-1, C_batch.shape[-1])

        mlp_coeffs = infer_mlp_coeffs(
            dynamics_model, latents, C_batch, inference_hyperparams
        )
        # mlp_coeffs = np.zeros((latents.shape[0], dynamics_model.num_operators))
        # mlp_coeffs[:, 0] = 1
        # mlp_coeffs = jnp.array(mlp_coeffs)

        diff_dynamics_model, static_dynamics_model = eqx.partition(
            dynamics_model, filter_spec
        )
        recon_loss, recon_grads = recon_loss_value_and_grad(
            diff_dynamics_model, static_dynamics_model, mlp_coeffs, C_batch, latents
        )
        updated_model, delta_model = update_dynamics_model(
            dynamics_model,
            recon_grads,
            lr[i],
        )

        delta_str = f"Recon. Loss: {recon_loss:.4f}"
        delta_str += eqx_module_to_string(delta_model)
        progress_bar.set_postfix_str(delta_str)

        if i == 1000:
            inference_hyperparams = replace(inference_hyperparams, l1_coeff=0.2)
        if i == 2000:
            inference_hyperparams = replace(inference_hyperparams, l1_coeff=0.4)

        dynamics_model = updated_model

    return dynamics_model


@eqx.filter_jit
def update_dynamics_model(dynamics_model: HierarchicalDecomposedDynamics, grads, lr):

    grad_updates = jax.tree.map(lambda grad: -lr * grad, grads)
    updated_model = eqx.apply_updates(dynamics_model, grad_updates)

    delta_params = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        eqx.filter(updated_model, eqx.is_inexact_array),
        eqx.filter(dynamics_model, eqx.is_inexact_array),
    )

    return updated_model, delta_params


def fine_tune_hierarchical():
    pass


@eqx.filter_jit
def infer_mlp_coeffs(
    model: HierarchicalDecomposedDynamics,
    X: Array,
    C: Array,
    hyperparams: NoObsInferenceHyperparams,
):
    solver = ProximalGradient(
        functools.partial(recon_loss, model=model),
        hyperparams.prox,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )
    return vmap(_infer_mlp_coeffs_one, (None, None, 0, 0, None))(
        model, solver, X, C, hyperparams
    )


@eqx.filter_jit
def _infer_mlp_coeffs_one(
    model: HierarchicalDecomposedDynamics,
    solver: ProximalGradient,
    x_t: Array,
    target_c_t: Array,
    hyperparams: NoObsInferenceHyperparams,
):
    coeffs, _ = solver.run(
        jnp.zeros(model.num_operators),
        hyperparams_prox=hyperparams.l1_coeff,
        target_c_t=target_c_t,
        x_t=x_t,
    )
    coeffs = jnp.where(jnp.any(jnp.isnan(coeffs)), jnp.zeros_like(coeffs), coeffs)
    return coeffs


@eqx.filter_jit
def recon_loss(
    c_t: Array, model: HierarchicalDecomposedDynamics, target_c_t: Array, x_t: Array
):
    predicted_cs = model._compute_coeff_predictions(model.G, x_t)
    predicted_cs = model.predict_next_state(x_t, c_t, predicted_cs)
    return l2_loss(predicted_cs, target_c_t).sum()


@eqx.filter_jit
def recon_loss_diff(
    diff_model: HierarchicalDecomposedDynamics,
    static_model: HierarchicalDecomposedDynamics,
    c_t: Array,
    target_c_t: Array,
    x_t: Array,
):
    model = eqx.combine(diff_model, static_model)
    predicted_cs = model._compute_coeff_predictions_batched(model.G, x_t)
    predicted_cs = model.predict_next_state(x_t, c_t, predicted_cs)
    x_t = x_t.reshape(100, 20, -1)

    weights = jnp.linalg.norm(jnp.diff(x_t, axis=1), axis=-1) ** 2
    weights = jnp.hstack((weights, jnp.expand_dims(weights[:, -1], axis=1)))
    weights = weights / weights.sum()

    return (weights.flatten() * l2_loss(predicted_cs, target_c_t).sum(axis=-1)).sum()


recon_loss_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(recon_loss_diff))
