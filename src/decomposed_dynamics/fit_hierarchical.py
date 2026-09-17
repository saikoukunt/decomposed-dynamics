import functools
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, vmap
from jaxopt import ProximalGradient
from optax import l2_loss
from tqdm import trange

from decomposed_dynamics.dynamics_models import HierarchicalDecomposedDynamics
from decomposed_dynamics.inference import (
    BPDNDFNoObsInference,
    NoObsInferenceBackend,
    InferenceHyperparams,
)
from decomposed_dynamics.utils import eqx_module_to_string, extract_snippets


def fit_hierarchical_mlps(
    data: dict,
    C: dict,
    dynamics_model: HierarchicalDecomposedDynamics,
    samples_per_snippet: int,
    num_snippets: int,
    lr_init: float,
    lr_end: float,
    filter_spec: HierarchicalDecomposedDynamics,
    max_iter: int = 200,
    inference_backend: NoObsInferenceBackend = BPDNDFNoObsInference(),
    inference_hyperparams: dict | InferenceHyperparams = {},
    prox_hyperparams_max: float = 0.4,
) -> HierarchicalDecomposedDynamics:

    progress_bar = trange(max_iter)

    lr = jnp.linspace(lr_init, lr_end, max_iter)
    prox_hyperparams_schedule = jnp.linspace(
        inference_hyperparams.prox_hyperparams, jnp.array(prox_hyperparams_max), int(max_iter / 2)
    )

    for i in progress_bar:
        latents, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)
        C_batch, _ = extract_snippets(C, num_snippets, samples_per_snippet, seed=i)

        if i < max_iter / 2:
            inference_hyperparams = replace(
                inference_hyperparams, prox_hyperparams=prox_hyperparams_schedule[i]
            )
        else:
            inference_hyperparams = replace(
                inference_hyperparams, prox_hyperparams=prox_hyperparams_schedule[-1]
            )
        mlp_coeffs = inference_backend.infer_batch(
            dynamics_model,
            latents,
            C_batch,
            inference_hyperparams,
            dynamics_model.compute_coeff_predictions,
        )

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

        dynamics_model = updated_model

    return dynamics_model


@eqx.filter_jit
def update_dynamics_model(
    dynamics_model: HierarchicalDecomposedDynamics,
    grads: HierarchicalDecomposedDynamics,
    lr: Array,
):

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
    hyperparams: InferenceHyperparams,
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
    hyperparams: InferenceHyperparams,
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
    predicted_cs = model.combine_operator_predictions(c_t, predicted_cs)
    return l2_loss(predicted_cs, target_c_t).sum()


@eqx.filter_jit
def recon_loss_diff(
    diff_model: HierarchicalDecomposedDynamics,
    static_model: HierarchicalDecomposedDynamics,
    coeffs: Array,
    targets: Array,
    latents: Array,
    loss_weights: Array,
):
    model = eqx.combine(diff_model, static_model)
    predicted_cs = model._compute_coeff_predictions_batched(model.G, latents)
    predicted_cs = model.combine_operator_predictions(coeffs, predicted_cs)

    return (loss_weights * l2_loss(predicted_cs, targets).sum(axis=-1)).sum()


@eqx.filter_jit
def recon_loss_diff_batched(
    diff_model: HierarchicalDecomposedDynamics,
    static_model: HierarchicalDecomposedDynamics,
    coeffs: Array,
    targets: Array,
    latents: Array,
):
    loss = vmap(recon_loss_diff, in_axes=(None, None, 0, 0, 0, 0))
    weights = jnp.linalg.norm(jnp.diff(latents, axis=1), axis=-1) ** 2
    weights = jnp.hstack((weights, jnp.expand_dims(weights[:, -1], axis=1)))
    weights = weights / weights.sum()
    recon_loss = loss(
        diff_model,
        static_model,
        coeffs,
        targets,
        latents,
        weights,
    )

    return recon_loss.sum()


recon_loss_value_and_grad = eqx.filter_jit(
    eqx.filter_value_and_grad(recon_loss_diff_batched)
)
