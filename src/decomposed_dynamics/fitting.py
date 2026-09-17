import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, vmap
from optax import l2_loss
from tqdm import trange

from decomposed_dynamics.dynamics_models import (
    DecomposedDynamicsModel,
    OperatorHyperparams,
)
from decomposed_dynamics.inference import (
    InferenceBackend,
    InferenceHyperparams,
    NoObsInferenceBackend,
)
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.utils import eqx_module_to_string, extract_snippets


def fit(
    data: dict,
    dynamics_model: DecomposedDynamicsModel,
    samples_per_snippet: int,
    num_snippets: int,
    observation_model: ObservationModel | None = None,
    lr_init: float = 10.0,
    lr_end: None | float = None,
    max_iter: int = 200,
    inference_backend: InferenceBackend | NoObsInferenceBackend | None = None,
    inference_hyperparams: InferenceHyperparams = None,
    model_update_hyperparams: dict | OperatorHyperparams = {},
    filter_spec=None,
    prox_hyperparams_end: None | float = None,
) -> tuple[ObservationModel | None, DecomposedDynamicsModel]:
    """Fit `dynamics_model`, jointly with `observation_model` when one is given."""

    # TODO: wrap parameter initialization into a function
    if inference_backend is None and inference_hyperparams is None:
        raise ValueError("pass an inference backend, inference hyperparams, or both")
    if inference_backend is None:
        inference_backend = inference_hyperparams.get_backend(observation_model)
    elif inference_hyperparams is None:
        inference_hyperparams = inference_backend.initialize_hyperparams()

    if type(model_update_hyperparams) is dict:
        model_update_hyperparams = dynamics_model.initialize_hyperparams(
            **model_update_hyperparams
        )

    if filter_spec is None:
        filter_spec = jax.tree_util.tree_map(lambda _: True, dynamics_model)

    if lr_end is None:
        lr_end = lr_init
    lr = jnp.linspace(lr_init, lr_end, max_iter)

    if prox_hyperparams_end is None:
        prox_hyperparams_end = inference_hyperparams.prox_hyperparams
    prox_hyperparams_schedule = jnp.linspace(
        inference_hyperparams.prox_hyperparams, prox_hyperparams_end, int(max_iter / 2)
    )

    progress_bar = trange(max_iter)

    for i in progress_bar:
        snippets, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)

        inference_hyperparams = eqx.tree_at(
            lambda hyperparams: hyperparams.prox_hyperparams,
            inference_hyperparams,
            prox_hyperparams_schedule[min(i, prox_hyperparams_schedule.shape[0] - 1)],
        )

        if observation_model is None:
            latents = snippets
            operator_coeffs = inference_backend.infer_batch(
                dynamics_model,
                latents[:, :-1, :],
                latents[:, 1:, :],
                inference_hyperparams,
                dynamics_model.compute_operator_predictions,
            )
        else:
            latents, operator_coeffs = inference_backend.infer_batch(
                observation_model,
                dynamics_model,
                snippets,
                inference_hyperparams,
                dynamics_model.compute_operator_predictions,
            )

        diff_dynamics_model, static_dynamics_model = eqx.partition(
            dynamics_model, filter_spec
        )
        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            diff_dynamics_model, static_dynamics_model, latents, operator_coeffs
        )
        dynamics_model, delta_dynamics_model = update_dynamics_model(
            dynamics_model,
            dynamics_recon_grads,
            lr[i],
            model_update_hyperparams,
            latents,
        )

        delta_str = f"Recon. Loss: {dynamics_recon_loss:.4f}"

        if observation_model is not None:
            data_nll, data_nll_grads = data_nll_value_and_grad(
                observation_model, snippets, latents
            )
            observation_model, delta_obs_model = update_observation_model(
                observation_model, data_nll_grads, lr[i]
            )
            delta_str = f"Data NLL: {data_nll:.4f}, " + delta_str
            delta_str += eqx_module_to_string(delta_obs_model)

        delta_str += eqx_module_to_string(delta_dynamics_model)
        progress_bar.set_postfix_str(delta_str)

    return observation_model, dynamics_model


@eqx.filter_jit
def update_observation_model(
    obs_model: ObservationModel,
    grads,
    lr,
):
    grad_updates = jax.tree.map(lambda grad: -lr * grad, grads)
    updated_model = eqx.apply_updates(obs_model, grad_updates)
    updated_model = updated_model.apply_prox()

    delta_params = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        eqx.filter(updated_model, eqx.is_inexact_array),
        eqx.filter(obs_model, eqx.is_inexact_array),
    )

    return updated_model, delta_params


@eqx.filter_jit
def update_dynamics_model(
    dynamics_model: DecomposedDynamicsModel,
    grads,
    lr,
    hyperparams: OperatorHyperparams,
    latents: Array,
):

    grad_updates = jax.tree.map(lambda grad: -lr * grad, grads)
    updated_model = eqx.apply_updates(dynamics_model, grad_updates)
    updated_model = updated_model.regularize_operators(hyperparams, latents)

    delta_params = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        eqx.filter(updated_model, eqx.is_inexact_array),
        eqx.filter(dynamics_model, eqx.is_inexact_array),
    )
    return updated_model, delta_params


@eqx.filter_jit
def compute_dynamics_recon_loss(
    diff_dynamics_model: DecomposedDynamicsModel,
    static_dynamics_model: DecomposedDynamicsModel,
    states: Array,
    operator_coeffs: Array,
):
    dynamics_model = eqx.combine(diff_dynamics_model, static_dynamics_model)
    return vmap(compute_dynamics_recon_loss_sequence, in_axes=(None, 0, 0))(
        dynamics_model, states, operator_coeffs
    ).mean()


@eqx.filter_jit
def compute_dynamics_recon_loss_sequence(
    dynamics_model: DecomposedDynamicsModel,
    states: Array,
    operator_coeffs: Array,
):
    flows = dynamics_model.compute_operator_predictions(states[:-1, :])
    predictions = dynamics_model.combine_operator_predictions(operator_coeffs, flows)
    mse = l2_loss(predictions, states[1:, :]).sum(axis=-1).mean()

    variance = jnp.maximum(
        l2_loss(jnp.zeros_like(predictions), states[1:, :]).sum(axis=-1).mean(), 1e-4
    )

    return mse / variance


dynamics_recon_value_and_grad = eqx.filter_jit(
    eqx.filter_value_and_grad(compute_dynamics_recon_loss)
)


@eqx.filter_jit
def compute_data_nll(
    obs_model: ObservationModel,
    observations: Array,
    latents: Array,
):
    rates = obs_model.predict_rates(latents)
    return obs_model.neg_log_likelihood(rates, observations)


data_nll_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(compute_data_nll))
