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
    InferenceHyperparams,
    NoObsInferenceHyperparams,
    bpdn_inference,
    bpdn_inference_no_obs,
)
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.utils import eqx_module_to_string, extract_snippets


def fit(
    data: dict,
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    samples_per_snippet: int,
    num_snippets: int,
    lr_init: float = 10.0,
    lr_decay: float = 0.9995,
    max_iter: int = 200,
    operator_update_hyperparams: dict | OperatorHyperparams = {},
    inference_hyperparams: dict | InferenceHyperparams = {},
) -> tuple[ObservationModel, DecomposedDynamicsModel]:
    if type(operator_update_hyperparams) is dict:
        operator_update_hyperparams = dynamics_model.initialize_hyperparams(
            **operator_update_hyperparams
        )

    if type(inference_hyperparams) is dict:
        inference_hyperparams = InferenceHyperparams(**inference_hyperparams)

    lr = lr_init
    progress_bar = trange(max_iter)

    for i in progress_bar:
        observations, _ = extract_snippets(
            data, num_snippets, samples_per_snippet, seed=i
        )

        latents, operator_coeffs = bpdn_inference(
            observation_model,
            dynamics_model,
            observations,
            inference_hyperparams,
        )

        data_nll, data_nll_grads = data_nll_value_and_grad(
            observation_model, observations, latents
        )
        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            dynamics_model, latents, operator_coeffs
        )

        updated_obs_model, delta_obs_model = update_observation_model(
            observation_model, data_nll_grads, lr
        )
        updated_dynamics_model, delta_dynamics_model = update_dynamics_model(
            dynamics_model,
            dynamics_recon_grads,
            lr,
            operator_update_hyperparams,
        )

        delta_str = f"Data NLL: {data_nll:.4f}, Recon. Loss: {dynamics_recon_loss:.4f}"
        delta_str += eqx_module_to_string(delta_obs_model)
        delta_str += eqx_module_to_string(delta_dynamics_model)
        progress_bar.set_postfix_str(delta_str)

        observation_model = updated_obs_model
        dynamics_model = updated_dynamics_model
        lr *= lr_decay

    return observation_model, dynamics_model


def fit_no_obs(
    data: dict,
    dynamics_model: DecomposedDynamicsModel,
    samples_per_snippet: int,
    num_snippets: int,
    lr_init: float = 10.0,
    lr_decay: float = 0.9995,
    max_iter: int = 200,
    model_update_hyperparams: dict | OperatorHyperparams = {},
    inference_hyperparams: dict | NoObsInferenceHyperparams = {},
) -> DecomposedDynamicsModel:
    if type(model_update_hyperparams) is dict:
        model_update_hyperparams = dynamics_model.initialize_hyperparams(
            **model_update_hyperparams
        )

    if type(inference_hyperparams) is dict:
        inference_hyperparams = NoObsInferenceHyperparams(**inference_hyperparams)

    lr = lr_init
    progress_bar = trange(max_iter)

    for i in progress_bar:
        latents, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)

        operator_coeffs = bpdn_inference_no_obs(
            dynamics_model, latents, inference_hyperparams
        )

        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            dynamics_model, latents, operator_coeffs
        )
        updated_model, delta_model = update_dynamics_model(
            dynamics_model,
            dynamics_recon_grads,
            lr,
            model_update_hyperparams,
        )

        delta_str = f"Recon. Loss: {dynamics_recon_loss:.4f}"
        delta_str += eqx_module_to_string(delta_model)
        progress_bar.set_postfix_str(delta_str)

        dynamics_model = updated_model
        lr *= lr_decay

    return dynamics_model


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
):

    grad_updates = jax.tree.map(lambda grad: -lr * grad, grads)
    updated_model = eqx.apply_updates(dynamics_model, grad_updates)
    updated_model = updated_model.regularize_operators(hyperparams)

    delta_params = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        eqx.filter(updated_model, eqx.is_inexact_array),
        eqx.filter(dynamics_model, eqx.is_inexact_array),
    )
    return updated_model, delta_params


@eqx.filter_jit
def compute_dynamics_recon_loss(
    dynamics_model: DecomposedDynamicsModel,
    latents: Array,
    operator_coeffs: Array,
):
    return vmap(compute_dynamics_recon_loss_sequence, in_axes=(None, 0, 0))(
        dynamics_model, latents, operator_coeffs
    ).mean()


@eqx.filter_jit
def compute_dynamics_recon_loss_sequence(
    dynamics_model: DecomposedDynamicsModel,
    latents: Array,
    operator_coeffs: Array,
):
    flows = dynamics_model.compute_operator_flows(latents[:-1, :])
    predictions = dynamics_model.predict_next_state(
        latents[:-1, :], operator_coeffs, flows
    )
    mse = l2_loss(predictions, latents[1:, :]).sum(axis=-1).mean()

    null_predictions = dynamics_model.predict_next_state(
        latents[:-1, :], jnp.zeros_like(operator_coeffs), flows
    )
    variance = jnp.maximum(
        l2_loss(null_predictions, latents[1:, :]).sum(axis=-1).mean(), 1e-3
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
