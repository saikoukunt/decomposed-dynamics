import equinox as eqx
import jax
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
from decomposed_dynamics.observation_models import ObservationModel, ObservationParams
from decomposed_dynamics.utils import extract_snippets


def fit(
    data: Array,
    obs_model: ObservationModel,
    obs_params: ObservationParams,
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
        Y, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)

        X, C = bpdn_inference(
            obs_model,
            obs_params,
            dynamics_model,
            Y,
            **inference_hyperparams,
        )

        data_nll, data_nll_grads = data_nll_value_and_grad(Y, X, obs_model, obs_params)
        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            X, C, dynamics_model
        )

        updated_obs_params, delta_obs_params = update_obs_params(
            obs_model, obs_params, data_nll_grads, lr
        )
        updated_operators, delta_operators = update_dynamics_model(
            dynamics_model,
            dynamics_recon_grads,
            lr,
            operator_update_hyperparams,
        )

        progress_bar.set_postfix_str(
            f"Data NLL: {data_nll:.4f}, Recon. Loss: {dynamics_recon_loss:.4f},  \U0001d6abop: {delta_obs_params},\U0001d6abop: {delta_operators}"
        )

        obs_params = updated_obs_params
        operators = updated_operators
        lr *= lr_decay

    return obs_params, operators


def fit_no_obs(
    data: Array,
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
        X, _ = extract_snippets(data, num_snippets, samples_per_snippet, seed=i)

        C = bpdn_inference_no_obs(dynamics_model, X, inference_hyperparams)

        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            dynamics_model, X, C
        )
        updated_model, delta_model = update_dynamics_model(
            dynamics_model,
            dynamics_recon_grads,
            lr,
            model_update_hyperparams,
        )

        delta_str = ""
        for path, val in jax.tree.leaves_with_path(delta_model):
            delta_str += f", Avg \U0001d6ab{jax.tree_util.keystr(path)[1:]}: {val:.5f}"
        progress_bar.set_postfix_str(
            f"Recon. Loss: {dynamics_recon_loss:.4f}{delta_str}"
        )

        dynamics_model = updated_model
        lr *= lr_decay

    return dynamics_model


@eqx.filter_jit
def update_obs_params(
    obs_model: ObservationModel,
    obs_params: ObservationParams,
    grads,
    lr,
):
    updated_obs_params = jax.tree.map(
        lambda param, grad: param - lr * grad, obs_params, grads
    )
    updated_obs_params = obs_model.apply_prox(updated_obs_params)

    delta_obs_params = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        updated_obs_params,
        obs_params,
    )
    return updated_obs_params, delta_obs_params


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
    X: Array,
    C: Array,
):
    return vmap(compute_dynamics_recon_loss_sequence, in_axes=(None, 0, 0))(
        dynamics_model, X, C
    ).mean()


@eqx.filter_jit
def compute_dynamics_recon_loss_sequence(
    dynamics_model: DecomposedDynamicsModel,
    X: Array,
    C: Array,
):
    flows = dynamics_model.compute_operator_flows(X[:-1, :])
    predictions = dynamics_model.predict_next_state(C, flows)

    return l2_loss(predictions, X[1:, :]).sum(axis=-1).mean()


dynamics_recon_value_and_grad = eqx.filter_jit(
    eqx.filter_value_and_grad(compute_dynamics_recon_loss)
)


@eqx.filter_jit
def compute_data_nll(
    obs_model: ObservationModel,
    observations: Array,
    x: Array,
    params: ObservationParams,
):
    rates = obs_model.predict_rates(params, x)
    return obs_model.neg_log_likelihood(rates, observations)


data_nll_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(compute_data_nll))
