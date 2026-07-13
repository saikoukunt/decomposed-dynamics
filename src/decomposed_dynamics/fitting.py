import jax
from jax import Array, jit, value_and_grad, vmap
from optax import l2_loss
from tqdm import trange

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel, OperatorParams
from decomposed_dynamics.inference import bpdn_inference_no_obs
from decomposed_dynamics.observation_models import ObservationModel, ObservationParams
from decomposed_dynamics.utils import extract_snippets


# TODO: make data classes for params
def fit_no_obs(
    data: Array,
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    samples_per_snippet: int,
    num_snippets: int,
    lr_init: float = 10.0,
    lr_decay: float = 0.9995,
    max_iter: int = 200,
    operator_decorr_params: dict = {"operator_decorr_coeff": 0.05},
    operator_prox_params: dict = {"l1_coeff": 0.05, "l1_reweight_coeff": 200},
    inference_params: dict = {
        "l1_coeff": 0.2,
        "smooth_coeff": 0.4,
        "max_iter": 1000,
        "tol": 1e-4,
    },
) -> OperatorParams:
    num_timepoints = min(data.shape[1], samples_per_snippet)

    lr = lr_init
    progress_bar = trange(max_iter)

    for i in progress_bar:
        X, _ = extract_snippets(data, num_snippets, num_timepoints, seed=i)

        C = bpdn_inference_no_obs(dynamics_model, operators, X, **inference_params)

        dynamics_recon_loss, dynamics_recon_grads = dynamics_recon_value_and_grad(
            X, C, dynamics_model, operators
        )
        updated_operators, delta_operators = update_operators(
            dynamics_model,
            operators,
            dynamics_recon_grads,
            lr,
            operator_decorr_params,
            operator_prox_params,
        )
        progress_bar.set_postfix_str(
            f"Recon. Loss: {dynamics_recon_loss:.4f}, \U0001d6abop: {delta_operators}"
        )

        operators = updated_operators
        lr *= lr_decay

    return operators


@jit
def update_operators(
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
    grads,
    lr,
    operator_decorr_params,
    operator_prox_params,
):
    updated_operators = jax.tree.map(
        lambda param, grad: param - lr * grad, operators, grads
    )
    updated_operators = dynamics_model.decorrelate_operators(
        updated_operators, **operator_decorr_params
    )
    updated_operators = dynamics_model.apply_prox(
        updated_operators, **operator_prox_params
    )

    delta_operators = jax.tree.map(
        lambda new, old: ((new - old) ** 2).sum() / (old**2).sum(),
        updated_operators,
        operators,
    )
    return updated_operators, delta_operators


@jit
def compute_dynamics_recon_loss(
    X: Array,
    C: Array,
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
):
    return vmap(compute_dynamics_recon_loss_sequence, in_axes=(0, 0, None, None))(
        X, C, dynamics_model, operators
    ).mean()


@jit
def compute_dynamics_recon_loss_sequence(
    X: Array,
    C: Array,
    dynamics_model: DecomposedDynamicsModel,
    operators: OperatorParams,
):
    flows = dynamics_model.compute_operator_flows(operators, X[:-1, :])
    predictions = dynamics_model.predict_next_state(C, flows)

    return l2_loss(predictions, X[1:, :]).sum(axis=-1).mean()


dynamics_recon_value_and_grad = jit(value_and_grad(compute_dynamics_recon_loss, -1))


@jit
def compute_data_nll(
    observations: Array,
    x: Array,
    observation_model: ObservationModel,
    params: ObservationParams,
):
    rates = observation_model.predict_rates(params, x)
    return observation_model.neg_log_likelihood(rates, observations)


data_nll_value_and_grad = jit(value_and_grad(compute_data_nll, -1), static_argnums=0)
