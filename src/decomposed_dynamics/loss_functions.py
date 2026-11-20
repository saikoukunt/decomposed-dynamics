import equinox as eqx
import jax.numpy as jnp
from jax import Array
from optax import l2_loss

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel

VARIANCE_FLOOR = 1e-4


@eqx.filter_jit
def null_prediction_variance(
    dynamics_model: DecomposedDynamicsModel,
    states: Array,
    targets: Array,
    per_operator_predictions: Array,
    variance_floor: float = VARIANCE_FLOOR,
) -> Array:
    null_prediction = dynamics_model.combine_operator_predictions(
        states, jnp.zeros(per_operator_predictions.shape[:-1]), per_operator_predictions
    )
    return jnp.maximum(l2_loss(null_prediction, targets).sum(axis=-1), variance_floor)


@eqx.filter_jit
def normalized_dynamics_reconstruction_loss(
    dynamics_model: DecomposedDynamicsModel,
    states: Array,
    coeffs: Array,
    targets: Array,
    per_operator_predictions: Array,
    variance_floor: float = VARIANCE_FLOOR,
) -> Array:
    prediction = dynamics_model.combine_operator_predictions(
        states, coeffs, per_operator_predictions
    )
    reconstruction_loss = l2_loss(prediction, targets).sum(axis=-1)
    variance = null_prediction_variance(
        dynamics_model, states, targets, per_operator_predictions, variance_floor
    )

    return reconstruction_loss / variance


@eqx.filter_jit
def coeff_smoothness_loss(coeffs: Array, prev_coeffs: Array) -> Array:
    return l2_loss(coeffs, prev_coeffs).sum(axis=-1)
