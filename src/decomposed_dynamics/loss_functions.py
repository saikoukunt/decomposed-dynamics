import equinox as eqx
import jax.numpy as jnp
from jax import Array
from optax import l2_loss

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel


@eqx.filter_jit
def normalized_dynamics_reconstruction_loss(
    dynamics_model: DecomposedDynamicsModel,
    states: Array,
    coeffs: Array,
    targets: Array,
    per_operator_predictions: Array,
    variance_floor: float = 1e-4,
) -> Array:
    prediction = dynamics_model.combine_operator_predictions(
        states, coeffs, per_operator_predictions
    )
    null_prediction = dynamics_model.combine_operator_predictions(
        states, jnp.zeros_like(coeffs), per_operator_predictions
    )
    reconstruction_loss = l2_loss(prediction, targets).sum(axis=-1)
    variance = jnp.maximum(
        l2_loss(null_prediction, targets).sum(axis=-1), variance_floor
    )

    return reconstruction_loss / variance


@eqx.filter_jit
def coeff_smoothness_loss(coeffs: Array, prev_coeffs: Array) -> Array:
    return l2_loss(coeffs, prev_coeffs).sum(axis=-1)
