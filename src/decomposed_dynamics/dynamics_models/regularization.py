import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel


@jit
def spectral_normalize(F: Array):
    return F / jnp.linalg.matrix_norm(F, keepdims=True, ord=2)


@jit
def operator_correlation(F: Array) -> Array:
    pairwise_corrs = jnp.einsum("kij, lij -> kl", F, F)
    return jnp.sum(jnp.triu(jnp.clip(pairwise_corrs, 0) ** 2, k=1))


@eqx.filter_jit
def operator_flow_correlation(
    model: DecomposedDynamicsModel, locations: Array
) -> Array:
    locations = locations.reshape(-1, model.state_dim)
    flows = model.compute_operator_flows(locations).transpose(1, 0, 2)

    flows = flows.reshape(model.num_operators, -1)
    flows = flows / (jnp.linalg.norm(flows, axis=-1, keepdims=True) + 1e-8)
    pairwise_corrs = jnp.einsum("in, jn -> ij", flows, flows)

    num_entries = model.num_operators * (model.num_operators - 1) / 2

    return jnp.sum(jnp.triu(jnp.clip(pairwise_corrs, 0.0) ** 2, k=1)) / num_entries
