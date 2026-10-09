from typing import Callable, Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    DeltaDynamics,
    OperatorHyperparams,
)
from decomposed_dynamics.dynamics_models.regularization import operator_flow_correlation


class MLPOperatorHyperparams(OperatorHyperparams, eqx.Module):
    normalization_grid: Array
    decorr_coeff: float = 0.01


def visited_grid(x: Array, bin_width: float = 0.1) -> Array:
    """Centers of the grid bins visited by any point in `x`."""
    bins = jnp.unique(jnp.floor(x.reshape(-1, x.shape[-1]) / bin_width), axis=0)
    return (bins + 0.5) * bin_width


class MLPDecomposedDynamics(DecomposedDynamicsModel):
    G: eqx.nn.MLP

    def __init__(
        self,
        num_operators: int,
        state_dim: int,
        key: Array,
        layer_width: int = 5,
        num_hidden_layers: int = 2,
        activation_fn: Callable = jax.nn.relu,
    ):
        super().__init__(
            num_operators,
            state_dim,
            key,
            layer_width=layer_width,
            num_hidden_layers=num_hidden_layers,
            activation_fn=activation_fn,
        )

    @override
    def initialize_params(
        self,
        key: Array,
        layer_width: int,
        num_hidden_layers: int,
        activation_fn: Callable,
        **primitive_kwargs,
    ):
        keys = jr.split(key, self.num_operators)
        self.G = self.initialize_mlps(
            keys, layer_width, num_hidden_layers, activation_fn
        )

    @eqx.filter_vmap(in_axes=(None, 0, None, None, None))
    def initialize_mlps(
        self,
        key: Array,
        layer_width: int,
        num_hidden_layers: int,
        activation_fn: Callable,
    ):
        return eqx.nn.MLP(
            self.state_dim,
            self.state_dim,
            width_size=layer_width,
            depth=num_hidden_layers,
            activation=activation_fn,
            key=key,
        )

    @override
    def compute_operator_predictions(self, x: Array) -> Array:
        flows = self._compute_operator_predictions_batched(
            self.G, x.reshape(-1, self.state_dim)
        )
        return jnp.squeeze(flows)

    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None))
    def _compute_operator_predictions(self, G: eqx.nn.MLP, x: Array) -> Array:
        return G(x)

    _compute_operator_predictions_batched = eqx.filter_vmap(
        _compute_operator_predictions, in_axes=(None, None, 0)
    )

    @override
    def initialize_hyperparams(self, **kwargs) -> OperatorHyperparams:
        return MLPOperatorHyperparams(**kwargs)

    @override
    @eqx.filter_jit
    def regularize_operators(
        self, hyperparams: MLPOperatorHyperparams, latents: Array
    ) -> Self:
        updated_model = self.decorrelate_operators(latents, hyperparams.decorr_coeff)

        return updated_model.normalize_operators(hyperparams.normalization_grid)

    @eqx.filter_jit
    def normalize_operators(self, grid: Array) -> Self:
        """Rescale the last layers so each operator's mean norm over `grid` is 1."""
        norms = jnp.linalg.norm(self.compute_operator_predictions(grid), axis=-1)
        scale = norms.mean(axis=0)
        last = self.G.layers[-1]
        return eqx.tree_at(
            lambda m: (m.G.layers[-1].weight, m.G.layers[-1].bias),
            self,
            (last.weight / scale[:, None, None], last.bias / scale[:, None]),
        )

    @eqx.filter_jit
    def apply_prox(self, **kwargs) -> eqx.nn.MLP:
        return self.G

    @eqx.filter_jit
    def decorrelate_operators(self, latents: Array, decorr_coeff: float) -> Self:
        decorr_gradient = eqx.filter_grad(operator_flow_correlation)(self, latents)
        grad_updates = jax.tree.map(lambda grad: -decorr_coeff * grad, decorr_gradient)
        updated_model = eqx.apply_updates(self, grad_updates)

        return updated_model


class MLPDecomposedDeltaDynamics(DeltaDynamics, MLPDecomposedDynamics):
    pass
