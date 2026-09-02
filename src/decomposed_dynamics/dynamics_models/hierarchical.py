from dataclasses import dataclass
from typing import Callable, Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorHyperparams,
)


@dataclass(frozen=True)
class HierarchicalOperatorHyperparams(OperatorHyperparams):
    primitive_hyperparams: OperatorHyperparams


class HierarchicalDecomposedDynamics(DecomposedDynamicsModel):
    G: eqx.nn.MLP
    primitives: DecomposedDynamicsModel
    num_primitives: int
    primitive_type: type[DecomposedDynamicsModel]

    def __init__(
        self,
        num_nonlinear_operators: int,
        num_primitives: int,
        num_latents: int,
        primitive_type: type[DecomposedDynamicsModel],
        key: Array,
        layer_width: int = 5,
        num_hidden_layers: int = 2,
        activation_fn: Callable = jax.nn.relu,
        **primitive_kwargs: Array,
    ):
        self.num_primitives = num_primitives
        self.primitive_type = primitive_type
        super().__init__(
            num_nonlinear_operators,
            num_latents,
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
        keys = jr.split(key, self.num_operators + 1)
        self.primitives = self.primitive_type(
            self.num_primitives, self.num_latents, keys[0], **primitive_kwargs
        )
        self.G = self.initialize_mlps(
            keys[1:], layer_width, num_hidden_layers, activation_fn
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
            self.num_latents,
            self.num_primitives,
            width_size=layer_width,
            depth=num_hidden_layers,
            activation=activation_fn,
            key=key,
        )

    @override
    def initialize_hyperparams(self, **kwargs) -> OperatorHyperparams:
        primitive_hyperparams = self.primitives.initialize_hyperparams(**kwargs)

        return HierarchicalOperatorHyperparams(primitive_hyperparams)

    @override
    def compute_operator_flows(self, x: Array) -> Array:
        primitive_flows = self.primitives.compute_operator_flows(x)
        primitive_coeffs = jnp.squeeze(
            self._compute_coeff_predictions_batched(
                self.G, x.reshape(-1, self.num_latents)
            )
        )

        return jnp.einsum("...mk, ...ki -> ...mi", primitive_coeffs, primitive_flows)

    def compute_coeff_predictions(self, x: Array) -> Array:
        return self._compute_coeff_predictions(self.G, x)

    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None))
    def _compute_coeff_predictions(self, G: eqx.nn.MLP, x: Array) -> Array:
        return G(x)

    _compute_coeff_predictions_batched = eqx.filter_vmap(
        _compute_coeff_predictions, in_axes=(None, None, 0)
    )

    @override
    @eqx.filter_jit
    def regularize_operators(
        self, hyperparams: HierarchicalOperatorHyperparams
    ) -> Self:
        # primitives = self.primitives.regularize_operators(
        # hyperparams.primitive_hyperparams
        # )
        # updated_model = eqx.tree_at(lambda model: model.primitives, self, primitives)
        # return updated_model
        return self

    @eqx.filter_jit
    def apply_prox(self, **kwargs) -> eqx.nn.MLP:
        return self.G

    @eqx.filter_jit
    def decorrelate_operators(self) -> eqx.nn.MLP:
        return self.G
