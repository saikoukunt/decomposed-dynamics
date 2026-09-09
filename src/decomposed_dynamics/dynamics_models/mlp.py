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
class MLPOperatorHyperparams(OperatorHyperparams):
    pass


class MLPDecomposedDynamics(DecomposedDynamicsModel):
    G: eqx.nn.MLP

    def __init__(
        self,
        num_operators: int,
        num_latents: int,
        key: Array,
        layer_width: int = 5,
        num_hidden_layers: int = 2,
        activation_fn: Callable = jax.nn.relu,
    ):
        super().__init__(
            num_operators,
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
            self.num_latents,
            self.num_latents,
            width_size=layer_width,
            depth=num_hidden_layers,
            activation=activation_fn,
            key=key,
        )

    @override
    def compute_operator_flows(self, x: Array) -> Array:
        flows = self._compute_operator_flows_batched(
            self.G, x.reshape(-1, self.num_latents)
        )
        return jnp.squeeze(flows)

    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None))
    def _compute_operator_flows(self, G: eqx.nn.MLP, x: Array) -> Array:
        return G(x)

    _compute_operator_flows_batched = eqx.filter_vmap(
        _compute_operator_flows, in_axes=(None, None, 0)
    )

    @override
    def initialize_hyperparams(self, **kwargs) -> OperatorHyperparams:
        return MLPOperatorHyperparams()

    @override
    @eqx.filter_jit
    def regularize_operators(self, hyperparams: MLPOperatorHyperparams) -> Self:
        return self

    @eqx.filter_jit
    def apply_prox(self, **kwargs) -> eqx.nn.MLP:
        return self.G

    @eqx.filter_jit
    def decorrelate_operators(self) -> eqx.nn.MLP:
        return self.G
