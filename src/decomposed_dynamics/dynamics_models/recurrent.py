from dataclasses import dataclass
from typing import Callable, Self, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, grad

from decomposed_dynamics.dynamics_models.base import (
    DecomposedDynamicsModel,
    OperatorHyperparams,
)
from decomposed_dynamics.utils import operator_correlation, reweighted_l1_prox


@dataclass(frozen=True)
class RecurrentOperatorHyperparams(OperatorHyperparams):
    l1_coeff: float = 0.03
    l1_reweight_coeff: float = 200
    decorr_coeff: float = 0.01


class DecomposedRecurrentDynamics(DecomposedDynamicsModel):
    F: Array
    activation_fn: Callable

    def initialize_params(self, activation_fn: Callable, key: Array):
        F = jr.normal(key, (self.num_operators, self.num_latents, self.num_latents))
        self.F = F
        self.activation_fn = activation_fn

    @staticmethod
    def initialize_hyperparams(**kwargs) -> RecurrentOperatorHyperparams:
        return RecurrentOperatorHyperparams(**kwargs)

    @eqx.filter_jit
    def compute_operator_flows(self, x: Array):
        return jnp.einsum("kij, ...j -> ...ki", self.F, x)

    @override
    @eqx.filter_jit
    def predict_next_state(self, x: Array, c: Array, flows: Array, dt: Array) -> Array:
        pre_activation = jnp.einsum("...k, ...ki -> ...i", c, flows)
        return x + dt * (-x + self.activation_fn(pre_activation))

    @eqx.filter_jit
    def regularize_operators(self, hyperparams: RecurrentOperatorHyperparams) -> Self:
        F = self.apply_prox(self.F, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff)
        F = self.decorrelate_operators(F, hyperparams.decorr_coeff)
        updated_model = eqx.tree_at(lambda model: model.F, self, F)

        return updated_model

    @eqx.filter_jit
    def apply_prox(
        self,
        F: Array,
        l1_coeff: float,
        l1_reweight_coeff: float,
    ) -> Array:
        F = reweighted_l1_prox(F, l1_coeff, l1_reweight_coeff)

        return F

    @eqx.filter_jit
    def decorrelate_operators(self, F: Array, operator_decorr_coeff: float) -> Array:
        decorr_gradient = grad(operator_correlation)(F)
        F = F - operator_decorr_coeff * decorr_gradient

        return F


class DecomposedJacobianRecurrentDynamics(DecomposedRecurrentDynamics):
    @eqx.filter_jit
    def compute_operator_flows(self, x: Array):
        return self.activation_fn(-x + jnp.einsum("kij, ...j -> ...ki", self.F, x))

    @override
    @eqx.filter_jit
    def predict_next_state(self, x: Array, c: Array, flows: Array, dt: Array) -> Array:
        flow = jnp.einsum("...k, ...ki -> ...i", c, flows)
        return x + dt * flow
