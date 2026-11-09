from typing import override

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax, vmap

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import (
    InferenceHyperparams,
    NoObsInferenceBackend,
    Penalty,
)
from decomposed_dynamics.loss_functions import normalized_dynamics_reconstruction_loss


class BinaryCoordinateDescentHyperparams(InferenceHyperparams):
    l0: Penalty = Penalty(0.1)
    group: Penalty = Penalty(0.5)
    smooth_coeff: Penalty = Penalty(0.4)
    num_sweeps: int = eqx.field(static=True, default=10)


def viterbi(state_objective_costs: Array, switch_costs: Array) -> tuple[Array, Array]:

    def forward(best_prev_sequence_cost, state_objective_cost):
        """For each state at the current timestep, finds the lowest cost previous state sequence"""

        prev_sequence_plus_switch_cost = best_prev_sequence_cost[:, None] + switch_costs

        best_sequence_plus_switch_cost = prev_sequence_plus_switch_cost.min(axis=0)
        new_sequence_cost = best_sequence_plus_switch_cost + state_objective_cost

        return (
            new_sequence_cost,
            prev_sequence_plus_switch_cost.argmin(axis=0),
        )

    best_sequence_cost_per_final_state, best_prev_state = lax.scan(
        forward, state_objective_costs[0], state_objective_costs[1:]
    )

    def trace_sequence_backwards(current_state, best_prev_state_per_current_state):
        prev_state = best_prev_state_per_current_state[current_state]
        return prev_state, prev_state

    final_state = jnp.argmin(best_sequence_cost_per_final_state)
    _, best_sequence = lax.scan(
        trace_sequence_backwards, final_state, best_prev_state, reverse=True
    )

    return best_sequence_cost_per_final_state[final_state], jnp.append(
        best_sequence, final_state
    )


class BinaryCoordinateDescentNoObsInference(NoObsInferenceBackend):
    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> BinaryCoordinateDescentHyperparams:
        return BinaryCoordinateDescentHyperparams(**kwargs)

    @override
    @classmethod
    @eqx.filter_jit
    def infer_batch(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: BinaryCoordinateDescentHyperparams,
    ) -> Array:
        objective = BinaryObjective(dynamics_model, states, targets, hyperparams)
        initial_coeffs = best_single_operator_per_timepoint(objective)

        def coordinate_descent(step, coeffs):
            operator_idx = step % dynamics_model.num_operators
            updated_coeffs = update_one_operator_coeffs(objective, operator_idx, coeffs)
            coeffs = jnp.where(
                objective.batch(updated_coeffs) <= objective.batch(coeffs),
                updated_coeffs,
                coeffs,
            )
            coeffs = try_replacing_operator(objective, operator_idx, coeffs)

            return coeffs

        return lax.fori_loop(
            0,
            hyperparams.num_sweeps * dynamics_model.num_operators,
            coordinate_descent,
            initial_coeffs,
        )

    @override
    @staticmethod
    def infer_trial(
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: BinaryCoordinateDescentHyperparams,
    ) -> Array:
        raise NotImplementedError("the group penalty couples trials; use infer_batch")


class BinaryObjective(eqx.Module):
    dynamics_model: DecomposedDynamicsModel
    states: Array
    targets: Array
    per_operator_predictions: Array
    hyperparams: BinaryCoordinateDescentHyperparams

    def __init__(self, dynamics_model, states, targets, hyperparams):
        self.dynamics_model = dynamics_model
        self.states = states
        self.targets = targets
        self.per_operator_predictions = vmap(
            vmap(dynamics_model.compute_operator_predictions)
        )(states)
        self.hyperparams = hyperparams

    def per_timepoint(self, coeffs: Array) -> Array:
        recon_loss = vmap(normalized_dynamics_reconstruction_loss, (None, 0, 0, 0, 0))(
            self.dynamics_model,
            self.states,
            coeffs,
            self.targets,
            self.per_operator_predictions,
        )
        return recon_loss + self.hyperparams.l0 * coeffs.sum(-1)

    def batch(self, coeffs: Array) -> Array:
        num_timepoints = coeffs.shape[0] * coeffs.shape[1]
        smoothness_cost = (
            self.hyperparams.smooth_coeff
            * 0.5
            * jnp.abs(jnp.diff(coeffs, axis=1)).sum()
        )
        group_l0 = coeffs.any(axis=(0, 1)).sum()
        return (
            self.per_timepoint(coeffs).sum() + smoothness_cost
        ) / num_timepoints + self.hyperparams.group * group_l0


def compute_objective_one_operator_on(
    objective: BinaryObjective, operator_idx: Array
) -> Array:
    all_operators_off = jnp.zeros(
        (*objective.states.shape[:2], objective.dynamics_model.num_operators)
    )
    return objective.per_timepoint(all_operators_off.at[..., operator_idx].set(1.0))


def best_single_operator_per_timepoint(objective: BinaryObjective) -> Array:
    num_operators = objective.dynamics_model.num_operators
    objective_one_operator_on = vmap(compute_objective_one_operator_on, (None, 0), -1)(
        objective, jnp.arange(num_operators)
    )
    return jnp.eye(num_operators)[objective_one_operator_on.argmin(-1)]


def update_one_operator_coeffs(
    objective: BinaryObjective, operator_idx: int, coeffs: Array
) -> Array:

    coeffs_operator_off = coeffs.at[..., operator_idx].set(0.0)
    no_other_operator_on = coeffs_operator_off.sum(-1) == 0

    objective_operator_off = jnp.where(
        no_other_operator_on, jnp.inf, objective.per_timepoint(coeffs_operator_off)
    )
    objective_operator_on = objective.per_timepoint(
        coeffs.at[..., operator_idx].set(1.0)
    )
    switch_costs = objective.hyperparams.smooth_coeff * 0.5 * (1 - jnp.eye(2))

    _, on_off_sequence = vmap(viterbi, (0, None))(
        jnp.stack([objective_operator_off, objective_operator_on], -1), switch_costs
    )
    return coeffs.at[..., operator_idx].set(on_off_sequence)


# TODO: maybe do this per timepoint w relaxed group l0 instead
def try_replacing_operator(
    objective: BinaryObjective, operator_idx: int, coeffs: Array
) -> tuple[Array, Array]:

    operator_is_on = coeffs[..., operator_idx] == 1
    coeffs_operator_off = coeffs.at[..., operator_idx].set(0.0)

    def replace_operator(alternative_idx):
        alternative_operator_coeffs = jnp.where(
            operator_is_on, 1, coeffs[..., alternative_idx]
        )

        return coeffs_operator_off.at[..., alternative_idx].set(
            alternative_operator_coeffs
        )

    coeffs_operator_replaced = vmap(replace_operator)(
        jnp.arange(objective.dynamics_model.num_operators)
    )

    best_costs = vmap(objective.batch)(coeffs_operator_replaced)
    best_operator_idx = best_costs.argmin()

    return coeffs_operator_replaced[best_operator_idx]
