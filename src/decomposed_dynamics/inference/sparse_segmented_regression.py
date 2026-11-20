from functools import partial
from typing import Callable, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax, vmap
from jax.ops import segment_sum
from jaxopt import ProximalGradient
from optax import l2_loss

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import (
    InferenceHyperparams,
    NoObsInferenceBackend,
    Penalty,
)
from decomposed_dynamics.loss_functions import (
    normalized_dynamics_reconstruction_loss,
    null_prediction_variance,
)


class SparseSegmentedHyperparams(InferenceHyperparams):
    l0: Penalty = Penalty(0.1)
    group: Penalty = Penalty(0.5)
    switch_penalty: Penalty = Penalty(0.4)
    smooth_coeff: Penalty = Penalty(0.4)
    num_sweeps: int = eqx.field(static=True, default=10)
    min_segment_length: int = eqx.field(static=True, default=3)


class SparseSegmentedNoObsInference(NoObsInferenceBackend):
    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> SparseSegmentedHyperparams:
        return SparseSegmentedHyperparams(**kwargs)

    @override
    @classmethod
    @eqx.filter_jit
    def infer_batch(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: SparseSegmentedHyperparams,
    ) -> Array:

        support_coeffs = cls.estimate_support(
            dynamics_model, states, targets, hyperparams
        )
        coeffs = cls.compute_smoothed_coeffs(
            dynamics_model, support_coeffs, states, targets, hyperparams
        )

        return coeffs

    @override
    @staticmethod
    def infer_trial(
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: SparseSegmentedHyperparams,
    ) -> Array:
        raise NotImplementedError("the group penalty couples trials; use infer_batch")

    @classmethod
    @eqx.filter_jit
    def estimate_support(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        states: Array,
        targets: Array,
        hyperparams: SparseSegmentedHyperparams,
    ) -> Array:

        objective = SparsePELTObjective(dynamics_model, states, targets, hyperparams)

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
            jnp.zeros((states.shape[0], states.shape[1], dynamics_model.num_operators)),
        )

    @classmethod
    @eqx.filter_jit
    def compute_smoothed_coeffs(
        cls,
        dynamics_model: DecomposedDynamicsModel,
        support_coeffs: Array,
        states: Array,
        targets: Array,
        hyperparams: SparseSegmentedHyperparams,
    ) -> Array:

        def smooth_trial(trial):
            support_coeffs, states, targets = trial
            support_mask = support_coeffs > 0
            solver = ProximalGradient(
                partial(
                    smoothed_coeffs_loss,
                    dynamics_model=dynamics_model,
                    per_operator_predictions=vmap(
                        dynamics_model.compute_operator_predictions
                    )(states),
                    states=states,
                    targets=targets,
                    support_mask=support_mask,
                    smooth_coeff=hyperparams.smooth_coeff,
                ),
                project_onto_support,
                maxiter=hyperparams.max_iter,
                tol=hyperparams.tol,
            )
            coeffs, _ = solver.run(support_coeffs, hyperparams_prox=support_mask)
            return coeffs

        return lax.map(smooth_trial, (support_coeffs, states, targets))


def project_onto_support(
    x: Array, support_mask: Array, scaling: float = 1.0
) -> Array:
    return jnp.maximum(x, 0.0) * support_mask


def smoothed_coeffs_loss(
    coeffs: Array,
    dynamics_model: DecomposedDynamicsModel,
    per_operator_predictions: Array,
    states: Array,
    targets: Array,
    support_mask: Array,
    smooth_coeff: Array,
) -> Array:

    recon_loss = normalized_dynamics_reconstruction_loss(
        dynamics_model, states, coeffs, targets, per_operator_predictions
    ).sum()
    is_on_at_both_timepoints = support_mask[1:] & support_mask[:-1]
    smoothness_loss = (
        is_on_at_both_timepoints * l2_loss(coeffs[1:], coeffs[:-1])
    ).sum()

    return recon_loss + smooth_coeff * smoothness_loss


class SparsePELTObjective(eqx.Module):
    dynamics_model: DecomposedDynamicsModel
    states: Array
    targets: Array
    per_operator_predictions: Array
    null_prediction_variance: Array
    hyperparams: SparseSegmentedHyperparams

    def __init__(self, dynamics_model, states, targets, hyperparams):
        self.dynamics_model = dynamics_model
        self.states = states
        self.targets = targets
        self.per_operator_predictions = vmap(
            vmap(dynamics_model.compute_operator_predictions)
        )(states)
        self.null_prediction_variance = null_prediction_variance(
            dynamics_model, states, targets, self.per_operator_predictions
        )
        self.hyperparams = hyperparams

    def batch(self, coeffs: Array) -> Array:
        """Objective each PELT update minimizes per operator, plus the group penalty."""
        num_timepoints = coeffs.shape[0] * coeffs.shape[1]
        recon_loss = normalized_dynamics_reconstruction_loss(
            self.dynamics_model,
            self.states,
            coeffs,
            self.targets,
            self.per_operator_predictions,
        ).sum()
        l0 = self.hyperparams.l0 * (coeffs > 0).sum()
        num_changepoints = (jnp.diff(coeffs, axis=1) != 0).sum()
        group_l0 = coeffs.any(axis=(0, 1)).sum()

        return (
            recon_loss + l0 + self.hyperparams.switch_penalty * num_changepoints
        ) / num_timepoints + self.hyperparams.group * group_l0

    def compute_off_segment_cost(
        self, start: int, end: int, residual_cumulative_sum_of_squares: Array
    ) -> tuple[Array, Array]:

        off_segment_cost = 0.5 * (
            residual_cumulative_sum_of_squares[end]
            - residual_cumulative_sum_of_squares[start]
        )
        return off_segment_cost, jnp.zeros_like(off_segment_cost)

    def compute_on_segment_cost(
        self,
        start: int,
        end: int,
        residual_cumulative_sum_of_squares: Array,
        residual_prediction_cumulative_sum_of_products: Array,
        prediction_cumulative_sum_of_squares: Array,
    ) -> tuple[Array, Array]:

        l0 = self.hyperparams.l0 * (end - start)
        off_segment_cost, _ = self.compute_off_segment_cost(
            start, end, residual_cumulative_sum_of_squares
        )

        segment_residual_prediction_sum_of_products = (
            residual_prediction_cumulative_sum_of_products[end]
            - residual_prediction_cumulative_sum_of_products[start]
        )
        segment_prediction_sum_of_squares = (
            prediction_cumulative_sum_of_squares[end]
            - prediction_cumulative_sum_of_squares[start]
        )
        segment_coeff = jnp.maximum(
            segment_residual_prediction_sum_of_products
            / segment_prediction_sum_of_squares,
            0.0,
        )
        reconstruction_loss_reduction = (
            segment_residual_prediction_sum_of_products * segment_coeff / 2
        )

        return l0 + off_segment_cost - reconstruction_loss_reduction, segment_coeff


def update_one_operator_coeffs(
    objective: SparsePELTObjective,
    operator_idx: int,
    coeffs: Array,
) -> Array:

    coeffs_operator_off = coeffs.at[..., operator_idx].set(0.0)
    predictions_with_operator_off = objective.dynamics_model.combine_operator_predictions(
        objective.states, coeffs_operator_off, objective.per_operator_predictions
    )

    residuals = objective.targets - predictions_with_operator_off
    operator_prediction = (
        objective.dynamics_model.combine_operator_predictions(
            objective.states,
            coeffs_operator_off.at[..., operator_idx].set(1.0),
            objective.per_operator_predictions,
        )
        - predictions_with_operator_off
    )

    def normalized_cumulative_sum_from_zero(x):
        return jnp.pad(
            jnp.cumsum(x / objective.null_prediction_variance, axis=1),
            ((0, 0), (1, 0)),
        )

    residual_cumulative_sum_of_squares = normalized_cumulative_sum_from_zero(
        (residuals**2).sum(-1)
    )
    residual_prediction_cumulative_sum_of_products = (
        normalized_cumulative_sum_from_zero((operator_prediction * residuals).sum(-1))
    )
    prediction_cumulative_sum_of_squares = normalized_cumulative_sum_from_zero(
        (operator_prediction**2).sum(-1)
    )

    _, (_, _, operator_coeffs) = vmap(on_off_PELT, (None, 0, 0, 0, None, None))(
        objective,
        residual_cumulative_sum_of_squares,
        residual_prediction_cumulative_sum_of_products,
        prediction_cumulative_sum_of_squares,
        objective.states.shape[1],
        objective.hyperparams.switch_penalty,
    )

    coeffs = coeffs.at[..., operator_idx].set(operator_coeffs)

    return coeffs


def try_replacing_operator(
    objective: SparsePELTObjective, operator_idx: int, coeffs: Array
) -> Array:
    """Let another operator take over this operator's on-segments, refitting its
    per-segment nonnegative coefficients, and keep whichever choice (including no
    change) has the lowest batch objective."""

    def replace_operator(alternative_idx):
        coeffs_both_off = (
            coeffs.at[..., operator_idx].set(0.0).at[..., alternative_idx].set(0.0)
        )
        predictions_both_off = objective.dynamics_model.combine_operator_predictions(
            objective.states, coeffs_both_off, objective.per_operator_predictions
        )
        residuals = objective.targets - predictions_both_off
        alternative_prediction = (
            objective.dynamics_model.combine_operator_predictions(
                objective.states,
                coeffs_both_off.at[..., alternative_idx].set(1.0),
                objective.per_operator_predictions,
            )
            - predictions_both_off
        )

        # segments of the merged on-path split wherever either operator's
        # coefficient changes; ids are unique across trials
        is_on = (coeffs[..., operator_idx] > 0) | (coeffs[..., alternative_idx] > 0)
        is_segment_start = jnp.pad(
            (jnp.diff(coeffs[..., operator_idx], axis=1) != 0)
            | (jnp.diff(coeffs[..., alternative_idx], axis=1) != 0),
            ((0, 0), (1, 0)),
            constant_values=True,
        )
        segment_ids = jnp.cumsum(is_segment_start.ravel()) - 1

        def normalized_sum_per_segment(x):
            return segment_sum(
                (x / objective.null_prediction_variance).ravel(),
                segment_ids,
                num_segments=segment_ids.size,
            )

        segment_coeffs = jnp.maximum(
            normalized_sum_per_segment((alternative_prediction * residuals).sum(-1))
            / normalized_sum_per_segment((alternative_prediction**2).sum(-1)),
            0.0,
        )
        alternative_coeffs = jnp.where(
            is_on, segment_coeffs[segment_ids].reshape(is_on.shape), 0.0
        )

        return coeffs_both_off.at[..., alternative_idx].set(alternative_coeffs)

    coeffs_operator_replaced = vmap(replace_operator)(
        jnp.arange(objective.dynamics_model.num_operators)
    )

    # replacing with itself only refits, so compare against the unchanged coeffs too
    best_replaced = coeffs_operator_replaced[
        vmap(objective.batch)(coeffs_operator_replaced).argmin()
    ]
    return jnp.where(
        objective.batch(best_replaced) < objective.batch(coeffs), best_replaced, coeffs
    )


def on_off_PELT(
    objective: SparsePELTObjective,
    residual_cumulative_sum_of_squares: Array,
    residual_prediction_cumulative_sum_of_products: Array,
    prediction_cumulative_sum_of_squares: Array,
    num_timepoints: int,
    switch_penalty: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:
    start_candidates = jnp.arange(num_timepoints + 1)
    min_segment_length = objective.hyperparams.min_segment_length

    def extend_partition(carry, t):
        best_partition_cost_until, first_pruned_step = carry

        off_segment_costs, off_segment_coeffs = vmap(objective.compute_off_segment_cost, (0, None, None))(
            start_candidates, t, residual_cumulative_sum_of_squares
        )
        on_segment_costs, on_segment_coeffs = vmap(
            objective.compute_on_segment_cost, (0, None, None, None, None)
        )(
            start_candidates,
            t,
            residual_cumulative_sum_of_squares,
            residual_prediction_cumulative_sum_of_products,
            prediction_cumulative_sum_of_squares,
        )
        segment_costs = jnp.where(
            off_segment_costs <= on_segment_costs, off_segment_costs, on_segment_costs
        )
        segment_state = jnp.where(
            off_segment_costs <= on_segment_costs,
            jnp.zeros_like(off_segment_costs),
            jnp.ones_like(on_segment_costs),
        )
        segment_coeff = jnp.where(
            off_segment_costs <= on_segment_costs, off_segment_coeffs, on_segment_coeffs
        )

        # pruning s at step u only rules s out from step u + min_segment_length on,
        # since the dominating partition's last segment [u, t) must be long enough
        is_start_candidate = (start_candidates <= t - min_segment_length) & (
            first_pruned_step > t - min_segment_length
        )
        new_partition_costs = jnp.where(
            is_start_candidate,
            best_partition_cost_until + segment_costs,
            jnp.inf,
        )
        changepoint = new_partition_costs.argmin()
        best_cost = new_partition_costs[changepoint] + switch_penalty
        first_pruned_step = jnp.where(
            is_start_candidate & (new_partition_costs > best_cost),
            jnp.minimum(first_pruned_step, t),
            first_pruned_step,
        )

        return (
            best_partition_cost_until.at[t].set(best_cost),
            first_pruned_step,
        ), (changepoint, segment_state[changepoint], segment_coeff[changepoint])

    # best_partition_costs[t] is the best cost for partition that ends before timepoint[t]
    best_partition_costs = jnp.full(num_timepoints + 1, jnp.inf)
    best_partition_costs = best_partition_costs.at[0].set(-switch_penalty)
    never_pruned = jnp.full(num_timepoints + 1, num_timepoints + 1)

    (
        (best_partition_costs, _),
        (best_prev_changepoint, best_segment_state, best_segment_coeff),
    ) = lax.scan(
        extend_partition,
        (best_partition_costs, never_pruned),
        jnp.arange(1, num_timepoints + 1),
    )
    best_prev_changepoint = jnp.append(0, best_prev_changepoint)
    best_segment_state = jnp.append(0, best_segment_state)
    best_segment_coeff = jnp.append(0, best_segment_coeff)

    def trace_segments_backwards(segment_end, t):
        is_segment_start = t == best_prev_changepoint[segment_end]
        return jnp.where(is_segment_start, t, segment_end), (
            is_segment_start,
            best_segment_state[segment_end],
            best_segment_coeff[segment_end],
        )

    _, segment_starts = lax.scan(
        trace_segments_backwards,
        num_timepoints,
        jnp.arange(num_timepoints),
        reverse=True,
    )

    return best_partition_costs[-1], segment_starts
