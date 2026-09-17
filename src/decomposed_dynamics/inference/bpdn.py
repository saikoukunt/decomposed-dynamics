import functools
from dataclasses import dataclass
from typing import Callable, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference.base import (
    InferenceBackend,
    InferenceHyperparams,
    NoObsInferenceBackend,
    _prox_coeffs_only,
    solve_reweighted,
)
from decomposed_dynamics.loss_functions import (
    coeff_smoothness_loss,
    normalized_dynamics_reconstruction_loss,
)
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.proximal_operators import _reweight_prox_hyperparams


class BPDNDFHyperparams(InferenceHyperparams):
    prox_hyperparams: Array = eqx.field(default=0.25, converter=jnp.array)
    prox_reweight_coeff: Array = eqx.field(default=200.0, converter=jnp.array)
    smooth_coeff: Array = eqx.field(default=0.4, converter=jnp.array)


@dataclass(frozen=True)
class BPDNDFNoObsInference(NoObsInferenceBackend):
    prox: Callable = prox_non_negative_lasso

    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> BPDNDFHyperparams:
        return BPDNDFHyperparams(**kwargs)

    @override
    @eqx.filter_jit
    def infer_trial(
        self,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        states: Array,
        targets: Array,
        hyperparams: BPDNDFHyperparams,
    ) -> Array:

        solver = ProximalGradient(
            functools.partial(self.bpdn_df_smooth_loss, dynamics_model=dynamics_model),
            self.prox,
            maxiter=hyperparams.max_iter,
            tol=hyperparams.tol,
        )
        infer_one_timestep = functools.partial(
            self.infer_one_timestep,
            dynamics_model=dynamics_model,
            compute_per_operator_predictions=compute_per_operator_predictions,
            solver=solver,
            hyperparams=hyperparams,
        )
        _, coeffs = lax.scan(
            infer_one_timestep,
            (jnp.zeros(dynamics_model.num_operators), jnp.bool_(True)),
            (states, targets),
        )

        return coeffs

    @eqx.filter_jit
    def infer_one_timestep(
        self,
        carry: tuple[Array, Array],
        xs: tuple[Array, Array],
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        solver: ProximalGradient,
        hyperparams: BPDNDFHyperparams,
    ) -> tuple[tuple[Array, Array], Array]:

        prev_coeffs, is_first_timestep = carry
        states, targets = xs
        per_operator_predictions = compute_per_operator_predictions(states)
        smooth_coeff = jnp.where(is_first_timestep, 0.0, hyperparams.smooth_coeff)

        coeffs = solve_reweighted(
            solver,
            jnp.zeros(dynamics_model.num_operators),
            hyperparams.prox_hyperparams,
            hyperparams.prox_reweight_coeff,
            per_operator_predictions=per_operator_predictions,
            targets=targets,
            prev_coeffs=prev_coeffs,
            smooth_coeff=smooth_coeff,
        )

        return (coeffs, jnp.bool_(False)), coeffs

    @staticmethod
    @eqx.filter_jit
    def bpdn_df_smooth_loss(
        coeffs: Array,
        dynamics_model: DecomposedDynamicsModel,
        per_operator_predictions: Array,
        targets: Array,
        prev_coeffs: Array,
        smooth_coeff: Array,
    ) -> Array:

        recon_loss = normalized_dynamics_reconstruction_loss(
            dynamics_model, coeffs, targets, per_operator_predictions
        )
        smoothness_loss = coeff_smoothness_loss(coeffs, prev_coeffs)

        return recon_loss + smooth_coeff * smoothness_loss


@dataclass(frozen=True)
class BPDNDFInference(InferenceBackend):
    prox: Callable = prox_non_negative_lasso

    @override
    @staticmethod
    def initialize_hyperparams(**kwargs) -> BPDNDFHyperparams:
        return BPDNDFHyperparams(**kwargs)

    @override
    @eqx.filter_jit
    def infer_trial(
        self,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        observations: Array,
        hyperparams: BPDNDFHyperparams,
    ) -> Array:
        solver = ProximalGradient(
            functools.partial(
                self.bpdn_df_smooth_loss,
                observation_model=observation_model,
                dynamics_model=dynamics_model,
            ),
            _prox_coeffs_only(self.prox, dynamics_model.state_dim),
            maxiter=hyperparams.max_iter,
            tol=hyperparams.tol,
        )

        infer_one_timestep = functools.partial(
            self.infer_one_timestep,
            observation_model=observation_model,
            dynamics_model=dynamics_model,
            compute_per_operator_predictions=compute_per_operator_predictions,
            solver=solver,
            hyperparams=hyperparams,
        )
        _, state = lax.scan(
            infer_one_timestep,
            (
                jnp.zeros(dynamics_model.state_dim + dynamics_model.num_operators),
                jnp.bool_(True),
            ),
            observations,
        )

        return state

    @eqx.filter_jit
    def infer_one_timestep(
        self,
        carry: tuple[Array, Array],
        observations: Array,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        compute_per_operator_predictions: Callable,
        solver: ProximalGradient,
        hyperparams: BPDNDFHyperparams,
    ) -> tuple[tuple[Array, Array], Array]:

        state, is_first = carry
        prev_latents = state[: dynamics_model.state_dim]
        prev_coeffs = state[dynamics_model.state_dim :]

        per_operator_predictions = compute_per_operator_predictions(prev_latents)
        smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)

        state = solve_reweighted(
            solver,
            jnp.zeros(dynamics_model.state_dim + dynamics_model.num_operators),
            hyperparams.prox_hyperparams,
            hyperparams.prox_reweight_coeff,
            reweight=lambda state, prox_hyperparams, reweight_coeff: (
                _reweight_prox_hyperparams(
                    state[..., dynamics_model.state_dim :],
                    prox_hyperparams,
                    reweight_coeff,
                )
            ),
            per_operator_predictions=per_operator_predictions,
            observations=observations,
            prev_coeffs=prev_coeffs,
            prev_latents=prev_latents,
            dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
            smooth_coeff=smooth_coeff,
        )

        return (state, jnp.bool_(False)), state

    @staticmethod
    @eqx.filter_jit
    def bpdn_df_smooth_loss(
        state: Array,
        observation_model: ObservationModel,
        dynamics_model: DecomposedDynamicsModel,
        per_operator_predictions: Array,
        observations: Array,
        prev_coeffs: Array,
        prev_latents: Array,
        dynamics_loss_coeff: Array,
        smooth_coeff: Array,
    ) -> Array:
        latents = state[: dynamics_model.state_dim]
        coeffs = state[dynamics_model.state_dim :]

        predicted_rates = observation_model.predict_rates(latents)
        data_nll = observation_model.neg_log_likelihood(predicted_rates, observations)

        dynamics_recon_loss = (
            dynamics_loss_coeff
            * normalized_dynamics_reconstruction_loss(
                dynamics_model, coeffs, latents, per_operator_predictions
            )
        )
        smooth_loss = smooth_coeff * coeff_smoothness_loss(coeffs, prev_coeffs)

        return data_nll + dynamics_recon_loss + smooth_loss
