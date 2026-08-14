from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
from jax import Array


class RateModel(eqx.Module):
    def __init__(self, num_observations, num_latents):
        self.num_observations = num_observations
        self.num_latents = num_latents

    @abstractmethod
    def initialize_params(self, key: Array, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_rates(self, x: Array) -> Array:
        raise NotImplementedError


class NoiseModel(ABC):
    @abstractmethod
    def neg_log_likelihood(self, rates: Array, observations: Array) -> Array:
        raise NotImplementedError


class ObservationModel(eqx.Module):
    num_observations: int
    num_latents: int
    rate_model: RateModel
    noise_model: NoiseModel

    def __init__(
        self, rate_model: RateModel, noise_model: NoiseModel, key: Array, **init_kwargs
    ):
        self.num_observations = rate_model.num_observations
        self.num_latents = rate_model.num_latents
        self.rate_model = rate_model
        self.noise_model = noise_model
        self.initialize_params(key, **init_kwargs)

    def initialize_params(self, key: Array, **kwargs) -> Self:
        self.rate_model.initialize_params()

    def predict_rates(self, x: Array) -> Array:
        return self.rate_model.predict_rates(x)

    def neg_log_likelihood(self, rates: Array, observations: Array) -> Array:
        return self.noise_model.neg_log_likelihood(rates, observations)

    def apply_prox(self) -> Self:
        rate_model = self.rate_model.apply_prox()
        return eqx.tree_at(lambda model: model.rate_model, self, rate_model)
