from abc import ABC, abstractmethod

from jax import Array


class ObservationParams(ABC):
    pass


class ObservationModel:
    def __init__(self, rate_model, noise_model):
        self.num_observations = rate_model.num_observations
        self.num_latents = rate_model.num_latents
        self.rate_model = rate_model
        self.noise_model = noise_model

    def initialize_params(self, key: Array, **kwargs) -> ObservationParams:
        return self.rate_model.initialize_params()

    def predict_rates(self, params: ObservationParams, x: Array) -> Array:
        return self.rate_model.predict_rates(params, x)

    def neg_log_likelihood(self, rates: Array, observations: Array) -> Array:
        return self.noise_model.neg_log_likelihood(rates, observations)


class RateModel(ABC):
    def __init__(self, num_observations, num_latents):
        self.num_observations = num_observations
        self.num_latents = num_latents

    @abstractmethod
    def initialize_params(self, key: Array, **kwargs) -> ObservationParams:
        raise NotImplementedError

    @abstractmethod
    def predict_rates(self, params: ObservationParams, x: Array) -> Array:
        raise NotImplementedError


class NoiseModel(ABC):
    @abstractmethod
    def neg_log_likelihood(self, rates: Array, observations: Array) -> Array:
        raise NotImplementedError
