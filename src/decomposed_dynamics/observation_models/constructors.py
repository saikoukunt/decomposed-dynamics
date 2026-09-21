from decomposed_dynamics.observation_models.base import ObservationModel
from decomposed_dynamics.observation_models.noise_models import (
    GaussianNoiseModel,
    PoissonNoiseModel,
)
from decomposed_dynamics.observation_models.rate_models import (
    LinearRateModel,
    SoftplusLinearRateModel,
)


def LinearGaussianObservations(num_observations, num_latents, key):
    rate_model = LinearRateModel(num_observations, num_latents, key)
    return ObservationModel(rate_model, GaussianNoiseModel())


def SoftplusPoissonObservations(num_observations, num_latents, key):
    rate_model = SoftplusLinearRateModel(num_observations, num_latents, key)
    return ObservationModel(rate_model, PoissonNoiseModel())
