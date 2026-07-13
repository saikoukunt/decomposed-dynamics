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
    rate_model = LinearRateModel(num_observations, num_latents)
    noise_model = GaussianNoiseModel()

    observation_model = ObservationModel(rate_model, noise_model)
    params = observation_model.initialize_params(key)

    return observation_model, params


def SoftplusPoissonObservations(num_observations, num_latents, key):
    rate_model = SoftplusLinearRateModel(num_observations, num_latents)
    noise_model = PoissonNoiseModel()

    observation_model = ObservationModel(rate_model, noise_model)
    params = observation_model.initialize_params(key)

    return observation_model, params
