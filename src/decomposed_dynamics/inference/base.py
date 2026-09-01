from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxopt.prox import prox_non_negative_lasso


class NoObsInferenceHyperparams(eqx.Module):
    prox: Callable = prox_non_negative_lasso
    dynamics_loss_coeff: Array
    l1_coeff: Array
    l1_reweight_coeff: Array
    smooth_coeff: Array
    max_iter: int
    tol: float

    def __init__(
        self,
        prox: Callable = prox_non_negative_lasso,
        dynamics_loss_coeff: float = 1.0,
        l1_coeff: float = 0.25,
        l1_reweight_coeff: float = 200,
        smooth_coeff: float = 0.4,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.prox = prox
        self.dynamics_loss_coeff = jnp.array(dynamics_loss_coeff)
        self.l1_coeff = jnp.array(l1_coeff)
        self.l1_reweight_coeff = jnp.array(l1_reweight_coeff)
        self.smooth_coeff = jnp.array(smooth_coeff)
        self.max_iter = max_iter
        self.tol = tol


class InferenceHyperparams(NoObsInferenceHyperparams):
    pass
