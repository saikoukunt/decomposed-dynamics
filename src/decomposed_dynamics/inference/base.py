from dataclasses import dataclass
from typing import Callable

from jaxopt.prox import prox_non_negative_lasso


@dataclass(frozen=True)
class NoObsInferenceHyperparams:
    prox: Callable = prox_non_negative_lasso
    dynamics_loss_coeff: float = 1.0
    l1_coeff: float = 0.25
    l1_reweight_coeff: float = 200
    smooth_coeff: float = 0.4
    max_iter: int = 1000
    tol: float = 1e-4


@dataclass(frozen=True)
class InferenceHyperparams(NoObsInferenceHyperparams):
    dynamics_loss_coeff: float = 0.5
