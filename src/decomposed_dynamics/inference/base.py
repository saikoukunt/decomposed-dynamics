from dataclasses import dataclass


@dataclass(frozen=True)
class NoObsInferenceHyperparams:
    dynamics_loss_coeff: float = 1.0
    l1_coeff: float = 0.25
    l1_reweight_coeff: float = 200
    smooth_coeff: float = 0.4
    max_iter: int = 1000
    tol: float = 1e-4


@dataclass(frozen=True)
class InferenceHyperparams(NoObsInferenceHyperparams):
    dynamics_loss_coeff: float = 0.5
