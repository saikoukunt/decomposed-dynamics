from dataclasses import dataclass


@dataclass(frozen=True)
class NoObsInferenceHyperparams:
    dynamics_loss_coeff: float = 1.0
    l1_coeff: float = 0.25
    l1_reweight_coeff: float = 200
    smooth_coeff: float = 0.4
    max_iter: int = 1000
    tol: float = 1e-4
    solver: str = "yall1"
    spgl1_sigma: float = None
    spgl1_max_iter: int = 200
    yall1_rho: int = 1e-2



@dataclass(frozen=True)
class InferenceHyperparams(NoObsInferenceHyperparams):
    dynamics_loss_coeff: float = 0.5
