import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from .utils import create_random_dynamics, generate_switching_c


def simulate_two_subsystems_no_obs(
    num_timepoints: int,
    num_latents: list[int],
    num_motifs: list[int],
    num_trials: bool = True,
    min_switch_time: int = 100,
    max_extra_switch_time: int = 300,
):
    total_latents = sum(num_latents)
    total_motifs = sum(num_motifs)

    F = np.zeros((total_motifs, total_latents, total_latents))
    F[: num_motifs[0], : num_latents[0], : num_latents[0]] = create_random_dynamics(
        num_latents[0], num_motifs[0]
    )
    F[num_motifs[0] :, num_latents[0] :, num_latents[0] :] = create_random_dynamics(
        num_latents[1], num_motifs[1]
    )

    X, C = _simulate_no_obs_state(
        F,
        num_trials,
        num_timepoints,
        num_latents,
        num_motifs,
        min_switch_time,
        max_extra_switch_time,
    )

    return jnp.array(X), jnp.array(C), jnp.array(F)


def _simulate_no_obs_state(
    F: Array,
    num_trials: int,
    num_timepoints: int,
    num_latents: npt.NDArray,
    num_motifs: npt.NDArray,
    min_switch_time: int = 100,
    max_extra_switch_time: int = 300,
):
    total_latents = sum(num_latents)
    total_motifs = sum(num_motifs)

    X = np.zeros((num_trials, total_latents, num_timepoints))
    C = np.zeros((num_trials, total_motifs, num_timepoints - 1))

    for trial in range(num_trials):
        C[trial][: num_motifs[0]] = generate_switching_c(
            num_motifs[0],
            num_timepoints,
            min_switch_time,
            max_extra_switch_time,
            seed=trial,
        )
        C[trial][num_motifs[0] :] = generate_switching_c(
            num_motifs[1],
            num_timepoints,
            min_switch_time,
            max_extra_switch_time,
            seed=trial + num_trials,
        )
        X[trial] = _simulate_latent_trajectory(
            C[trial], F, num_latents, num_motifs, num_timepoints
        )

    return X, C


def _simulate_latent_trajectory(
    C: Array,
    F: Array,
    num_latents: list[int],
    num_motifs: list[int],
    num_timepoints: int,
):
    rng = np.random.default_rng(42)
    total_latents = sum(num_latents)

    F_t = np.zeros((total_latents, total_latents))
    X = np.zeros((total_latents, num_timepoints))

    X[:, 0] = rng.standard_normal((total_latents))
    X[: num_latents[0], 0] /= np.linalg.norm(X[: num_latents[0], 0])
    X[num_latents[0] :, 0] /= np.linalg.norm(X[num_latents[0] :, 0])

    for t in range(1, num_timepoints):
        F_t = np.einsum("k, kij -> ij", C[:, t - 1], F)

        if (X[: num_latents[0], t - 1] == 0).all() and (
            C[: num_motifs[0], t - 1] != 0
        ).any():
            X[: num_latents[0], t - 1] = rng.standard_normal((num_latents[0]))
            X[: num_latents[0], t - 1] /= np.linalg.norm(X[: num_latents[0], t - 1])

        if (X[num_latents[0] :, t - 1] == 0).all() and (
            C[num_motifs[0] :, t - 1] != 0
        ).any():
            X[num_latents[0] :, t - 1] = rng.standard_normal((num_latents[0]))
            X[num_latents[0] :, t - 1] /= np.linalg.norm(X[num_latents[0] :, t - 1])

        X[:, t] = np.einsum("ij, j -> i", F_t, X[:, t - 1])

    return X
