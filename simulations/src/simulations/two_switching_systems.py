import jax.numpy as jnp
import jax.random as jr
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
    seed: int = 0,
):
    total_latents = sum(num_latents)
    total_motifs = sum(num_motifs)

    keys = jr.split(jr.key(seed), 3)

    F = np.zeros((total_motifs, total_latents, total_latents))
    F[: num_motifs[0], : num_latents[0], : num_latents[0]] = create_random_dynamics(
        num_latents[0], num_motifs[0], keys[0]
    )
    F[num_motifs[0] :, num_latents[0] :, num_latents[0] :] = create_random_dynamics(
        num_latents[1], num_motifs[1], keys[1]
    )

    X, C = _simulate_no_obs_state(
        F,
        num_trials,
        num_timepoints,
        num_latents,
        num_motifs,
        keys[2],
        min_switch_time,
        max_extra_switch_time,
    )
    X = {i: X[i] for i in range(num_trials)}
    C = {i: C[i] for i in range(num_trials)}

    return X, C, jnp.array(F)


def _simulate_no_obs_state(
    F: Array,
    num_trials: int,
    num_timepoints: int,
    num_latents: npt.NDArray,
    num_motifs: npt.NDArray,
    key: Array,
    min_switch_time: int = 100,
    max_extra_switch_time: int = 300,
):
    total_latents = sum(num_latents)
    total_motifs = sum(num_motifs)

    X = np.zeros((num_trials, num_timepoints, total_latents))
    C = np.zeros((num_trials, num_timepoints, total_motifs))

    for trial in range(num_trials):
        key, c1_key, c2_key, x_key = jr.split(key, 4)

        C[trial, :, : num_motifs[0]] = generate_switching_c(
            num_motifs[0],
            num_timepoints,
            c1_key,
            min_switch_time,
            max_extra_switch_time,
        )
        C[trial, :, num_motifs[0] :] = generate_switching_c(
            num_motifs[1],
            num_timepoints,
            c2_key,
            min_switch_time,
            max_extra_switch_time,
        )
        X[trial] = _simulate_latent_trajectory(
            C[trial], F, num_latents, num_motifs, num_timepoints, x_key
        )

    return X, C


def _simulate_latent_trajectory(
    C: Array,
    F: Array,
    num_latents: list[int],
    num_motifs: list[int],
    num_timepoints: int,
    key: Array,
):
    total_latents = sum(num_latents)

    F_t = np.zeros((total_latents, total_latents))
    X = np.zeros((num_timepoints, total_latents))

    key, subkey = jr.split(key)
    X[0, :] = jr.normal(subkey, (total_latents))
    X[0, : num_latents[0]] /= np.linalg.norm(X[0, : num_latents[0]])
    X[0, num_latents[0] :] /= np.linalg.norm(X[0, num_latents[0] :])

    for t in range(1, num_timepoints):
        F_t = np.einsum("k, kij -> ij", C[t, :], F)

        key, *subkeys = jr.split(key, 3)
        if (X[t - 1, : num_latents[0]] == 0).all() and (
            C[t, : num_motifs[0]] != 0
        ).any():
            X[t - 1, : num_latents[0]] = jr.normal(subkeys[0], (num_latents[0]))
            X[t - 1, : num_latents[0]] /= np.linalg.norm(X[t - 1, : num_latents[0]])

        if (X[t - 1, num_latents[0] :] == 0).all() and (
            C[t, num_motifs[0] :] != 0
        ).any():
            X[t - 1, num_latents[0] :] = jr.normal(subkeys[1], (num_latents[0]))
            X[t - 1, num_latents[0] :] /= np.linalg.norm(X[t - 1, num_latents[0] :])

        X[t, :] = np.einsum("ij, j -> i", F_t, X[t - 1, :])

    return X
