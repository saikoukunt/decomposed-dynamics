import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng


def create_random_dynamics(
    num_latents: int, num_motifs: int, correlation_tol: float = 0.02
):
    F = np.zeros((num_motifs, num_latents, num_latents))
    for i in range(num_motifs):
        repeat = True
        while repeat:
            F[i] = sample_orthogonal_matrix(num_latents)
            repeat = False

            for j in range(i):
                correlation = np.abs(np.corrcoef(F[i].flatten(), F[j].flatten()))
                if correlation > correlation_tol:
                    repeat = True
                    break

    return jnp.array(F)


def sample_orthogonal_matrix(num_latents: int):
    rng = default_rng(42)
    Q, R = np.linalg.qr(rng.standard_normal((num_latents, num_latents)))
    f = Q @ np.diag(np.sign(np.diag(R)))

    return f


def generate_switching_c(
    num_motifs: int,
    num_timepoints: int,
    min_switch_time: int = 100,
    max_extra_switch_time: int = 300,
):
    rng = default_rng(42)
    C = np.zeros((num_motifs, num_timepoints))
    t = 0
    while t < num_timepoints:
        active_length = min_switch_time + rng.integers(0, max_extra_switch_time)
        end_time = min(t + active_length, num_timepoints)
        active_motif = rng.integers(0, num_motifs, endpoint=True)

        if active_motif != 0:
            C[active_motif, t:end_time] = 1

        t = end_time

    return jnp.array(C)
