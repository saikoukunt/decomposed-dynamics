import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, lax


def create_random_dynamics(
    num_latents: int, num_motifs: int, key: Array, correlation_tol: float = 0.1
):
    F = np.zeros((num_motifs, num_latents, num_latents))

    for i in range(num_motifs):
        repeat = True
        while repeat:
            key, subkey = jr.split(key)
            F[i] = sample_orthogonal_matrix(num_latents, subkey)
            repeat = False

            for j in range(i):
                correlation = np.abs(np.corrcoef(F[i].flatten(), F[j].flatten()))[0, 1]
                if correlation > correlation_tol:
                    repeat = True
                    break

    return jnp.array(F)


def sample_orthogonal_matrix(num_latents: int, key: Array):
    Q, R = lax.linalg.qr(jr.normal(key, (num_latents, num_latents)))
    f = Q @ jnp.diag(jnp.sign(jnp.diag(R)))

    return f


def generate_switching_c(
    num_motifs: int,
    num_timepoints: int,
    key: Array,
    min_switch_time: int = 100,
    max_extra_switch_time: int = 300,
):
    C = np.zeros((num_timepoints - 1, num_motifs))
    t = 0
    while t < num_timepoints - 1:
        key, length_key, motif_key = jr.split(key, 3)

        active_length = (
            min_switch_time + jr.randint(length_key, 1, 0, max_extra_switch_time)[0]
        )
        end_time = min(t + active_length, num_timepoints - 1)
        active_motif = jr.randint(motif_key, 1, 0, num_motifs + 1)[0]

        if active_motif != 0:
            C[t:end_time, active_motif - 1] = 1

        t = end_time

    return jnp.array(C)
