import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, jit


def extract_snippets(
    trial_data: dict, num_snippets: int, samples_per_snippet: int, seed: int
) -> tuple[Array, Array]:
    num_trials = len(trial_data)
    num_observations = trial_data[0].shape[0]
    trial_length = min(trial_data[int(key)].shape[1] for key in trial_data)

    rng = np.random.default_rng(seed)

    snippet_length = min(samples_per_snippet, trial_length)
    snippets = np.zeros((num_snippets, num_observations, snippet_length))
    snippet_times = np.zeros((num_snippets, 2), dtype=np.int32)

    if num_snippets == num_trials:
        trial_inds = np.arange(num_snippets)
    else:
        trial_inds = rng.choice(num_trials, num_snippets)

    for i, trial_ind in enumerate(trial_inds):
        t_start = rng.choice(trial_length - snippet_length + 1)
        t_end = t_start + snippet_length
        snippets[i] = trial_data[trial_ind][:, t_start:t_end]
        snippet_times[i] = [t_start, t_end]

    return jnp.array(snippets), jnp.array(snippet_times)


@jit
def reweighted_l1_prox(x: Array, l1_coeff: Array, reweight_coeff: Array) -> Array:
    reweighted_coeffs = _reweight_l1(x, l1_coeff, reweight_coeff)
    x = jnp.sign(x) * jnp.maximum(jnp.abs(x) - reweighted_coeffs, 0)

    return x


@jit
def _reweight_l1(x: Array, l1_coeff: Array, reweight_coeff: float = 200) -> Array:
    return l1_coeff / (1 + reweight_coeff * jnp.abs(x))


@jit
def spectral_normalize(F: Array):
    return F / jnp.linalg.matrix_norm(F, keepdims=True, ord=2)
