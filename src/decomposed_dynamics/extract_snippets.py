import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array


def extract_snippets(
    trial_data: npt.NDArray, num_snippets: int, samples_per_snippet: int
) -> tuple[Array, Array]:
    num_trials = len(trial_data)
    num_observations = trial_data[0].shape[0]
    trial_length = trial_data[0].shape[1]

    rng = np.random.default_rng(0)

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
        snippets[i] = trial_data[trial_ind, :, t_start:t_end]
        snippet_times[i] = [t_start, t_end]

    return jnp.array(snippets), jnp.array(snippet_times)
