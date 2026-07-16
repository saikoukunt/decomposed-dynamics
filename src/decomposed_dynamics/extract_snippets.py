import jax.numpy as jnp
import numpy as np
from jax import Array


def extract_snippets(
    trial_data: dict, num_snippets: int, samples_per_snippet: int, seed: int
) -> tuple[Array, Array]:
    keys = list(trial_data.keys())

    num_trials = len(trial_data)
    num_observations = trial_data[keys[0]].shape[0]
    min_trial_length = min(trial_data[key].shape[1] for key in trial_data)

    rng = np.random.default_rng(seed)

    snippet_length = min(samples_per_snippet, min_trial_length)
    snippets = np.zeros((num_snippets, num_observations, snippet_length))
    snippet_times = np.zeros((num_snippets, 2), dtype=np.int32)

    if num_snippets == num_trials:
        trial_inds = np.arange(num_snippets)
    else:
        trial_inds = rng.choice(num_trials, num_snippets)

    for i, trial_ind in enumerate(trial_inds):
        t_start = rng.choice(min_trial_length - snippet_length + 1)
        t_end = t_start + snippet_length
        snippets[i] = trial_data[keys[trial_ind]][:, t_start:t_end]
        snippet_times[i] = [t_start, t_end]

    return jnp.array(snippets), jnp.array(snippet_times)
