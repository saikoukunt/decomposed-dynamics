import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array


def min_trial_length(data: dict) -> int:
    """
    Return the minimum size of axis=1 across all array-like values in `data`.
    Raises on empty dict or if any value is not at least 2D-array-like.
    """
    if not data:
        raise ValueError("input dictionary is empty")

    sizes = []
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim < 2:
            raise ValueError(f"value for key {k!r} has ndim < 2 (shape={arr.shape})")
        sizes.append(int(arr.shape[1]))

    return int(min(sizes))


def extract_one_snippet(
    trial_data: npt.NDArray, samples_per_snippet: int, seed: int
) -> tuple[Array, Array]:
    observation_dim = trial_data.shape[0]
    trial_length = trial_data.shape[1]

    rng = np.random.default_rng(seed)

    snippet_length = min(samples_per_snippet, trial_length)
    snippets = np.zeros((observation_dim, snippet_length))
    snippet_times = np.zeros((2), dtype=np.int32)

    t_start = rng.choice(trial_length - snippet_length + 1)
    t_end = t_start + snippet_length
    snippets = trial_data[:, t_start:t_end]
    snippet_times = [t_start, t_end]

    return jnp.array(snippets), jnp.array(snippet_times)


def extract_snippets(
    trial_data: npt.NDArray, num_snippets: int, samples_per_snippet: int, seed: int
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    snippet_seeds = rng.integers(num_snippets,size=num_snippets)

    num_trials = len(trial_data)
    dict_keys = list(trial_data.keys())
    observation_dim = trial_data[dict_keys[1]].shape[0]
    snippet_length = min(samples_per_snippet, min_trial_length(trial_data))

    snippets = np.zeros((num_snippets, observation_dim, snippet_length))
    snippet_times = np.zeros((num_snippets, 2), dtype=np.int32)
    trial_inds = rng.choice(num_trials, num_snippets)

    for i, trial_ind in enumerate(trial_inds):
        snippets[i], snippet_times[i] = extract_one_snippet(
            trial_data[dict_keys[trial_ind]], snippet_length, snippet_seeds[i]
        )

    return jnp.array(snippets), jnp.array(snippet_times)
