from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, tree_util


def extract_snippets(
    trial_data: dict, num_snippets: int, samples_per_snippet: int, seed: int
) -> tuple[Array, Array]:
    keys = list(trial_data.keys())

    num_trials = len(trial_data)
    num_observations = trial_data[keys[0]].shape[1]
    trial_length = min(trial_data[key].shape[0] for key in trial_data)

    rng = np.random.default_rng(seed)

    snippet_length = min(samples_per_snippet, trial_length)
    snippets = np.zeros((num_snippets, snippet_length, num_observations))
    snippet_times = np.zeros((num_snippets, 2), dtype=np.int32)

    if num_snippets == num_trials:
        trial_inds = np.arange(num_snippets)
    else:
        trial_inds = rng.choice(num_trials, num_snippets)

    for i, trial_ind in enumerate(trial_inds):
        t_start = rng.choice(trial_length - snippet_length + 1)
        t_end = t_start + snippet_length
        snippets[i] = trial_data[keys[trial_ind]][t_start:t_end, :]
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


@jit
def operator_correlation(F: Array) -> Array:
    pairwise_corrs = jnp.einsum("kij, lij -> kl", F, F)
    return jnp.sum(jnp.triu(pairwise_corrs**2, k=1))


def repackage_C_hat(C_hat, trial_ids):
    """
    Repackage the inferred dynamics coefficients C_hat into a dictionary where each key corresponds to a unique trial_id.

    Parameters:
    - C_hat: dict
        A dictionary containing inferred dynamics coefficients for each trial.
    - trial_ids: array-like
        An array of trial IDs corresponding to the trials in C_hat.

    Returns:
    - C_hat_repackaged: dict
        A dictionary where each key is a unique trial_id and the value is an array of dynamics coefficients for that trial.
    """
    C_hat_repackaged = {i: np.array([]) for i in np.unique(trial_ids)}
    print(
        f"Repackaging {len(C_hat)} inferred dynamics coefficients into {len(C_hat_repackaged)} = {np.unique(trial_ids)} trials..."
    )
    Ckeys = list(C_hat.keys())
    for i, trial_id in enumerate(trial_ids):
        Ctmp = np.atleast_3d(C_hat[Ckeys[i]].squeeze())
        print(f"trial_id: {trial_id}, Ctmp.shape: {Ctmp.shape}")
        print(
            f"trial_id: {trial_id}, C_hat_repackaged[{int(trial_id)}].shape: {np.atleast_3d(C_hat_repackaged[int(trial_id)]).shape}"
        )
        if len(C_hat_repackaged[int(trial_id)]) == 0:
            C_hat_repackaged[int(trial_id)] = np.atleast_3d(
                np.array(C_hat[Ckeys[i]].squeeze())
            )
        else:
            C_hat_repackaged[int(trial_id)] = np.append(
                np.atleast_3d(C_hat_repackaged[int(trial_id)]),
                np.atleast_3d(np.array(C_hat[Ckeys[i]].squeeze())),
                axis=2,
            )

    return C_hat_repackaged


def eqx_module_to_string(module):
    str = ""
    for path, val in jax.tree.leaves_with_path(module):
        str += f", Avg \U0001d6ab{jax.tree_util.keystr(path)[1:]}: {val:.5f}"

    return str


def prox_binary(x: Any, l1reg: Optional[float] = None, scaling: float = 1.0) -> Any:
    if l1reg is None:
        l1reg = 1.0

    def prox(y):
        to_zero = jax.nn.relu(y - l1reg * scaling)
        one_dist = y - 1
        to_one = 1 + jnp.sign(one_dist) * jax.nn.relu(
            jnp.abs(one_dist) - l1reg * scaling
        )
        to_one = jax.nn.relu(to_one)

        def obj(z):
            R = jnp.minimum(z, jnp.abs(z - 1))
            return 0.5 * (z - y) ** 2 + R * l1reg * scaling

        return jnp.where(obj(to_zero) <= obj(to_one), to_zero, to_one)

    return tree_util.tree_map(prox, x)
