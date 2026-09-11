from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, tree_util
from jaxopt.prox import prox_lasso, prox_non_negative_lasso

from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel


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


# WARN: this means x can't be 1D
@jit
def _reweight_l1(x: Array, l1_coeff: Array, reweight_coeff: float = 200) -> Array:
    denominator = 1 + reweight_coeff[jnp.newaxis, ...] * jnp.abs(x)[..., jnp.newaxis]
    return jnp.squeeze(l1_coeff[jnp.newaxis, ...] / denominator)


@jit
def spectral_normalize(F: Array):
    return F / jnp.linalg.matrix_norm(F, keepdims=True, ord=2)


@jit
def operator_correlation(F: Array) -> Array:
    pairwise_corrs = jnp.einsum("kij, lij -> kl", F, F)
    return jnp.sum(jnp.triu(pairwise_corrs**2, k=1))


@eqx.filter_jit
def operator_flow_correlation(
    model: DecomposedDynamicsModel, locations: Array
) -> Array:
    locations = locations.reshape(-1, model.num_latents)
    predictions = model.compute_operator_predictions(locations)
    flows = predictions - locations[..., jnp.newaxis, :]
    flows = flows.transpose(1, 0, 2)

    # flows = flows / (jnp.linalg.norm(flows, axis=-1, keepdims=True) + 1e-8)
    flows = flows.reshape(model.num_operators, -1)
    flows = flows / (jnp.linalg.norm(flows, axis=-1, keepdims=True) + 1e-8)
    pairwise_corrs = jnp.einsum("in, jn -> ij", flows, flows)

    num_entries = model.num_operators * (model.num_operators - 1) / 2

    return jnp.sum(jnp.triu(jnp.clip(pairwise_corrs, 0.0) ** 2, k=1)) / num_entries


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


def prox_binary(x: Any, _lambda: Optional[float] = None, scaling: float = 1.0) -> Any:
    if _lambda is None:
        _lambda = 1.0

    def prox(y):
        to_zero = jax.nn.relu(y - _lambda * scaling)
        one_dist = y - 1
        to_one = 1 + jnp.sign(one_dist) * jax.nn.relu(
            jnp.abs(one_dist) - _lambda * scaling
        )
        to_one = jax.nn.relu(to_one)

        def obj(z):
            R = jnp.minimum(z, jnp.abs(z - 1))
            return 0.5 * (z - y) ** 2 + R * _lambda * scaling

        return jnp.where(obj(to_zero) <= obj(to_one), to_zero, to_one)

    return tree_util.tree_map(prox, x)


def prox_l1_binary(x: Any, hyperparams_prox: Array, scaling: float = 1.0) -> Any:
    l1_coeff = hyperparams_prox[..., 0]
    _lambda = hyperparams_prox[..., 1]
    x = prox_non_negative_lasso(x, l1_coeff, scaling)
    x = prox_binary(x, _lambda, scaling)

    return x


def prox_unit_tv(x: Any, _lambda: Array, scaling: float = 1.0) -> Any:
    lam = _lambda * scaling
    n_iter = 30
    L = 4.0

    def D(u):
        return jnp.diff(u, axis=0)

    def DT(p):
        return jnp.concatenate([-p[:1, :], p[:-1, :] - p[1:, :], p[-1:, :]], axis=0)

    def project_on_C(u):
        return jnp.clip(u, 0.0, 1.0)

    def project_on_P(p):
        return jnp.clip(p, -lam, lam)

    p_0 = jnp.zeros((x.shape[0] - 1, x.shape[1]))

    def body(_, state):
        p_prev, r, t = state
        p = project_on_P(r + 1 / L * D(project_on_C(x - DT(r))))
        t_next = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
        r = p + ((t - 1) / t_next) * (p - p_prev)
        return p, r, t_next

    p, _, _ = jax.lax.fori_loop(0, n_iter, body, (p_0, p_0, 1.0))

    return project_on_C(x - DT(p))


def prox_l1_unit_tv(x: Any, hyperparams_prox: Array, scaling: float = 1.0) -> Any:
    l1_coeff = hyperparams_prox[..., 0]
    _lambda = hyperparams_prox[..., 1]
    x = prox_unit_tv(x, _lambda, scaling)
    x = prox_lasso(x, l1_coeff, scaling)

    return x
