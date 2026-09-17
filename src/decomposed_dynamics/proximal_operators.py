from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array, jit, tree_util
from jaxopt.prox import prox_lasso, prox_non_negative_lasso


@jit
def reweighted_l1_prox(x: Array, l1_coeff: Array, reweight_coeff: Array) -> Array:
    reweighted_coeffs = _reweight_prox_hyperparams(x, l1_coeff, reweight_coeff)
    x = jnp.sign(x) * jnp.maximum(jnp.abs(x) - reweighted_coeffs, 0)

    return x


# WARN: this means x can't be 1D
@jit
def _reweight_prox_hyperparams(
    x: Array, prox_hyperparams: Array, reweight_coeff: float = 200
) -> Array:
    denominator = 1 + reweight_coeff[jnp.newaxis, ...] * jnp.abs(x)[..., jnp.newaxis]
    return jnp.squeeze(prox_hyperparams[jnp.newaxis, ...] / denominator)


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
