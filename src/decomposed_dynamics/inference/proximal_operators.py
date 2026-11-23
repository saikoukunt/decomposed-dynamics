import math
from abc import abstractmethod
from typing import Any, Callable, Optional, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, tree_util, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_group_lasso, prox_lasso, prox_non_negative_lasso

from decomposed_dynamics.inference.base import Penalty, is_penalty


class ProximalOperator(eqx.Module):
    @abstractmethod
    def __call__(self, x: Array, scaling: float = 1.0) -> Array:
        raise NotImplementedError

    def reweight(self, x: Array) -> Self:
        return jax.tree.map(lambda p: p.reweighted(x), self, is_leaf=is_penalty)


def _prox_coeffs_only(num_latents: int) -> Callable:
    """Apply the prox to the operator coefficients of a `[latents, coeffs]` state only."""

    def wrapped(x: Array, prox: ProximalOperator, scaling: float = 1.0) -> Array:
        latents, coeffs = x[..., :num_latents], x[..., num_latents:]
        return jnp.concatenate([latents, prox(coeffs, scaling)], axis=-1)

    return wrapped


def solve_reweighted(
    solver: ProximalGradient,
    init: Array,
    prox: ProximalOperator,
    num_latents: int = 0,
    **loss_kwargs,
) -> Array:
    solution, _ = solver.run(init, hyperparams_prox=prox, **loss_kwargs)
    solution, _ = solver.run(
        solution,
        hyperparams_prox=prox.reweight(solution[..., num_latents:]),
        **loss_kwargs,
    )

    return jnp.where(jnp.any(jnp.isnan(solution)), jnp.zeros_like(solution), solution)


class NonNegativeL1Prox(ProximalOperator):
    l1: Penalty

    def __call__(self, x: Array, scaling: float = 1.0) -> Array:
        return prox_non_negative_lasso(x, self.l1._lambda, scaling)


class L1BinaryProx(ProximalOperator):
    l1: Penalty
    binary: Penalty

    def __call__(self, x: Array, scaling: float = 1.0) -> Array:
        x = prox_non_negative_lasso(x, self.l1._lambda, scaling)
        x = jnp.clip(x, max=1.0)
        return prox_binary(x, self.binary._lambda, scaling)


class L1GroupBinaryProx(ProximalOperator):
    l1: Penalty
    group: Penalty
    binary: Penalty

    def __call__(self, x: Array, scaling: float = 1.0) -> Array:
        group_coeff = jnp.broadcast_to(
            self.group._lambda * jnp.sqrt(_group_size(x)), x.shape[-1:]
        )
        x = prox_non_negative_lasso(x, self.l1._lambda, scaling)
        x = jnp.clip(x, max=1.0)
        x = vmap(prox_group_lasso, in_axes=(-1, 0, None), out_axes=-1)(
            x, group_coeff, scaling
        )
        return prox_binary(x, self.binary._lambda, scaling)

    def reweight(self, x: Array) -> Self:
        group_norms = jnp.linalg.norm(x, axis=tuple(range(x.ndim - 1))) / jnp.sqrt(
            _group_size(x)
        )
        return L1GroupBinaryProx(
            self.l1.reweighted(x),
            self.group.reweighted(group_norms),
            self.binary.reweighted(x),
        )


class L1UnitTVProx(ProximalOperator):
    l1: Penalty
    tv: Penalty

    def __call__(self, x: Array, scaling: float = 1.0) -> Array:
        x = prox_unit_tv(x, self.tv._lambda, scaling)
        return prox_lasso(x, self.l1._lambda, scaling)

    def reweight(self, x: Array) -> Self:
        return L1UnitTVProx(
            self.l1.reweighted(x), self.tv.reweighted(jnp.diff(x, axis=0))
        )


def apply_prox(x: Array, prox: ProximalOperator, scaling: float = 1.0) -> Array:
    return prox(x, scaling)


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


def _group_size(x: Array) -> int:
    return math.prod(x.shape[:-1])


def prox_unit_tv(x: Any, _lambda: Array, scaling: float = 1.0) -> Any:
    lam = _lambda * scaling
    n_iter = 30
    L = 4.0

    def D(u):
        return jnp.diff(u, axis=0)

    def DT(p):
        return jnp.concatenate([-p[:1, :], p[:-1, :] - p[1:, :], p[-1:, :]], axis=0)

    def project_on_C(u):
        return jnp.clip(u, 0.0)  # , 1.0)

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
