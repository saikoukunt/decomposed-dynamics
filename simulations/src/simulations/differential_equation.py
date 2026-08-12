import functools
from abc import abstractmethod
from math import floor
from typing import Callable, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, jit, vmap
from jax.nn import tanh
from numpy.typing import NDArray


class DifferentialEquation(eqx.Module):
    state_dim: int
    dt: float

    def __init__(
        self,
        state_dim: int,
        dt: float = 0.05,
    ):
        self.state_dim = state_dim
        self.dt = dt

    @abstractmethod
    def compute_xdot(self, x: Array, **input_kwargs) -> Array:
        raise NotImplementedError

    @eqx.filter_jit
    def euler_step(
        self,
        x: Array,
        key: Array,
        sigma: float,
        **input_kwargs,
    ) -> tuple[Array, Array]:
        flow = self.dt * self.compute_xdot(x, **input_kwargs)
        noise = sigma * jnp.sqrt(self.dt) * jr.normal(key, x.shape)

        return x + flow + noise, x + flow + noise

    @eqx.filter_jit
    def euler_trajectory(
        self, x_0: Array, sigma: float, num_iter: int, seed: int, **input_kwargs
    ) -> Array:

        key = jr.key(seed)
        keys = jr.split(key, num_iter)

        euler_step = functools.partial(self.euler_step, sigma=sigma, **input_kwargs)
        _, x = jax.lax.scan(euler_step, init=x_0, xs=keys)

        return x

    def sample_trajectories(
        self,
        x_0: Array,
        num_trajectories: int,
        sigma: float,
        T: float,
        dt: float,
        seed: int,
        **input_kwargs,
    ) -> NDArray:
        num_iter = floor(T / dt)
        seeds = seed + np.arange(num_trajectories)
        single_trajectory = functools.partial(
            self.euler_trajectory, x_0, sigma, num_iter, **input_kwargs
        )
        trajectories = vmap(single_trajectory)(seeds)

        return np.array(trajectories)
