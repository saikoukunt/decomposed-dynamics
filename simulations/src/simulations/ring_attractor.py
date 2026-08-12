from typing import override

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .differential_equation import DifferentialEquation
from .utils import cartesian_to_polar, polar_to_cartesian


class RingAttractorSimulation(DifferentialEquation):
    def __init__(self, dt=0.05):
        super().__init__(2, dt)

    @override
    def compute_xdot(self, x: Array, **input_kwargs) -> Array:
        r, theta = cartesian_to_polar(x)
        radial_flow = -r * (2 * r - 1) * (r - 1)
        x_dot, y_dot = polar_to_cartesian(jnp.vstack((radial_flow, theta)).T)

        return jnp.array([x_dot, y_dot]).T
