import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

from .differential_equation import DifferentialEquation


def plot_speed(ax: Axes, grid: NDArray, flows: NDArray, **plot_kwargs):
    speed = np.sqrt(flows[:, :, 0] ** 2 + flows[:, :, 1] ** 2)
    xmin = grid[:, :, 0].min()
    xmax = grid[:, :, 0].max()
    ymin = grid[:, :, 1].min()
    ymax = grid[:, :, 1].max()

    im = ax.imshow(
        speed,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="YlGn_r",
        **plot_kwargs,
    )
    ax.figure.colorbar(im, ax=ax)


def plot_nullclines(
    ax: Axes,
    grid: NDArray,
    flows: NDArray,
):
    ax.contour(
        grid[:, :, 0],
        grid[:, :, 1],
        flows[:, :, 0],
        levels=[0],
        colors="blue",
    )
    ax.contour(
        grid[:, :, 0],
        grid[:, :, 1],
        flows[:, :, 1],
        levels=[0],
        colors="red",
    )


def plot_trajectories(
    ax: Axes, model: DifferentialEquation, num_trajectories: int, x_0: Array, **kwargs
):
    trajectories = model.sample_trajectories(x_0, num_trajectories, **kwargs)

    lc = LineCollection(trajectories, colors="black", linewidth=0.7, alpha=0.7)
    ax.add_collection(lc)
    ax.scatter(trajectories[:, -1, 0], trajectories[:, -1, 1], c="k", s=10, alpha=0.5)


def compute_flow_field(
    model: DifferentialEquation, start: float, stop: float, step: float, **input_kwargs
):
    axis = jnp.arange(start, stop + step, step)
    grid = jnp.meshgrid(*([axis] * model.state_dim))
    flat_grid = jnp.array([axis.flatten() for axis in grid]).T
    flows = model.compute_xdot(flat_grid, **input_kwargs)

    flows = flows.reshape(axis.shape[0], axis.shape[0], model.state_dim)

    return np.array(grid).transpose(1, 2, 0), np.array(flows), np.array(axis)


def plot_flow_field(
    ax: Axes,
    model: DifferentialEquation,
    start: float,
    stop: float,
    step: float,
    **input_kwargs,
):
    if model.state_dim > 3:
        raise RuntimeError("Can't plot flow fields in more than 3 dimensions!")

    grid, flows, axes = compute_flow_field(model, start, stop, step, **input_kwargs)
    ax.streamplot(
        axes,
        axes,
        flows[:, :, 0],
        flows[:, :, 1],
        density=1,
        color="grey",
        linewidth=1,
    )

    return grid, flows
