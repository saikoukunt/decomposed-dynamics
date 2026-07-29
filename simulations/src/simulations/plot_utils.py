import jax.numpy as jnp
from jax import Array
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .rnn import RNN


def plot_nullclines(ax: Axes, model: RNN, x_min: float, x_max: float, **input_kwargs):
    grid, flows = compute_flow_field(
        model, x_min, x_max, (x_max - x_min) / 100, **input_kwargs
    )
    grid = grid.reshape(101, 101, model.state_dim)
    flows = flows.reshape(101, 101, model.state_dim)
    ax.contour(
        grid[:, :, 0],
        grid[:, :, 1],
        flows[:, :, 0],
        levels=[0],
        colors="blue",
        linewidth=1,
        label="x nullcline",
    )
    ax.contour(
        grid[:, :, 0],
        grid[:, :, 1],
        flows[:, :, 1],
        levels=[0],
        colors="red",
        linewidth=1,
        label="y nullcline",
    )


def plot_trajectories(ax: Axes, model: RNN, x_0: Array, **kwargs):
    trajectories = model.sample_trajectories(x_0, **kwargs)

    lc = LineCollection(trajectories, colors="green", linewidth=0.5, alpha=0.3)
    ax.add_collection(lc)
    ax.scatter(trajectories[:, -1, 0], trajectories[:, -1, 1], c="green", s=3)


def compute_flow_field(
    model: RNN, start: float, stop: float, step: float, **input_kwargs
):
    axis = jnp.arange(start, stop + step, step)
    grid = jnp.meshgrid(*([axis] * model.state_dim))
    grid = jnp.array([axis.flatten() for axis in grid]).T
    flows = model.compute_xdot(grid, **input_kwargs)

    return grid, flows


def plot_flow_field(
    ax: Axes, model: RNN, start: float, stop: float, step: float, **input_kwargs
):
    if model.state_dim > 3:
        raise RuntimeError("Can't plot flow fields in more than 3 dimensions!")

    grid, arrows = compute_flow_field(model, start, stop, step, **input_kwargs)
    ax.quiver(grid[:, 0], grid[:, 1], arrows[:, 0], arrows[:, 1], color="darkgrey")
