import jax.numpy as jnp
import matplotlib.pyplot as plt

from .rnn import RNN


def plot_flow_field(model: RNN, start: float, stop: float, step: float, **input_kwargs):
    if model.state_dim > 3:
        raise RuntimeError("Can't plot flow fields in more than 3 dimensions!")

    axis = jnp.arange(start, stop, step)
    grid = jnp.meshgrid(*([axis] * model.state_dim))
    grid = jnp.array([axis.flatten() for axis in grid]).T
    arrows = model.compute_xdot(grid, **input_kwargs)

    plt.figure(figsize=(6, 6))
    plt.quiver(grid[:, 0], grid[:, 1], arrows[:, 0], arrows[:, 1])
