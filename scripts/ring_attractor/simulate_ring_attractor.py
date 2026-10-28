import os
import sys
from dataclasses import dataclass, fields
from types import SimpleNamespace

import click
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from simulations.plot_utils import (
    compute_flow_field,
    plot_flow_field,
    plot_speed,
    plot_trajectories,
)
from simulations.ring_attractor import RingAttractorSimulation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved_params import SavedParams

CONTEXT_SETTINGS = {"show_default": True, "help_option_names": ["-h", "--help"]}
SIMULATION_ROOT = "results/ring_attractor/simulations"


@dataclass(frozen=True)
class RingAttractorParams(SavedParams):
    num_trajectories: int = 2000
    dt: float = 0.05
    tau: float = 0.2
    sigma: float = 0.0
    T: float = 5.0
    min_radius: float = 0.0
    max_radius: float = 2.0
    seed: int = 0

    @classmethod
    def from_args(cls, args, seed: int) -> "RingAttractorParams":
        kwargs = {
            field.name: getattr(args, field.name)
            for field in fields(cls)
            if field.name != "seed"
        }
        return cls(**kwargs, seed=seed)

    @property
    def sim_dir(self) -> str | None:
        return self.find_dir(SIMULATION_ROOT)

    def save(self, trajectories: NDArray):
        self.make_dir(
            SIMULATION_ROOT,
            lambda path: np.save(f"{path}/trajectories.npy", trajectories),
        )


def simulate_ring_attractor(
    params: RingAttractorParams,
) -> tuple[RingAttractorSimulation, NDArray]:
    simulation = RingAttractorSimulation(dt=params.dt, tau=params.tau)
    keys = jr.split(jr.key(params.seed), 2)

    x_0 = jr.uniform(keys[0], shape=(params.num_trajectories, 2), minval=-1, maxval=1)
    radius = jr.uniform(
        keys[1],
        shape=(params.num_trajectories, 1),
        minval=params.min_radius,
        maxval=params.max_radius,
    )
    x_0 = radius * x_0 / jnp.linalg.norm(x_0, axis=1, keepdims=True)

    trajectories = simulation.sample_trajectories(
        x_0,
        params.num_trajectories,
        sigma=params.sigma,
        T=params.T,
        dt=params.dt,
        seed=params.seed,
    )

    return simulation, np.asarray(trajectories)


def load_or_simulate_ring_attractor(
    params: RingAttractorParams,
) -> tuple[RingAttractorSimulation, NDArray]:
    if params.sim_dir is not None:
        simulation = RingAttractorSimulation(dt=params.dt, tau=params.tau)
        return simulation, np.load(f"{params.sim_dir}/trajectories.npy")

    simulation, trajectories = simulate_ring_attractor(params)
    params.save(trajectories)

    return simulation, trajectories


def plot_ring_attractor_flow_and_trajectories(
    args, simulation: RingAttractorSimulation, trajectories: NDArray
):
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
    grid, _ = plot_flow_field(
        ax[0], simulation, -args.max_radius, args.max_radius, 0.05
    )
    ax[0].set_title("Ring attractor flow field")

    grid, flows, _ = compute_flow_field(
        simulation, -args.max_radius, args.max_radius, 1 / 100
    )
    plot_speed(ax[0], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
    plot_speed(ax[1], grid, flows, vmin=0, vmax=0.25, alpha=0.5)

    plot_trajectories(ax[1], trajectories[:300])
    ax[1].set_title("Many trajectories")

    return grid, flows


def ring_attractor_options(f):
    options = [
        click.option(
            "-n",
            "--num_trajectories",
            default=RingAttractorParams.num_trajectories,
            help="number of trajectories to sample",
        ),
        click.option(
            "--dt",
            default=RingAttractorParams.dt,
            help="time step in seconds for Euler approximation",
        ),
        click.option(
            "--tau",
            default=RingAttractorParams.tau,
            help="timescale in seconds of the flow field",
        ),
        click.option(
            "--sigma", default=RingAttractorParams.sigma, help="white noise variance"
        ),
        click.option(
            "--T",
            "T",
            default=RingAttractorParams.T,
            help="duration of sampled trajectories in seconds",
        ),
        click.option(
            "--min_radius",
            default=RingAttractorParams.min_radius,
            help="minimum radius of initial conditions",
        ),
        click.option(
            "--max_radius",
            default=RingAttractorParams.max_radius,
            help="maximum radius of initial conditions",
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f


@click.command(context_settings=CONTEXT_SETTINGS)
@ring_attractor_options
@click.option("--seed", default=RingAttractorParams.seed, help="random seed")
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    params = RingAttractorParams.from_args(args, args.seed)
    simulation, trajectories = load_or_simulate_ring_attractor(params)
    print(params.sim_dir)
    plot_ring_attractor_flow_and_trajectories(args, simulation, trajectories)
    plt.show()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
