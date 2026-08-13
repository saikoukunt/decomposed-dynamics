import argparse
import sys

import jax
import jax.random as jr
import matplotlib.pyplot as plt
from simulations.plot_utils import (
    compute_flow_field,
    plot_flow_field,
    plot_speed,
    plot_trajectories,
)
from simulations.ring_attractor import RingAttractorSimulation


def parse_args(argv: list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n",
        "--num_trajectories",
        default=100,
        type=int,
        help="number of trajectories to sample",
    )
    parser.add_argument(
        "--dt",
        default=0.05,
        type=float,
        help="time step in seconds for Euler approximation",
    )
    parser.add_argument(
        "--sigma",
        default=0.1,
        type=float,
        help="white noise variance",
    )
    parser.add_argument(
        "--T",
        default=7.5,
        type=float,
        help="duration of sampled trajectories in seconds",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--min_value",
        default=-1.5,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--max_value",
        default=1.5,
        type=int,
        help="random seed",
    )
    if "-h" in argv or "--help" in argv:
        parser.print_help()
        return None

    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    if args is None:
        return

    model = RingAttractorSimulation()
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
    grid, arrows = plot_flow_field(ax[0], model, -1.25, 1.25, 0.05)

    grid, flows, axes = compute_flow_field(model, -1.25, 1.25, 1 / 100)
    plot_speed(ax[0], grid, flows, vmin=0, vmax=0.075, alpha=0.5)
    plot_speed(ax[1], grid, flows, vmin=0, vmax=0.075, alpha=0.5)

    key, subkey = jr.split(jr.key(args.seed))
    x_0 = jr.uniform(
        key,
        shape=(args.num_trajectories, 2),
        minval=args.min_value,
        maxval=args.max_value,
    )

    trajectories = model.sample_trajectories(
        x_0,
        args.num_trajectories,
        sigma=args.sigma,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )
    plot_trajectories(ax[1], trajectories)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
