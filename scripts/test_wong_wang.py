import argparse
import sys

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from simulations.plot_utils import plot_flow_field, plot_nullclines, plot_trajectories

from simulations import wong_and_wang_RNN


def parse_args(argv: list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--coherence",
        default=0.0,
        type=float,
        help="coherence of the stimulus, between 0 and 100",
    )
    parser.add_argument(
        "-m",
        "--mu0",
        default=30.0,
        type=float,
        help="excitation current when stimulus is presented",
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
        default=1.0,
        type=float,
        help="duration of sampled trajectories in seconds",
    )
    parser.add_argument(
        "--num_trajectories",
        default=100,
        type=int,
        help="number of trajectories to sample",
    )
    parser.add_argument(
        "--seed",
        default=0,
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

    model = wong_and_wang_RNN()
    fig, ax = plt.subplots(figsize=(6, 6))
    grid, arrows = plot_flow_field(
        ax, model, 0, 0.8, 0.05, coherence=args.coherence, mu_0=args.mu0
    )
    plot_trajectories(
        ax,
        model,
        args.num_trajectories,
        jnp.array([0.0, 0.0]),
        dt=args.dt,
        T=args.T,
        sigma=args.sigma,
        seed=args.seed,
        coherence=args.coherence,
        mu_0=args.mu0,
    )

    plot_nullclines(ax, model, 0, 1, coherence=args.coherence, mu_0=args.mu0)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.85)
    plt.show()


if __name__ == "__main__":
    main()
