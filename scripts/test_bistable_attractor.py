import jax.numpy as jnp
import matplotlib

matplotlib.use("WebAgg")
import argparse
import sys

import matplotlib.pyplot as plt
from simulations.plot_utils import plot_flow_field, plot_nullclines, plot_trajectories

from simulations import bistable_RNN


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
        default=2.0,
        type=float,
        help="excitation current when stimulus is presented",
    )
    parser.add_argument(
        "--w_exc",
        default=0.0,
        type=float,
        help="strength of recurrent self-excitation",
    )
    parser.add_argument(
        "--w_inhib",
        default=6.8,
        type=float,
        help="strength of recurrent mutual inhibition",
    )
    parser.add_argument(
        "--w_input",
        default=0.01,
        type=float,
        help="coefficient of coherence input",
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

    model = bistable_RNN(
        W_exc=args.w_exc, W_inhib=args.w_inhib, W_input=args.w_input, dt=args.dt
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_flow_field(ax, model, 0, 1, 0.05, coherence=args.coherence, mu_0=args.mu0)
    plot_trajectories(
        ax,
        model,
        jnp.array([0, 0]),
        dt=args.dt,
        T=args.T,
        sigma=args.sigma,
        seed=args.seed,
        num_trajectories=args.num_trajectories,
        coherence=args.coherence,
        mu_0=args.mu0,
    )
    plot_nullclines(ax, model, 0, 1, coherence=args.coherence, mu_0=args.mu0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
