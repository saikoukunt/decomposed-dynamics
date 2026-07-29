import matplotlib

matplotlib.use("WebAgg")
import argparse
import sys

import matplotlib.pyplot as plt
from simulations.plot_utils import plot_flow_field

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
        "--w_exc", default=0.0, type=float, help="strength of recurrent self-excitation"
    )
    parser.add_argument(
        "--w_inhib",
        default=6.8,
        type=float,
        help="strength of recurrent mutual inhibition",
    )
    parser.add_argument(
        "--w_input", default=0.01, type=float, help="coefficient of coherence input"
    )
    parser.add_argument(
        "--dt", default=0.05, type=float, help="time step in Euler approximation"
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
    plot_flow_field(model, 0, 1, 0.05, coherence=args.coherence, mu_0=args.mu0)
    plt.show()


if __name__ == "__main__":
    main()
