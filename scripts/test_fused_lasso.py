import argparse
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from decomposed_dynamics.dynamics_models import DecomposedLinearDynamics
from decomposed_dynamics.inference import FusedLassoHyperparams


def parse_args(argv: list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n", "--num_trials", default=20, type=int, help="number of trials to simulate"
    )
    parser.add_argument(
        "--num_operators", default=8, type=int, help="size of the operator dictionary"
    )
    parser.add_argument(
        "--num_active",
        default=2,
        type=int,
        help="number of operators active at any given time",
    )
    parser.add_argument(
        "--state_dim", default=6, type=int, help="dimension of the state space"
    )
    parser.add_argument(
        "--min_amplitude",
        default=0.3,
        type=float,
        help="smallest amplitude an active operator can take",
    )
    parser.add_argument(
        "--detection_threshold",
        default=0.1,
        type=float,
        help="inferred coefficient above which an operator counts as detected",
    )
    parser.add_argument(
        "--num_segments",
        default=5,
        type=int,
        help="number of constant-coefficient segments per trial",
    )
    parser.add_argument(
        "--segment_length",
        default=20,
        type=int,
        help="duration of a segment in samples",
    )
    parser.add_argument(
        "--sigma",
        default=0.1,
        type=float,
        help="noise standard deviation, relative to the target scale",
    )
    parser.add_argument("--l1_coeff", default=0.05, type=float, help="l1 penalty")
    parser.add_argument("--tv_coeff", default=0.7, type=float, help="tv penalty")
    parser.add_argument(
        "--l1_reweight_coeff", default=0.0, type=float, help="l1 reweighting strength"
    )
    parser.add_argument(
        "--tv_reweight_coeff", default=0.0, type=float, help="tv reweighting strength"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    if "-h" in argv or "--help" in argv:
        parser.print_help()
        return None

    return parser.parse_args(argv)


def simulate_step_coeffs(args, key: Array) -> Array:
    def sample_segment(key: Array) -> Array:
        index_key, amplitude_key = jr.split(key)
        active = jr.choice(
            index_key, args.num_operators, (args.num_active,), replace=False
        )
        amplitudes = jr.uniform(
            amplitude_key, (args.num_active,), minval=args.min_amplitude, maxval=1.0
        )
        return jnp.zeros(args.num_operators).at[active].set(amplitudes)

    def sample_trial(key: Array) -> Array:
        segments = jax.vmap(sample_segment)(jr.split(key, args.num_segments))
        return jnp.repeat(segments, args.segment_length, axis=0)

    return jax.vmap(sample_trial)(jr.split(key, args.num_trials))


def simulate_sparse_switching_dynamics(
    args, keys: Array
) -> tuple[DecomposedLinearDynamics, Array, Array, Array]:
    model = DecomposedLinearDynamics(args.num_operators, args.state_dim, keys[0])
    coeffs = simulate_step_coeffs(args, keys[1])

    num_samples = args.num_segments * args.segment_length
    states = jr.normal(keys[2], (args.num_trials, num_samples, args.state_dim))
    targets = model.combine_operator_predictions(
        coeffs, model.compute_operator_predictions(states)
    )
    targets += args.sigma * targets.std() * jr.normal(keys[3], targets.shape)

    return model, coeffs, states, targets


def plot_coeff_maps(true_coeffs: Array, inferred_coeffs: Array, trial_inds: list):
    fig, axes = plt.subplots(
        2,
        len(trial_inds),
        figsize=(3 * len(trial_inds), 4),
        squeeze=False,
        layout="constrained",
    )

    for column, trial in enumerate(trial_inds):
        for row, (coeffs, label) in enumerate(
            [(true_coeffs, "true"), (inferred_coeffs, "inferred")]
        ):
            ax = axes[row, column]
            image = ax.matshow(
                np.array(coeffs[trial]).T, aspect="auto", vmin=0, vmax=1, cmap="magma"
            )
            ax.xaxis.set_ticks_position("bottom")
            ax.set_title(f"trial {trial}, {label}")
            ax.set_ylabel("operator")
    axes[-1, 0].set_xlabel("time")
    fig.colorbar(image, ax=axes, shrink=0.8)

    return fig


def plot_coeff_traces(true_coeffs: Array, inferred_coeffs: Array, trial: int):
    num_operators = true_coeffs.shape[-1]
    fig, axes = plt.subplots(
        num_operators, 1, figsize=(8, 1.2 * num_operators), sharex=True, sharey=True
    )

    for operator, ax in enumerate(axes):
        ax.plot(np.array(true_coeffs[trial, :, operator]), "k--", label="true")
        ax.plot(np.array(inferred_coeffs[trial, :, operator]), label="inferred")
        ax.set_ylabel(f"$c_{operator}$", rotation=0, labelpad=15)
    fig.suptitle(f"coefficient traces, trial {trial}")

    fig.tight_layout()
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time")
    return fig


def plot_recovery_summary(true_coeffs: Array, inferred_coeffs: Array):
    true = np.array(true_coeffs).ravel()
    active = true > 0
    inferred = np.array(inferred_coeffs).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(0, max(inferred.max(), 1.0), 50)
    axes[0].hist(inferred[~active], bins=bins, alpha=0.6, label="true off")
    axes[0].hist(inferred[active], bins=bins, alpha=0.6, label="true on")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("inferred coefficient")
    axes[0].set_ylabel("count")
    axes[0].legend()

    changes = np.diff(np.array(true_coeffs), axis=1) != 0
    axes[1].plot(np.abs(np.diff(np.array(inferred_coeffs), axis=1))[changes], ".")
    axes[1].set_xlabel("true switch")
    axes[1].set_ylabel("inferred step size at switch")

    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].plot(true[active], inferred[active], ".", alpha=0.2)
    axes[2].set_xlabel("true amplitude")
    axes[2].set_ylabel("inferred amplitude")

    fig.tight_layout()
    fig.suptitle("support recovery and switch detection")

    return fig


def main():
    args = parse_args(sys.argv[1:])
    if args is None:
        return

    keys = jr.split(jr.key(args.seed), 4)
    model, true_coeffs, states, targets = simulate_sparse_switching_dynamics(args, keys)

    inference_hyperparams = FusedLassoHyperparams(
        prox_hyperparams=jnp.array([args.l1_coeff, args.tv_coeff]),
        prox_reweight_coeff=jnp.array([args.l1_reweight_coeff, args.tv_reweight_coeff]),
    )
    inference_backend = inference_hyperparams.get_backend()
    inferred_coeffs = inference_backend.infer_batch(
        model,
        states,
        targets,
        inference_hyperparams,
        model.compute_operator_predictions,
    )

    active = true_coeffs > 0
    detected = inferred_coeffs > args.detection_threshold
    print(f"mean absolute error: {jnp.abs(inferred_coeffs - true_coeffs).mean():.4f}")
    print(f"support accuracy: {(active == detected).mean():.4f}")
    print(f"mean inferred coefficient when on: {inferred_coeffs[active].mean():.4f}")
    print(f"mean inferred coefficient when off: {inferred_coeffs[~active].mean():.4f}")
    print(
        "amplitude correlation when on: "
        f"{jnp.corrcoef(true_coeffs[active], inferred_coeffs[active])[0, 1]:.4f}"
    )

    plot_coeff_maps(true_coeffs, inferred_coeffs, [0, 1, 2])
    plot_coeff_traces(true_coeffs, inferred_coeffs, 0)
    plot_recovery_summary(true_coeffs, inferred_coeffs)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
