import time
from types import SimpleNamespace

import click
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, vmap

from decomposed_dynamics.dynamics_models import (
    DecomposedDynamicsModel,
    DecomposedLinearDynamics,
)
from decomposed_dynamics.inference import Penalty
from decomposed_dynamics.inference.bpdn import (
    BPDNDFHyperparams,
    BPDNDFJointNoObsInference,
)
from decomposed_dynamics.inference.proximal_operators import NonNegativeL1Prox
from decomposed_dynamics.inference.sparse_segmented_regression import (
    SparseSegmentedHyperparams,
    SparseSegmentedNoObsInference,
)
from decomposed_dynamics.loss_functions import normalized_dynamics_reconstruction_loss

CONTEXT_SETTINGS = {"show_default": True, "help_option_names": ["-h", "--help"]}


def simulate_piecewise_smooth_coeffs(args, key: Array) -> Array:
    """Each segment: random active operators, amplitude eases smoothly between two random values."""
    ease = (1 - jnp.cos(jnp.pi * jnp.arange(args.segment_length) / args.segment_length)) / 2

    def sample_segment(key: Array) -> Array:
        index_key, start_key, end_key = jr.split(key, 3)
        active = jr.choice(
            index_key, args.num_operators, (args.num_active,), replace=False
        )
        start, end = (
            jr.uniform(k, (args.num_active,), minval=args.min_amplitude, maxval=1.0)
            for k in (start_key, end_key)
        )
        amplitudes = start + ease[:, None] * (end - start)
        return jnp.zeros((args.segment_length, args.num_operators)).at[:, active].set(
            amplitudes
        )

    def sample_trial(key: Array) -> Array:
        segments = jax.vmap(sample_segment)(jr.split(key, args.num_segments))
        return segments.reshape(-1, args.num_operators)

    return jax.vmap(sample_trial)(jr.split(key, args.num_trials))


def simulate_piecewise_smooth_dynamics(
    args, keys: Array
) -> tuple[DecomposedLinearDynamics, Array, Array, Array]:
    model = DecomposedLinearDynamics(args.num_operators, args.state_dim, keys[0])
    coeffs = simulate_piecewise_smooth_coeffs(args, keys[1])

    states = jr.normal(keys[2], (*coeffs.shape[:2], args.state_dim))
    targets = model.combine_operator_predictions(
        states, coeffs, model.compute_operator_predictions(states)
    )
    targets += args.sigma * targets.std() * jr.normal(keys[3], targets.shape)

    return model, coeffs, states, targets


def run_inference(backend, model, states, targets, hyperparams) -> tuple[Array, float]:
    start = time.perf_counter()
    coeffs = backend.infer_batch(model, states, targets, hyperparams)
    return coeffs.block_until_ready(), time.perf_counter() - start


def mean_run_length(is_on: np.ndarray) -> float:
    """Mean length of consecutive-True runs along time, for (trials, time, operators)."""
    padded = np.pad(is_on, ((0, 0), (1, 1), (0, 0))).astype(int)
    num_runs = (np.diff(padded, axis=1) == 1).sum()
    return is_on.sum() / max(num_runs, 1)


def print_recovery_metrics(
    name: str,
    model: DecomposedDynamicsModel,
    true_coeffs: Array,
    coeffs: Array,
    states: Array,
    targets: Array,
    detection_threshold: float,
    seconds: float,
):
    active = true_coeffs > 0
    detected = coeffs > detection_threshold
    recon_loss = normalized_dynamics_reconstruction_loss(
        model,
        states,
        coeffs,
        targets,
        vmap(vmap(model.compute_operator_predictions))(states),
    )
    print(f"{name} ({seconds:.1f}s incl. compile)")
    print(f"  mean absolute error:          {jnp.abs(coeffs - true_coeffs).mean():.4f}")
    print(f"  support accuracy:             {(active == detected).mean():.4f}")
    print(f"  false positive rate:          {detected[~active].mean():.4f}")
    print(f"  nonzero rate when off:        {(coeffs > 0)[~active].mean():.4f}")
    print(
        "  mean nonzero-when-off run:    "
        f"{mean_run_length(np.array((coeffs > 0) & ~active)):.2f}"
    )
    print(f"  false negative rate:          {(~detected[active]).mean():.4f}")
    print(
        "  amplitude correlation (on):   "
        f"{jnp.corrcoef(true_coeffs[active], coeffs[active])[0, 1]:.4f}"
    )
    print(f"  normalized recon loss (mean): {recon_loss.mean():.4f}")


def plot_coeff_maps(true_coeffs: Array, inferred: dict[str, Array], trial_inds: list):
    rows = {"true": true_coeffs, **inferred}
    fig, axes = plt.subplots(
        len(rows),
        len(trial_inds),
        figsize=(3 * len(trial_inds), 2 * len(rows)),
        squeeze=False,
        layout="constrained",
    )

    for column, trial in enumerate(trial_inds):
        for row, (label, coeffs) in enumerate(rows.items()):
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


def plot_coeff_traces(true_coeffs: Array, inferred: dict[str, Array], trial: int):
    num_operators = true_coeffs.shape[-1]
    fig, axes = plt.subplots(
        num_operators,
        len(inferred),
        figsize=(5 * len(inferred), 1.2 * num_operators),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    for column, (label, coeffs) in enumerate(inferred.items()):
        axes[0, column].set_title(label)
        axes[-1, column].set_xlabel("time")
        for operator in range(num_operators):
            ax = axes[operator, column]
            ax.plot(np.array(true_coeffs[trial, :, operator]), "k--", label="true")
            ax.plot(np.array(coeffs[trial, :, operator]), f"C{column}", label=label)
    for operator in range(num_operators):
        axes[operator, 0].set_ylabel(f"$c_{operator}$", rotation=0, labelpad=15)
    fig.suptitle(f"coefficient traces, trial {trial}")

    return fig


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-n", "--num_trials", default=20, help="number of trials to simulate")
@click.option("--num_operators", default=8, help="size of the operator dictionary")
@click.option("--num_active", default=2, help="operators active at any given time")
@click.option("--state_dim", default=6, help="dimension of the state space")
@click.option("--min_amplitude", default=0.3, help="smallest active amplitude")
@click.option("--num_segments", default=5, help="constant-coefficient segments/trial")
@click.option("--segment_length", default=20, help="duration of a segment in samples")
@click.option("--sigma", default=0.1, help="noise std, relative to the target scale")
@click.option("--l0", default=0.1, help="sparse segmented: per-timepoint on cost")
@click.option("--group", default=0.01, help="sparse segmented: per-operator cost")
@click.option("--switch_penalty", default=0.4, help="sparse segmented: changepoint cost")
@click.option("--smooth_coeff", default=0.4, help="sparse segmented: smoothing")
@click.option("--num_sweeps", default=10, help="sparse segmented: descent sweeps")
@click.option(
    "--min_segment_length", default=3, help="sparse segmented: shortest segment"
)
@click.option("--bpdn_l1", default=0.25, help="BPDN-DF: l1 penalty")
@click.option("--bpdn_l1_reweight", default=200.0, help="BPDN-DF: l1 reweighting")
@click.option("--bpdn_smooth_coeff", default=0.4, help="BPDN-DF: smoothing")
@click.option("--detection_threshold", default=0.1, help="coefficient counted as on")
@click.option("--seed", default=0, help="random seed")
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    keys = jr.split(jr.key(args.seed), 4)
    model, true_coeffs, states, targets = simulate_piecewise_smooth_dynamics(args, keys)

    def bpdn_hyperparams(smooth_coeff: float) -> BPDNDFHyperparams:
        return BPDNDFHyperparams(
            prox=NonNegativeL1Prox(Penalty(args.bpdn_l1, reweight=args.bpdn_l1_reweight)),
            smooth_coeff=Penalty(smooth_coeff),
        )

    # both smoothing terms are smooth_coeff * sum 0.5 (c_t - c_{t-1})^2 on top of the
    # summed normalized recon loss, so the same value means the same smoothing
    backends_and_hyperparams = {
        "sparse segmented": (
            SparseSegmentedNoObsInference,
            SparseSegmentedHyperparams(
                l0=Penalty(args.l0),
                group=Penalty(args.group),
                switch_penalty=Penalty(args.switch_penalty),
                smooth_coeff=Penalty(args.smooth_coeff),
                num_sweeps=args.num_sweeps,
                min_segment_length=args.min_segment_length,
            ),
        ),
        f"joint BPDN-DF (smooth {args.bpdn_smooth_coeff:g})": (
            BPDNDFJointNoObsInference,
            bpdn_hyperparams(args.bpdn_smooth_coeff),
        ),
        f"joint BPDN-DF (smooth {args.smooth_coeff:g}, matched)": (
            BPDNDFJointNoObsInference,
            bpdn_hyperparams(args.smooth_coeff),
        ),
    }
    results = {
        label: run_inference(backend, model, states, targets, hyperparams)
        for label, (backend, hyperparams) in backends_and_hyperparams.items()
    }
    inferred = {label: coeffs for label, (coeffs, _) in results.items()}

    for label, (coeffs, seconds) in results.items():
        print_recovery_metrics(
            label,
            model,
            true_coeffs,
            coeffs,
            states,
            targets,
            args.detection_threshold,
            seconds,
        )

    plot_coeff_maps(true_coeffs, inferred, [0, 1, 2])
    plot_coeff_traces(true_coeffs, inferred, 0)
    plt.show()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
