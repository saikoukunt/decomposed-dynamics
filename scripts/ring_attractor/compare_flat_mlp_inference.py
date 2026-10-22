import os
import sys
from types import SimpleNamespace

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from simulations.plot_utils import plot_speed, plot_trajectories

from decomposed_dynamics.dynamics_models import MLPDecomposedDynamics
from decomposed_dynamics.dynamics_models.mlp import visited_grid
from decomposed_dynamics.fitting import (
    dynamics_recon_value_and_grad,
    update_dynamics_model,
)
from decomposed_dynamics.inference import BPDNDFHyperparams
from decomposed_dynamics.inference.bpdn import (
    BPDNDFBatchNoObsInference,
    BPDNDFBatchRandomOnNoObsInference,
    BPDNDFRandomOnHyperparams,
)
from decomposed_dynamics.utils import make_filter_spec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate_ring_attractor import (
    CONTEXT_SETTINGS,
    load_or_simulate_ring_attractor,
    plot_ring_attractor_flow_and_trajectories,
    ring_attractor_options,
)
from test_ring_attractor import seed_options, simulation_params

NUM_PLOTTED_TRAJECTORIES = 10
LEARNING_RATE = 1.0
GROUP_COEFFS = [0, 5, 5, 5, 5, 5]

INFERENCE_CONFIGS = [
    (
        BPDNDFBatchRandomOnNoObsInference(
            turn_on_operators=(True, False, False, False, False, False)
        ),
        BPDNDFRandomOnHyperparams(
            prox_hyperparams=(0, GROUP_COEFFS, 0.1),
            prox_reweight_coeff=(200, 1, 0),
            smooth_coeff=0,
            turn_on_prob=0.2,
        ),
    ),
    (
        BPDNDFBatchRandomOnNoObsInference(
            turn_on_operators=(False, False, False, False, False, False)
        ),
        BPDNDFRandomOnHyperparams(
            prox_hyperparams=(0, GROUP_COEFFS, 0.1),
            prox_reweight_coeff=(200, 1, 0),
            smooth_coeff=0,
            turn_on_prob=0.2,
        ),
    ),
    (
        BPDNDFBatchNoObsInference(),
        BPDNDFHyperparams(
            prox_hyperparams=(0, GROUP_COEFFS, 0.1),
            prox_reweight_coeff=(200, 1, 0),
            smooth_coeff=0,
        ),
    ),
    (
        BPDNDFBatchNoObsInference(),
        BPDNDFHyperparams(
            prox_hyperparams=(0, GROUP_COEFFS, 0.4),
            prox_reweight_coeff=(200, 1, 0),
            smooth_coeff=0,
        ),
    ),
]


def describe_config(backend, hyperparams):
    prox_hyperparams = ", ".join(
        str(np.round(np.asarray(p), 2))
        for p in jax.tree.leaves(hyperparams.prox_hyperparams)
    )
    return (
        f"{type(backend).__name__} ({backend.prox.__name__}), "
        f"prox_hyperparams=({prox_hyperparams})"
    )


def plot_trajectories_with_coeffs(coeffs, trajectories, grid, flows, title):
    num_rows = NUM_PLOTTED_TRAJECTORIES // 2
    fig, ax = plt.subplots(figsize=(12, 2 * num_rows), nrows=num_rows, ncols=4)
    for i in range(NUM_PLOTTED_TRAJECTORIES):
        row_ind = i % num_rows
        col_ind = 2 * (i // num_rows)

        plot_speed(ax[row_ind, col_ind], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
        plot_trajectories(ax[row_ind, col_ind], trajectories[i : i + 1])
        ax[row_ind, col_ind + 1].imshow(coeffs[i].T)

    fig.suptitle(title)
    fig.tight_layout()


@click.command(context_settings=CONTEXT_SETTINGS)
@ring_attractor_options
@seed_options
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    keys = jr.split(jr.key(args.seed), 7)
    simulation, trajectories = load_or_simulate_ring_attractor(simulation_params(args))

    plotted_grid, plotted_simulation_flows = plot_ring_attractor_flow_and_trajectories(
        args, simulation, trajectories
    )
    batch_inds = jr.choice(keys[4], args.num_trajectories, (1000,), replace=False)
    batch = trajectories[batch_inds]

    model = MLPDecomposedDynamics(
        num_operators=6,
        state_dim=2,
        key=keys[3],
        layer_width=20,
        num_hidden_layers=4,
    )

    fit_keys = jr.split(keys[5], 10)
    # for i in range(10):
    #     active_operators = jr.randint(
    #         fit_keys[i], (batch.shape[0], batch.shape[1] - 1), 0, model.num_operators
    #     )
    #     fake_coeffs = jax.nn.one_hot(active_operators, model.num_operators)
    #     filter_spec = make_filter_spec(model)
    #     _, grads = dynamics_recon_value_and_grad(
    #         *eqx.partition(model, filter_spec), batch, fake_coeffs
    #     )
    #     model, _ = update_dynamics_model(
    #         model,
    #         grads,
    #         LEARNING_RATE,
    #         model.initialize_hyperparams(
    #             normalization_grid=visited_grid(trajectories), decorr_coeff=0.0
    #         ),
    #         batch,
    #         filter_spec,
    #     )

    batch_inds = jr.choice(
        keys[6], args.num_trajectories, (NUM_PLOTTED_TRAJECTORIES,), replace=False
    )
    batch = trajectories[batch_inds]
    for i, (backend, hyperparams) in enumerate(INFERENCE_CONFIGS):
        coeffs = backend.infer_batch(
            model, batch[:, :-1, :], batch[:, 1:, :], hyperparams
        )
        plot_trajectories_with_coeffs(
            coeffs,
            batch,
            plotted_grid,
            plotted_simulation_flows,
            f"Config {i}: {describe_config(backend, hyperparams)}",
        )

    plt.show()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
