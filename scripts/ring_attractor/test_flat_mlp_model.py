import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from decomposed_dynamics.dynamics_models import MLPDecomposedDynamics
from decomposed_dynamics.fitting import fit_no_obs
from decomposed_dynamics.inference import BPDNDFHyperparams, BPDNDFNoObsInference
from decomposed_dynamics.proximal_operators import prox_l1_binary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_ring_attractor import (
    parse_args,
    plot_coeff_spatial_maps,
    plot_example_trajectories_with_coeffs,
    plot_MLP_flow_field,
    plot_ring_attractor_flow_and_trajectories,
    simulate_ring_attractor,
)


def main():
    args = parse_args(sys.argv[1:])
    if args is None:
        return

    keys = jr.split(jr.key(args.seed), 6)
    simulation, trajectories = simulate_ring_attractor(args, keys)

    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    model = MLPDecomposedDynamics(
        num_operators=6, state_dim=2, key=keys[3], layer_width=20, num_hidden_layers=4
    )
    inference_hyperparams = NoObsInferenceHyperparams(
        l1_coeff=jnp.array([0.7, 0.4]),
        prox=prox_l1_binary,
        l1_reweight_coeff=jnp.array([200, 0.0]),
        smooth_coeff=0,
    )
    model = fit_no_obs(
        trajectory_dict,
        model,
        samples_per_snippet=20,
        num_snippets=50,
        max_iter=2000,
        lr_init=1,
        lr_end=1e-4,
        inference_hyperparams=inference_hyperparams,
        model_update_hyperparams=model.initialize_hyperparams(decorr_coeff=0.0),
        prox_hyperparams_end=jnp.array([0.7, 0.4]),
    )
    inference_backend = BPDNDFNoObsInference(prox=prox_l1_binary)
    inference_hyperparams = BPDNDFHyperparams(
        prox_hyperparams=jnp.array([0.7, 0.4]),
        prox_reweight_coeff=jnp.array([200, 0.0]),
    )

    mlp_coeffs = inference_backend.infer_batch(
        model,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        inference_hyperparams,
    )
    plotted_grid, plotted_simulation_flows = plot_ring_attractor_flow_and_trajectories(
        args, simulation, trajectories
    )
    plot_example_trajectories_with_coeffs(
        mlp_coeffs,
        trajectories,
        plotted_grid,
        plotted_simulation_flows,
        keys[4],
        "d",
        args,
        imshow=True,
    )
    fig = plot_coeff_spatial_maps(mlp_coeffs, trajectories, "d")
    fig.suptitle(r"spatial map of MLP coefficients $d$")

    for i in range(model.num_operators):
        fig = plot_MLP_flow_field(
            model,
            i,
            -args.max_radius,
            args.max_radius,
            0.05,
            vmin=0,
            vmax=0.25 / 4,
            alpha=0.5,
        )
        fig.suptitle(f"MLP {i} flow field")

    plt.show()


if __name__ == "__main__":
    # jax.disable_jit(disable=True)
    with jax.default_device(jax.devices("cpu")[0]):
        main()
