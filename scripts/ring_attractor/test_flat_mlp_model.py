import os
import sys
from types import SimpleNamespace

import click
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from decomposed_dynamics.dynamics_models import (
    MLPDecomposedDeltaDynamics,
    MLPDecomposedDynamics,
)
from decomposed_dynamics.dynamics_models.mlp import visited_grid
from decomposed_dynamics.fitting import fit
from decomposed_dynamics.inference import (
    BPDNDFHyperparams,
    BPDNDFNoObsInference,
    InferenceHyperparams,
    NoObsInferenceBackend,
)
from decomposed_dynamics.inference.bpdn import BPDNDFBatchNoObsInference
from decomposed_dynamics.inference.fused_lasso import (
    FusedLassoHyperparams,
    FusedLassoNoObsInference,
)
from decomposed_dynamics.proximal_operators import prox_l1_binary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_ring_attractor import main as ring_attractor_main
from test_ring_attractor import (
    plot_coeff_spatial_maps,
    plot_example_trajectories_with_coeffs,
    plot_MLP_flow_field,
    plot_ring_attractor_flow_and_trajectories,
    simulate_ring_attractor,
)


def fit_infer_flat_mlp_model(
    model: MLPDecomposedDynamics,
    trajectories,
    inference_backend: NoObsInferenceBackend,
    start_hyperparams: InferenceHyperparams,
    end_hyperparams: InferenceHyperparams,
):
    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    _, model = fit(
        trajectory_dict,
        model,
        samples_per_snippet=20,
        num_snippets=50,
        max_iter=1000,
        lr_init=1,
        lr_end=1e-4,
        inference_backend=inference_backend,
        inference_hyperparams=start_hyperparams,
        model_update_hyperparams=model.initialize_hyperparams(
            normalization_grid=visited_grid(trajectories), decorr_coeff=0.0
        ),
        prox_hyperparams_end=end_hyperparams.prox_hyperparams,
    )
    mlp_coeffs = inference_backend.infer_batch(
        model,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        end_hyperparams,
    )

    return model, mlp_coeffs


@click.command(
    params=ring_attractor_main.params,
    context_settings=ring_attractor_main.context_settings,
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    keys = jr.split(jr.key(args.seed), 6)
    simulation, trajectories = simulate_ring_attractor(args, keys)

    model = MLPDecomposedDynamics(
        num_operators=6,
        state_dim=2,
        key=keys[3],
        layer_width=20,
        num_hidden_layers=4,
    )
    model, mlp_coeffs = fit_infer_flat_mlp_model(
        model,
        trajectories,
        inference_backend=BPDNDFBatchNoObsInference(),
        start_hyperparams=BPDNDFHyperparams(
            prox_hyperparams=(0.0, 0.0, 0.4),
            prox_reweight_coeff=(200, 200, 0),
            smooth_coeff=0.4,
        ),
        end_hyperparams=BPDNDFHyperparams(
            prox_hyperparams=(0.1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.5),
            prox_reweight_coeff=(200, 5, 0),
            smooth_coeff=0.4,
        ),
        # inference_backend=BPDNDFNoObsInference(prox=prox_l1_binary),
        # start_hyperparams=BPDNDFHyperparams(
        #     prox_hyperparams=(0.0, 0.4),
        #     prox_reweight_coeff=(200, 0.0),
        #     smooth_coeff=0.4,
        # ),
        # end_hyperparams=BPDNDFHyperparams(
        #     prox_hyperparams=([0.1, 0.5, 0.5, 0.5, 0.5, 0.5], 0.4),
        #     prox_reweight_coeff=(200, 0.0),
        #     smooth_coeff=0.4,
        # ),
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
