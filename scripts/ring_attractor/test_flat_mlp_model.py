import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import click
import jax
import jax.random as jr
import matplotlib.pyplot as plt

from decomposed_dynamics.dynamics_models import (
    MLPDecomposedDynamics,
)
from decomposed_dynamics.dynamics_models.mlp import visited_grid
from decomposed_dynamics.fitting import fit
from decomposed_dynamics.inference import (
    InferenceHyperparams,
    NoObsInferenceBackend,
    Penalty,
)
from decomposed_dynamics.inference.binary_coordinate_descent import (
    BinaryViterbiHyperparams,
    BinaryViterbi,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved_params import SavedParams, load_or_fit
from simulate_ring_attractor import (
    load_or_simulate_ring_attractor,
    plot_ring_attractor_flow_and_trajectories,
)
from test_ring_attractor import main as ring_attractor_main
from test_ring_attractor import (
    plot_coeff_spatial_maps,
    plot_example_trajectories_with_coeffs,
    plot_MLP_flow_field,
    simulation_params,
)


@dataclass(frozen=True)
class FlatMLPFitParams(SavedParams):
    seed: int
    num_operators: int = 6
    layer_width: int = 20
    num_hidden_layers: int = 4
    samples_per_snippet: int = 20
    num_snippets: int = 50
    max_iter: int = 2000
    lr_init: float = 1.0
    lr_end: float = 1e-4
    decorr_coeff: float = 0.0
    inference_backend: type[NoObsInferenceBackend] = (
        BinaryViterbi
    )
    start_hyperparams: InferenceHyperparams = field(
        default_factory=lambda: BinaryViterbiHyperparams(
            l0=Penalty(0.1), group=Penalty(0.2), smooth_coeff=Penalty(0.4)
        )
    )
    end_hyperparams: InferenceHyperparams = field(
        default_factory=lambda: BinaryViterbiHyperparams(
            l0=Penalty(0.1), group=Penalty(0.2), smooth_coeff=Penalty(0.4)
        )
    )


def fit_flat_mlp(model: MLPDecomposedDynamics, trajectories, params: FlatMLPFitParams):
    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    _, model = fit(
        trajectory_dict,
        model,
        samples_per_snippet=params.samples_per_snippet,
        num_snippets=params.num_snippets,
        max_iter=params.max_iter,
        lr_init=params.lr_init,
        lr_end=params.lr_end,
        inference_backend=params.inference_backend,
        inference_hyperparams=params.start_hyperparams,
        model_update_hyperparams=model.initialize_hyperparams(
            normalization_grid=visited_grid(trajectories),
            decorr_coeff=params.decorr_coeff,
        ),
        inference_hyperparams_end=params.end_hyperparams,
    )
    mlp_coeffs = params.inference_backend.infer_batch(
        model,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        params.end_hyperparams,
    )

    return model, mlp_coeffs


@click.command(
    params=ring_attractor_main.params,
    context_settings=ring_attractor_main.context_settings,
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    keys = jr.split(jr.key(args.seed), 6)
    sim_params = simulation_params(args)
    simulation, trajectories = load_or_simulate_ring_attractor(sim_params)

    params = FlatMLPFitParams(seed=args.seed)
    model = MLPDecomposedDynamics(
        num_operators=params.num_operators,
        state_dim=2,
        key=keys[3],
        layer_width=params.layer_width,
        num_hidden_layers=params.num_hidden_layers,
    )
    _, model, mlp_coeffs = load_or_fit(
        params,
        f"{sim_params.sim_dir}/{type(model).__name__}",
        model,
        lambda: fit_flat_mlp(model, trajectories, params),
        "coeffs",
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
