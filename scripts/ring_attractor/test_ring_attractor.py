import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from simulations.plot_utils import (
    plot_speed,
    plot_trajectories,
)

from decomposed_dynamics.dynamics_models import (
    DecomposedLinearDynamics,
    HierarchicalDecomposedDynamics,
    linear_combination,
)
from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel
from decomposed_dynamics.fit_hierarchical import fit_hierarchical_mlps
from decomposed_dynamics.fitting import fit
from decomposed_dynamics.inference import BPDNDFHyperparams, NoObsInferenceBackend
from decomposed_dynamics.inference.bpdn import BPDNDFJointNoObsInference
from decomposed_dynamics.proximal_operators import (
    L1BinaryProx,
    NonNegativeL1Prox,
    Penalty,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import plot_Fs
from saved_params import SavedParams, load_or_fit
from simulate_ring_attractor import (
    CONTEXT_SETTINGS,
    RingAttractorParams,
    load_or_simulate_ring_attractor,
    plot_ring_attractor_flow_and_trajectories,
    ring_attractor_options,
)

NUM_TIMEPOINTS = 20
SAMPLES_PER_SNIPPET = 20


@dataclass(frozen=True)
class DLDSFitParams(SavedParams):
    seed: int
    num_primitives: int = 6
    samples_per_snippet: int = SAMPLES_PER_SNIPPET
    num_snippets: int = 50
    max_iter: int = 200
    lr_init: float = 1.0
    decorr_coeff: float = 0.01
    l1_coeff: float = 0.01
    inference_backend: type[NoObsInferenceBackend] = BPDNDFJointNoObsInference
    inference_hyperparams: BPDNDFHyperparams = field(
        default_factory=lambda: BPDNDFHyperparams(
            prox=NonNegativeL1Prox(Penalty(0.3, reweight=200.0)), smooth_coeff=0.1
        )
    )


@dataclass(frozen=True)
class HierarchicalFitParams(SavedParams):
    seed: int
    num_nonlinear_operators: int = 6
    layer_width: int = 20
    num_hidden_layers: int = 4
    samples_per_snippet: int = SAMPLES_PER_SNIPPET
    num_snippets: int = 10
    max_iter: int = 2000
    lr_init: float = 1e-3
    lr_end: float = 1e-4
    inference_backend: type[NoObsInferenceBackend] = BPDNDFJointNoObsInference
    start_hyperparams: BPDNDFHyperparams = field(
        default_factory=lambda: BPDNDFHyperparams(
            prox=L1BinaryProx(
                l1=Penalty(0.0, reweight=200), binary=Penalty(0.5)
            ),
            smooth_coeff=0.4,
        )
    )
    end_hyperparams: BPDNDFHyperparams = field(
        default_factory=lambda: BPDNDFHyperparams(
            prox=L1BinaryProx(
                l1=Penalty(0.1, reweight=200), binary=Penalty(0.5)
            ),
            smooth_coeff=0.4,
        )
    )


def seed_options(f):
    f = click.option(
        "--sim_seed",
        type=int,
        default=None,
        help="simulation random seed, defaults to --seed",
    )(f)
    return click.option(
        "--seed", default=RingAttractorParams.seed, help="model random seed"
    )(f)


def simulation_params(args) -> RingAttractorParams:
    return RingAttractorParams.from_args(
        args, args.seed if args.sim_seed is None else args.sim_seed
    )


def plot_MLP_flow_field(
    model: DecomposedDynamicsModel,
    mlp_ind: int,
    start: float,
    stop: float,
    step: float,
    **plot_kwargs,
):

    fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

    axis = jnp.arange(start, stop + step, step)
    grid = jnp.meshgrid(*([axis] * model.state_dim))
    flat_grid = jnp.array([axis.flatten() for axis in grid]).T

    flat_flow = model.compute_operator_flows(flat_grid)[:, mlp_ind, :]
    flows = flat_flow.reshape(axis.shape[0], axis.shape[0], model.state_dim)
    flows = np.array(flows)
    axis = np.array(axis)
    grid = np.array(grid).transpose(1, 2, 0)

    ax.streamplot(
        axis,
        axis,
        flows[:, :, 0],
        flows[:, :, 1],
        density=1,
        color="grey",
        linewidth=1,
    )

    plot_speed(ax, grid, flows, **plot_kwargs)

    return fig


def build_model(
    model_type: type[HierarchicalDecomposedDynamics],
    primitive_type: type[DecomposedDynamicsModel],
    dlds_params: DLDSFitParams,
    hierarchical_params: HierarchicalFitParams,
    key,
    **model_kwargs,
) -> HierarchicalDecomposedDynamics:
    return model_type(
        num_nonlinear_operators=hierarchical_params.num_nonlinear_operators,
        num_primitives=dlds_params.num_primitives,
        state_dim=2,
        primitive_type=primitive_type,
        key=key,
        layer_width=hierarchical_params.layer_width,
        num_hidden_layers=hierarchical_params.num_hidden_layers,
        **model_kwargs,
    )


def fit_dlds(primitives, trajectories, params: DLDSFitParams):
    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    _, primitives = fit(
        trajectory_dict,
        primitives,
        samples_per_snippet=params.samples_per_snippet,
        num_snippets=params.num_snippets,
        max_iter=params.max_iter,
        lr_init=params.lr_init,
        inference_backend=params.inference_backend,
        inference_hyperparams=params.inference_hyperparams,
        model_update_hyperparams=primitives.initialize_hyperparams(
            decorr_coeff=params.decorr_coeff, l1_coeff=params.l1_coeff
        ),
    )
    dlds_coeffs = params.inference_backend.infer_batch(
        primitives,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        params.inference_hyperparams,
    )

    return primitives, dlds_coeffs


def fit_hierarchical(
    model: HierarchicalDecomposedDynamics,
    trajectories,
    dlds_coeffs,
    params: HierarchicalFitParams,
):
    trajectory_dict = {i: trajectories[i, :-1, :] for i in range(trajectories.shape[0])}
    dlds_coeff_dict = {i: dlds_coeffs[i] for i in range(dlds_coeffs.shape[0])}

    model = fit_hierarchical_mlps(
        trajectory_dict,
        dlds_coeff_dict,
        model,
        samples_per_snippet=params.samples_per_snippet,
        num_snippets=params.num_snippets,
        max_iter=params.max_iter,
        lr_init=params.lr_init,
        lr_end=params.lr_end,
        inference_backend=params.inference_backend,
        inference_hyperparams=params.start_hyperparams,
        prox_end=params.end_hyperparams.prox,
    )

    coords = trajectories[:, :NUM_TIMEPOINTS, :]
    mlp_coeffs = params.inference_backend.infer_batch(
        model.coefficient_model(),
        coords,
        dlds_coeffs,
        params.end_hyperparams,
    )
    reinferred_mlp_coeffs = params.inference_backend.infer_batch(
        model,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        params.end_hyperparams,
    )

    return model, mlp_coeffs, reinferred_mlp_coeffs


def plot_hierarchical_model_fit(
    model,
    dlds_coeffs,
    mlp_coeffs,
    reinferred_mlp_coeffs,
    trajectories,
    plotted_grid,
    plotted_simulation_flows,
    keys,
    args,
):
    plot_Fs(model.primitives.F)
    plot_example_trajectories_with_coeffs(
        dlds_coeffs,
        trajectories,
        plotted_grid,
        plotted_simulation_flows,
        keys[4],
        "c",
        args,
    )
    fig = plot_coeff_spatial_maps(dlds_coeffs, trajectories, "c")
    fig.suptitle("dLDS inferred coefficients")
    coords = trajectories[:, :NUM_TIMEPOINTS, :]

    plot_example_trajectories_with_coeffs(
        mlp_coeffs.reshape(-1, NUM_TIMEPOINTS, model.num_operators),
        trajectories,
        plotted_grid,
        plotted_simulation_flows,
        keys[5],
        "d",
        args,
        imshow=True,
    )
    per_mlp_dlds_coeff_predictions = model._compute_coeff_predictions_batched(
        model.G, coords.reshape(-1, 2)
    )
    mlp_combined_dlds_coeff_predictions = linear_combination(
        mlp_coeffs.reshape(-1, model.num_operators),
        per_mlp_dlds_coeff_predictions,
    )

    fig = plot_coeff_spatial_maps(mlp_coeffs, trajectories, "d")
    fig.suptitle(r"spatial map of MLP coefficients $d$")

    fig = plot_coeff_spatial_maps(
        mlp_combined_dlds_coeff_predictions, trajectories, "c"
    )
    fig.suptitle("Combined MLP predictions of dLDS coefficients")

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

    for i in range(per_mlp_dlds_coeff_predictions.shape[1]):
        fig = plot_coeff_spatial_maps(
            per_mlp_dlds_coeff_predictions[:, i, :], trajectories, "c"
        )
        fig.suptitle(f"MLP {i} map")

    fig = plot_coeff_spatial_maps(reinferred_mlp_coeffs, trajectories, "d")
    fig.suptitle(r"spatial map of reinferred MLP coefficients $d$")


def plot_example_trajectories_with_coeffs(
    coeffs, trajectories, grid, flows, key, symbol, args, imshow=False
):
    fig, ax = plt.subplots(figsize=(12, 6), nrows=3, ncols=4)
    rand_inds = jr.randint(key, 6, 0, args.num_trajectories)
    for i in range(6):
        row_ind = i % 3
        col_ind = 2 * (i // 3)

        plot_speed(ax[row_ind, col_ind], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
        plot_trajectories(
            ax[row_ind, col_ind], jnp.expand_dims(trajectories[rand_inds[i]], axis=0)
        )
        if not imshow:
            ax[row_ind, col_ind + 1].plot(coeffs[rand_inds[i]])
        else:
            ax[row_ind, col_ind + 1].imshow(coeffs[rand_inds[i]].T)

    fig.suptitle(f"Example trajectories and inferred {symbol}s")
    fig.tight_layout()


def plot_coeff_spatial_maps(coeffs, trajectories, symbol):
    coords = trajectories[:, :NUM_TIMEPOINTS, :].reshape(-1, 2)
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    fig, ax = plt.subplots(
        figsize=(2 + 3 * coeffs.shape[-1] + 2, 3), nrows=1, ncols=coeffs.shape[-1]
    )
    for i in range(coeffs.shape[-1]):
        plot = ax[i].scatter(
            coords[:, 0],
            coords[:, 1],
            c=coeffs[:, i],
            vmin=0,
            vmax=max(1.5, coeffs.max()),
            alpha=0.5,
            s=20,
            cmap="YlGn_r",
        )
        fig.colorbar(plot, ax=ax[i])
        ax[i].set_title(rf"spatial map of ${symbol}_{i}$")

    plt.tight_layout()
    return fig


def run(
    args,
    model_type: type[HierarchicalDecomposedDynamics],
    primitive_type: type[DecomposedDynamicsModel],
    **model_kwargs,
):
    sim_params = simulation_params(args)
    simulation, trajectories = load_or_simulate_ring_attractor(sim_params)
    keys = jr.split(jr.key(args.seed), 6)
    plotted_grid, plotted_simulation_flows = plot_ring_attractor_flow_and_trajectories(
        args, simulation, trajectories
    )

    dlds_params = DLDSFitParams(seed=args.seed)
    hierarchical_params = HierarchicalFitParams(seed=args.seed)
    model = build_model(
        model_type,
        primitive_type,
        dlds_params,
        hierarchical_params,
        keys[3],
        **model_kwargs,
    )

    dlds_dir, primitives, dlds_coeffs = load_or_fit(
        dlds_params,
        f"{sim_params.sim_dir}/{primitive_type.__name__}",
        model.primitives,
        lambda: fit_dlds(model.primitives, trajectories, dlds_params),
        "coeffs",
    )
    model = eqx.tree_at(lambda model: model.primitives, model, primitives)
    plot_Fs(model.primitives.F)
    fig = plot_coeff_spatial_maps(dlds_coeffs, trajectories, "c")
    fig.suptitle("dLDS inferred coefficients")
    plt.show()

    _, model, mlp_coeffs, reinferred_mlp_coeffs = load_or_fit(
        hierarchical_params,
        f"{dlds_dir}/{model_type.__name__}",
        model,
        lambda: fit_hierarchical(model, trajectories, dlds_coeffs, hierarchical_params),
        "mlp_coeffs",
        "reinferred_mlp_coeffs",
    )
    plot_hierarchical_model_fit(
        model,
        dlds_coeffs,
        mlp_coeffs,
        reinferred_mlp_coeffs,
        trajectories,
        plotted_grid,
        plotted_simulation_flows,
        keys,
        args,
    )
    plt.show()


@click.command(context_settings=CONTEXT_SETTINGS)
@ring_attractor_options
@seed_options
def main(**kwargs):
    run(
        SimpleNamespace(**kwargs),
        HierarchicalDecomposedDynamics,
        DecomposedLinearDynamics,
    )


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
