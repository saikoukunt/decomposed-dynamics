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
from jax import Array
from simulations.plot_utils import (
    compute_flow_field,
    plot_flow_field,
    plot_speed,
    plot_trajectories,
)
from simulations.ring_attractor import RingAttractorSimulation

from decomposed_dynamics.dynamics_models import (
    DecomposedLinearDynamics,
    HierarchicalDecomposedDynamics,
)
from decomposed_dynamics.dynamics_models.base import DecomposedDynamicsModel
from decomposed_dynamics.fit_hierarchical import fit_hierarchical_mlps
from decomposed_dynamics.fitting import fit
from decomposed_dynamics.inference import BPDNDFHyperparams
from decomposed_dynamics.inference.bpdn import BPDNDFNoObsInference
from decomposed_dynamics.proximal_operators import prox_l1_binary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import plot_Fs


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

    flat_operator_predictions = model.compute_operator_predictions(flat_grid)[
        :, mlp_ind, :
    ]
    flat_flow = flat_operator_predictions - flat_grid
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


def plot_ring_attractor_flow_and_trajectories(
    args, simulation: RingAttractorSimulation, trajectories: Array
):
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
    grid, _ = plot_flow_field(
        ax[0], simulation, -args.max_radius, args.max_radius, 0.05
    )
    ax[0].set_title("Ring attractor flow field")

    grid, flows, _ = compute_flow_field(
        simulation, -args.max_radius, args.max_radius, 1 / 100
    )
    plot_speed(ax[0], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
    plot_speed(ax[1], grid, flows, vmin=0, vmax=0.25, alpha=0.5)

    plot_trajectories(ax[1], trajectories[:300])
    ax[1].set_title("Many trajectories")

    return grid, flows


def simulate_ring_attractor(args, keys):
    simulation = RingAttractorSimulation(dt=args.dt, tau=args.tau)

    x_0 = jr.uniform(
        keys[0],
        shape=(args.num_trajectories, 2),
        minval=-1,
        maxval=1,
    )
    radius = jr.uniform(
        keys[1],
        shape=(args.num_trajectories, 1),
        minval=args.min_radius,
        maxval=args.max_radius,
    )
    x_0 = radius * x_0 / jnp.linalg.norm(x_0, axis=1, keepdims=True)

    trajectories = simulation.sample_trajectories(
        x_0,
        args.num_trajectories,
        sigma=args.sigma,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )

    return simulation, trajectories


def load_dlds_coeffs(dlds_model, model_path, coeffs_path):
    if os.path.exists(model_path) and os.path.exists(coeffs_path):
        model_fit = eqx.tree_deserialise_leaves(model_path, dlds_model)
        dlds_coeffs = jnp.load(coeffs_path)

        return model_fit, dlds_coeffs
    else:
        return None, None


def fit_infer_hierarchical_model_all_stages(
    model: HierarchicalDecomposedDynamics,
    trajectory_dict,
    trajectories,
):
    model_path = "results/ring_attractor/dlds_model.eqx"
    coeffs_path = "results/ring_attractor/dlds_coeffs.npy"
    model_fit, dlds_coeffs = load_dlds_coeffs(model.primitives, model_path, coeffs_path)

    if model_fit is None:
        inference_hyperparams = BPDNDFHyperparams(prox_hyperparams=0.7)
        _, model_fit = fit(
            trajectory_dict,
            model.primitives,
            samples_per_snippet=60,
            num_snippets=50,
            max_iter=100,
            lr_init=1,
            inference_hyperparams=inference_hyperparams,
            model_update_hyperparams=model.initialize_hyperparams(
                decorr_coeff=0.02, l1_coeff=0.01
            ).primitive_hyperparams,
        )
        inference_hyperparams = BPDNDFHyperparams(
            prox_hyperparams=0.7, smooth_coeff=0.4
        )
        inference_backend = BPDNDFNoObsInference()
        dlds_coeffs = inference_backend.infer_batch(
            model_fit,
            trajectories[:, :-1, :],
            trajectories[:, 1:, :],
            inference_hyperparams,
            model_fit.compute_operator_predictions,
        )

        eqx.tree_serialise_leaves(model_path, model_fit)
        jnp.save(coeffs_path, dlds_coeffs)

    # fit hierarchical to coefficients
    model = eqx.tree_at(lambda model: model.primitives, model, model_fit)
    plot_Fs(model.primitives.F)
    fig = plot_coeff_spatial_maps(dlds_coeffs, trajectories, "c")
    fig.suptitle("dLDS inferred coefficients")
    coords = trajectories[:, :20, :]
    plt.show()

    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda model: model.G, filter_spec, replace=True)
    inference_backend = BPDNDFNoObsInference(prox=prox_l1_binary)
    inference_hyperparams = BPDNDFHyperparams(
        prox_hyperparams=[0, 0.1],
        prox_reweight_coeff=[200, 0],
        smooth_coeff=0.4,
    )
    trajectory_dict = {i: trajectories[i, :-1, :] for i in range(trajectories.shape[0])}
    dlds_coeff_dict = {i: dlds_coeffs[i] for i in range(dlds_coeffs.shape[0])}

    model = fit_hierarchical_mlps(
        trajectory_dict,
        dlds_coeff_dict,
        model,
        samples_per_snippet=20,
        num_snippets=10,
        max_iter=2000,
        lr_init=1,
        lr_end=1,
        inference_backend=inference_backend,
        inference_hyperparams=inference_hyperparams,
        filter_spec=filter_spec,
        prox_hyperparams_max=[0, 0.4],
    )

    inference_hyperparams = BPDNDFHyperparams(
        prox_hyperparams=[0.0, 0.4],
        prox_reweight_coeff=[200, 0],
        smooth_coeff=0.4,
    )
    coords = trajectories[:, :20, :]
    mlp_coeffs = inference_backend.infer_batch(
        model,
        coords,
        dlds_coeffs,
        inference_hyperparams,
        model.compute_coeff_predictions,
    )

    inference_hyperparams = BPDNDFHyperparams(
        prox_hyperparams=[0.0, 0.4],
        prox_reweight_coeff=[200, 0],
        smooth_coeff=0,
    )
    reinferred_mlp_coeffs = inference_backend.infer_batch(
        model,
        trajectories[:, :-1, :],
        trajectories[:, 1:, :],
        inference_hyperparams,
        model.compute_operator_predictions,
    )

    return model, dlds_coeffs, mlp_coeffs, reinferred_mlp_coeffs


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
    coords = trajectories[:, :20, :]

    plot_example_trajectories_with_coeffs(
        mlp_coeffs.reshape(-1, 20, model.num_operators),
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
    mlp_combined_dlds_coeff_predictions = model.combine_operator_predictions(
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
    coords = trajectories[:, :20, :].reshape(-1, 2)
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    fig, ax = plt.subplots(
        figsize=(2 + 3 * coeffs.shape[-1], 3), nrows=1, ncols=coeffs.shape[-1]
    )
    for i in range(coeffs.shape[-1]):
        plot = ax[i].scatter(
            coords[:, 0],
            coords[:, 1],
            c=coeffs[:, i],
            vmin=0,
            vmax=1.5,
            alpha=0.5,
            s=20,
            cmap="YlGn_r",
        )
        fig.colorbar(plot, ax=ax[i])
        ax[i].set_title(rf"spatial map of ${symbol}_{i}$")

    plt.tight_layout()
    return fig


@click.command(context_settings={"show_default": True, "help_option_names": ["-h", "--help"]})
@click.option("-n", "--num_trajectories", default=100, help="number of trajectories to sample")
@click.option("--dt", default=0.05, help="time step in seconds for Euler approximation")
@click.option("--tau", default=0.2, help="timescale in seconds of the flow field")
@click.option("--sigma", default=0.0, help="white noise variance")
@click.option("--T", "T", default=5.0, help="duration of sampled trajectories in seconds")
@click.option("--seed", default=0, help="random seed")
@click.option("--min_radius", default=0.0, help="minimum radius of initial conditions")
@click.option("--max_radius", default=2.0, help="maximum radius of initial conditions")
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    keys = jr.split(jr.key(args.seed), 6)
    simulation, trajectories = simulate_ring_attractor(args, keys)
    plotted_grid, plotted_simulation_flows = plot_ring_attractor_flow_and_trajectories(
        args, simulation, trajectories
    )

    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    model = HierarchicalDecomposedDynamics(
        num_nonlinear_operators=6,
        num_primitives=6,
        state_dim=2,
        primitive_type=DecomposedLinearDynamics,
        key=keys[3],
        layer_width=10,
        num_hidden_layers=4,
    )
    model, dlds_coeffs, mlp_coeffs, reinferred_mlp_coeffs = (
        fit_infer_hierarchical_model_all_stages(model, trajectory_dict, trajectories)
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


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
