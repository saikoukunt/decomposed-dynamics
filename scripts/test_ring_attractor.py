import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from plot_utils import plot_Fs
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
from decomposed_dynamics.fit_hierarchical import fit_hierarchical_mlps, infer_mlp_coeffs
from decomposed_dynamics.fitting import fit_no_obs
from decomposed_dynamics.inference import bpdn_inference_no_obs
from decomposed_dynamics.inference.base import NoObsInferenceHyperparams
from decomposed_dynamics.utils import prox_binary


def parse_args(argv: list):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n",
        "--num_trajectories",
        default=100,
        type=int,
        help="number of trajectories to sample",
    )
    parser.add_argument(
        "--dt",
        default=0.05,
        type=float,
        help="time step in seconds for Euler approximation",
    )
    parser.add_argument(
        "--tau",
        default=0.2,
        type=float,
        help="timescale in seconds of the flow field",
    )
    parser.add_argument(
        "--sigma",
        default=0.0,
        type=float,
        help="white noise variance",
    )
    parser.add_argument(
        "--T",
        default=5,
        type=float,
        help="duration of sampled trajectories in seconds",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--min_radius",
        default=0,
        type=float,
        help="random seed",
    )
    parser.add_argument(
        "--max_radius",
        default=2,
        type=float,
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

    # simulate trajectories and flow
    model = RingAttractorSimulation(dt=args.dt, tau=args.tau)
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
    grid, _ = plot_flow_field(ax[0], model, -args.max_radius, args.max_radius, 0.05)
    ax[0].set_title("Ring attractor flow field")

    grid, flows, _ = compute_flow_field(
        model, -args.max_radius, args.max_radius, 1 / 100
    )
    plot_speed(ax[0], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
    plot_speed(ax[1], grid, flows, vmin=0, vmax=0.25, alpha=0.5)

    keys = jr.split(jr.key(args.seed), 6)
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

    trajectories = model.sample_trajectories(
        x_0,
        args.num_trajectories,
        sigma=args.sigma,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )
    plot_trajectories(ax[1], trajectories[:300])
    ax[1].set_title("Many trajectories")

    # fit dLDS
    trajectory_dict = {i: trajectories[i] for i in range(trajectories.shape[0])}
    # model = DecomposedLinearDynamics(num_operators=6, num_latents=2, key=keys[3])
    model = HierarchicalDecomposedDynamics(
        num_nonlinear_operators=3,
        num_primitives=6,
        num_latents=2,
        primitive_type=DecomposedLinearDynamics,
        key=keys[3],
        layer_width=10,
        num_hidden_layers=4,
    )

    inference_hyperparams = NoObsInferenceHyperparams(l1_coeff=0.7)
    model_fit = fit_no_obs(
        trajectory_dict,
        model.primitives,
        samples_per_snippet=60,
        num_snippets=50,
        max_iter=100,
        lr_init=1,
        inference_hyperparams=inference_hyperparams,
        model_update_hyperparams=model.initialize_hyperparams(
            decorr_coeff=0.005, l1_coeff=0.01
        ).primitive_hyperparams,
    )
    C = bpdn_inference_no_obs(model_fit, trajectories, inference_hyperparams)

    plot_Fs(model_fit.F)
    plot_example_trials_with_C(C, trajectories, grid, flows, keys[4], args)
    plot_c_spatial_maps(C, trajectories)
    plt.show()

    # fit hierarchical to coefficients
    model = eqx.tree_at(lambda model: model.primitives, model, model_fit)

    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda model: model.G, filter_spec, replace=True)
    inference_hyperparams = NoObsInferenceHyperparams(
        l1_coeff=0.1, prox=prox_binary, l1_reweight_coeff=0
    )
    trajectory_dict = {i: trajectories[i, :-1, :] for i in range(trajectories.shape[0])}
    C_dict = {i: C[i] for i in range(C.shape[0])}
    C_orig = C.copy()

    model = fit_hierarchical_mlps(
        trajectory_dict,
        C_dict,
        model,
        samples_per_snippet=20,
        num_snippets=100,
        max_iter=5000,
        lr_init=1,
        inference_hyperparams=inference_hyperparams,
        filter_spec=filter_spec,
    )
    coords = trajectories[:, :20, :].reshape(-1, 2)

    inference_hyperparams = NoObsInferenceHyperparams(
        l1_coeff=0.4, prox=prox_binary, l1_reweight_coeff=0
    )
    C = infer_mlp_coeffs(model, coords, C.reshape(-1, 6), inference_hyperparams)

    plot_example_trials_with_C(
        C.reshape(-1, 20, 2), trajectories, grid, flows, keys[5], args
    )
    c_grid = model._compute_coeff_predictions_batched(model.G, coords)
    predicted_c = model.predict_next_state(coords, C, c_grid)

    plot_c_spatial_maps(C, trajectories)
    plot_c_spatial_maps(predicted_c, trajectories)
    plot_c_spatial_maps(c_grid[:, 0, :], trajectories)
    plot_c_spatial_maps(c_grid[:, 1, :], trajectories)
    plot_c_spatial_maps(c_grid[:, 2, :], trajectories)
    plt.figure()
    plot = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=c_grid[:, 0, 0] - C_orig.reshape(-1, 6)[:, 0],
        vmin=-0.1,
        vmax=0.1,
        alpha=0.5,
        s=20,
        cmap="YlGn_r",
    )
    plt.colorbar()
    plt.show()


def plot_example_trials_with_C(C, trajectories, grid, flows, key, args):
    fig, ax = plt.subplots(figsize=(12, 6), nrows=3, ncols=4)
    rand_inds = jr.randint(key, 6, 0, args.num_trajectories)
    for i in range(6):
        row_ind = i % 3
        col_ind = 2 * (i // 3)

        plot_speed(ax[row_ind, col_ind], grid, flows, vmin=0, vmax=0.25, alpha=0.5)
        plot_trajectories(
            ax[row_ind, col_ind], jnp.expand_dims(trajectories[rand_inds[i]], axis=0)
        )
        ax[row_ind, col_ind + 1].plot(C[rand_inds[i]])
    fig.suptitle("Example trajectories and inferred c")


def plot_c_spatial_maps(C, trajectories):
    coords = trajectories[:, :20, :].reshape(-1, 2)
    coeffs = C.reshape(-1, C.shape[-1])

    fig, ax = plt.subplots(figsize=(2 + 3 * C.shape[-1], 3), nrows=1, ncols=C.shape[-1])
    for i in range(C.shape[-1]):
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
        ax[i].set_title(rf"spatial map of $c_{i}$")

    plt.tight_layout()


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
