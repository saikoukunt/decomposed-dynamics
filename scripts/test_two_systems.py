import os
import sys

from tqdm import trange

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.devices()

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from plot_utils import plot_Fs, plot_X
from simulations.two_switching_systems import simulate_two_subsystems_no_obs

from decomposed_dynamics.dynamics_models import (
    AffineOperatorParams,
    DecomposedAffineDynamics,
    DecomposedLinearDynamics,
    LinearOperatorParams,
)
from decomposed_dynamics.fitting import (
    compute_dynamics_recon_loss,
    fit_no_obs,
    update_operators,
)
from decomposed_dynamics.inference import bpdn_inference_no_obs

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lr = 10.0
    with jax.default_device(jax.devices("cpu")[0]):
        X, C, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50, seed=seed)

        model = DecomposedAffineDynamics(num_operators=15, num_latents=8)
        operators = model.initialize_params(jr.key(seed))
        operators = fit_no_obs(X, model, operators, 200, 20, max_iter=200)
        plot_Fs(F)
        plot_Fs(operators.F)
        plt.show()
