import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.random as jr
import matplotlib.pyplot as plt
from plot_utils import plot_Fs
from simulations.two_switching_systems import simulate_two_subsystems_no_obs

from decomposed_dynamics.dynamics_models import (
    DecomposedAffineDynamics,
    DecomposedLinearDynamics,
)
from decomposed_dynamics.fitting import (
    fit_no_obs,
)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lr = 10.0
    with jax.default_device(jax.devices("cpu")[0]):
        X, C, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50, seed=seed)

        key = jr.key(seed)
        model = DecomposedAffineDynamics(num_operators=15, num_latents=8, key=key)
        model = fit_no_obs(X, model, 200, 20, max_iter=200)
        plot_Fs(F)
        plot_Fs(model.F)
        plt.show()
