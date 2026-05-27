import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from plot_utils import plot_C, plot_Fs, plot_X
from simulations.two_switching_systems import simulate_two_subsystems_no_obs

from decomposed_dynamics.dlds import fit_no_obs, infer_no_obs_state

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with jax.default_device(jax.devices("cpu")[0]):
        X, C, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50, seed=seed)

        F_hat = fit_no_obs(X, 15, 200, 20, F_lr_init=10.0, c_l1_coeff=0.2)
        plot_Fs(F)
        plot_Fs(F_hat)

        plt.show()
