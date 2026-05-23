import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from decomposed_dynamics.dlds import infer_no_obs_state, fit_no_obs
from experiments.scripts.plot_utils import plot_C, plot_Fs, plot_X
from experiments.simulations.two_switching_systems import simulate_two_subsystems_no_obs

if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        X, C, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)

        F_hat = fit_no_obs(X, 15, 200, 10, F_lr_init=5.0)
        plot_Fs(F)
        plot_Fs(F_hat)


        plt.show()
