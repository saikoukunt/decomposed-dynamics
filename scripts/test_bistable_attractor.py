import jax.numpy as jnp
import matplotlib

matplotlib.use("WebAgg")

import matplotlib.pyplot as plt
import numpy as np
from simulations.plot_utils import plot_flow_field

from simulations import RNN, bistable_RNN

if __name__ == "__main__":
    model = bistable_RNN(6, 9, 0.05)
    inputs = np.zeros(2)
    inputs[0] = 1
    inputs[1] = 2
    plot_flow_field(model, 0, 1, 0.05, jnp.array(inputs))
    plt.show()
