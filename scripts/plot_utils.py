import matplotlib.pyplot as plt
import numpy as np


def plot_Fs(F):
    num_dynamics = F.shape[0]
    num_rows = (num_dynamics - 1) // 6 + 1
    num_cols = min(6, num_dynamics)
    plt.figure(figsize=(num_cols, num_rows))

    for i in range(F.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(F[i], vmin=-1, vmax=1)
        plt.axis("off")

    plt.tight_layout()


def plot_X(X):
    num_latents = X.shape[0]
    num_timepoints = X.shape[1]

    plt.figure(figsize=(20, 8))
    for i in range(num_latents):
        plt.subplot(num_latents, 1, i + 1)
        plt.plot(np.arange(num_timepoints), X[i, :].T)

    plt.tight_layout()


def plot_C(C):
    plt.figure(figsize=(10, 2))
    plt.imshow(C, aspect="auto", interpolation="none")

    plt.tight_layout()
