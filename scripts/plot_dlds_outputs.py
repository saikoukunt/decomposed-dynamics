import math
import numpy as np
import matplotlib.pyplot as plt

def plot_nonzero_slices(
    volume: np.ndarray, 
    title: str, 
    output_dir: str,
    cmap="coolwarm"
):
    """
    Plot only the non-zero slices of a 3D array along axis 0.

    Parameters
    ----------
    volume : np.ndarray
        3D array with shape (n_slices, height, width)
    cmap : str
        Matplotlib colormap.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")

    # Find non-zero slices
    nonzero_mask = np.any(volume != 0, axis=(1, 2))
    indices = np.where(nonzero_mask)[0]
    slices = volume[indices]

    n = len(slices)
    if n == 0:
        print("No non-zero slices found.")
        return

    # Choose a grid that's close to square
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig.suptitle(title)

    axes = axes.ravel()

    # Plot each slice
    for ax, idx, slice_2d in zip(axes, indices, slices):
        im = ax.matshow(slice_2d, cmap=cmap, aspect="auto")
        ax.set_title(f"Motif {idx}")
        # ax.axis("off")

        # Add a colorbar for this subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.strip()}.png")