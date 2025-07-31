from __future__ import annotations

"""flow.py

Utility functions for training a *normalising flow* on star–particle
phase–space data (3D position + 3D velocity) stored in an HDF5 file.

Typical usage::

    from flow import (
        read_h5_to_dict,
        prepare_star_particle_dataset,
        train_flow,
        sample,
    )

    star_p = read_h5_to_dict("particles.h5")
    x_std, med, std = prepare_star_particle_dataset(star_p)

    key = jr.key(0)
    flow, _ = train_flow(key, x_std)

    new_samples = sample(flow, key, n_samples=len(star_p["pos3"]), median=med, std=std)
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
# from flowjax.bijections import RationalQuadraticSpline  # Optional transformer

################################################################################
# Data utilities
################################################################################


def read_h5_to_dict(filename: str | Path) -> dict[str, np.ndarray]:
    """Recursively read *all* datasets from an HDF5 file.

    The HDF5 group hierarchy is flattened: the full ("/"-separated) path of
    every dataset becomes the key inside the returned mapping.

    Parameters
    ----------
    filename
        Path to the ``.h5`` file on disk.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping "dataset path" → N-D *NumPy* array.
    """
    filename = Path(filename)
    data: dict[str, np.ndarray] = {}

    def _recursively_load(h5obj: h5py.Group | h5py.File, prefix: str = "") -> None:
        for key in h5obj:
            item = h5obj[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                data[path] = item[()]
            elif isinstance(item, h5py.Group):
                _recursively_load(item, path)

    with h5py.File(filename, "r") as f:
        _recursively_load(f)

    return data


def prepare_star_particle_dataset(
    star_particles: dict[str, np.ndarray],
    sample_size: int = 5_000,
    seed: int | None = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract, *subsample* and *standardise* (pos3, vel3) star-particle data.

    1. Concatenate the three-dimensional position and velocity arrays into a 6-D
       feature vector for each particle.
    2. Optionally subsample to *sample_size* particles (without replacement).
    3. Standardise the data: ``x' = (x − median) / std``.

    Returns
    -------
    x_std, median, std
        The standardised data along with the *median* and *std* used for later
        de-normalisation.
    """
    pos3 = star_particles["pos3"]
    vel3 = star_particles["vel3"]

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if sample_size and len(pos3) > sample_size:
        indices = rng.choice(len(pos3), size=sample_size, replace=False)
        pos3 = pos3[indices]
        vel3 = vel3[indices]

    x = np.hstack((pos3, vel3))

    median = np.median(x, axis=0)
    std = x.std(axis=0)
    x_std = (x - median) / std

    return x_std, median, std

################################################################################
# Flow utilities
################################################################################


def build_flow(key: jr.KeyArray, input_dim: int):
    """Create a *masked autoregressive flow* with a *Normal* base distribution."""
    return masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(input_dim)),
        # Uncomment to change the transformer, e.g.
        # transformer=RationalQuadraticSpline(knots=8, interval=4),
    )


def train_flow(
    rng_key: jr.KeyArray,
    x: np.ndarray | jnp.ndarray,
    learning_rate: float = 1.2e-4,
):
    """Fit a normalising flow to the dataset *x*.

    The data is expected to be *standardised*.  The function returns the
    trained *flow* and the list of *training losses*.
    """
    flow = build_flow(rng_key, input_dim=x.shape[1])
    rng_key, subkey = jr.split(rng_key)
    flow, losses = fit_to_data(subkey, flow, x, learning_rate=learning_rate)
    return flow, losses


def sample(
    flow,
    rng_key: jr.KeyArray,
    n_samples: int,
    median: np.ndarray,
    std: np.ndarray,
):
    """Draw *n_samples* from the flow and *de-standardise* them."""
    samples_std = flow.sample(rng_key, (n_samples,))
    return samples_std * std + median

################################################################################
# Visualisation helpers
################################################################################


def plot_marginals(
    original: np.ndarray,
    generated: np.ndarray,
    dims: tuple[int, ...] | None = None,
    bins: int = 100,
    labels: tuple[str, ...] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Quickly compare 1-D marginal PDFs (histograms) before/after training.

    Parameters
    ----------
    original, generated
        Arrays with shape (N, D).
    dims
        Tuple of column indices to plot.  If ``None`` (default) the first two
        dimensions are shown.
    bins
        Number of histogram bins.
    labels
        Optionally provide axis labels for each dimension.
    """
    if dims is None:
        dims = (0, 1)
    n_rows = len(dims)
    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 3 * n_rows), tight_layout=True)

    if n_rows == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        ax.hist(original[:, d], bins=bins, density=True, alpha=0.6, label="original")
        ax.hist(generated[:, d], bins=bins, density=True, alpha=0.6, label="generated")
        ax.set_ylabel("pdf")
        if labels and d < len(labels):
            ax.set_xlabel(labels[d])
        else:
            ax.set_xlabel(f"dim {d}")
        ax.legend()

    if save_path is not None:
        fig.savefig(save_path)
        print(f"Saved comparison figure to {save_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print("Could not display plot:", e)
            # fall back to saving
            fallback = Path(save_path or "marginals.png")
            fig.savefig(fallback)
            print("Figure saved to", fallback)


