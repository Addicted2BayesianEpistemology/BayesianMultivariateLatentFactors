#!/usr/bin/env python3
"""
Generate Figure 2:
Effect of heteroscedastic scaling on Matérn 5/2 basis functions.

Panels:
(a) Original basis (first 10 basis functions)
(b) Modified basis after scaling KL components
(c) Sample paths from original and modified processes
(d) Power spectral densities of the sample paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram


# ------------------------- EDIT THIS CALLABLE ------------------------- #
def scaling_rule(k: np.ndarray) -> np.ndarray:
    """
    Scaling applied to the square roots of the eigenvalues / singular values.

    Parameters
    ----------
    k : np.ndarray
        1-indexed component indices.

    Returns
    -------
    np.ndarray
        Multiplicative scaling factors.
    """
    k = np.asarray(k, dtype=float)
    return 30.0 * (1.0 - np.exp(-k / 100.0))
# --------------------------------------------------------------------- #


def matern52_covariance(x: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    """Matérn 5/2 covariance matrix on a 1D grid."""
    dx = np.abs(x[:, None] - x[None, :]) / lengthscale
    sqrt5 = np.sqrt(5.0)
    return variance * (1.0 + sqrt5 * dx + (5.0 / 3.0) * dx**2) * np.exp(-sqrt5 * dx)


def ensure_parent_dir(filepath: str | Path) -> Path:
    path = Path(filepath).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compute_basis_from_covariance(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenbasis B = U diag(sqrt(evals)) for a symmetric PSD covariance matrix.

    Returns
    -------
    evals : np.ndarray
        Eigenvalues in descending order.
    U : np.ndarray
        Eigenvectors in descending order.
    B : np.ndarray
        Basis matrix whose columns are scaled eigenvectors.
    """
    evals, U = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    U = U[:, order]
    B = U * np.sqrt(evals)
    return evals, U, B


def draw_realizations(B: np.ndarray, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw standard-normal coefficients Z and realizations Y = BZ.
    """
    Z = rng.standard_normal((B.shape[1], n_samples))
    Y = B @ Z
    return Z, Y


def compute_psd_matrix(Y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a periodogram for each column of Y.
    """
    freqs, p0 = periodogram(Y[:, 0], fs=fs)
    psd = np.empty((len(freqs), Y.shape[1]), dtype=float)
    psd[:, 0] = p0
    for j in range(1, Y.shape[1]):
        _, psd[:, j] = periodogram(Y[:, j], fs=fs)
    return freqs, psd


def style_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    label_fs: int,
    tick_fs: int,
) -> None:
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)


def plot_figure2(
    x: np.ndarray,
    B_orig: np.ndarray,
    B_mod: np.ndarray,
    Y_orig_plot: np.ndarray,
    Y_mod_plot: np.ndarray,
    freqs: np.ndarray,
    psd_orig_plot: np.ndarray,
    psd_mod_plot: np.ndarray,
    psd_orig_mean: np.ndarray,
    psd_mod_mean: np.ndarray,
    outpath: str | Path,
    label_fs: int = 18,
    tick_fs: int = 15,
    legend_fs: int = 15,
) -> None:
    """
    Create the 2x2 Figure 2 dashboard with larger fonts and labeled axes.
    """
    outpath = ensure_parent_dir(outpath)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # (a) Original basis
    ax = axes[0, 0]
    for j in range(min(10, B_orig.shape[1])):
        ax.plot(x, B_orig[:, j], linewidth=1.8)
    style_axis(
        ax,
        xlabel=r"$t$",
        ylabel=r"$b(t)$",
        label_fs=label_fs,
        tick_fs=tick_fs,
    )

    # (b) Modified basis
    ax = axes[0, 1]
    for j in range(min(10, B_mod.shape[1])):
        ax.plot(x, B_mod[:, j], linewidth=1.8)
    style_axis(
        ax,
        xlabel=r"$t$",
        ylabel=r"$\tilde{b}(t)$",
        label_fs=label_fs,
        tick_fs=tick_fs,
    )

    # (c) Realizations
    ax = axes[1, 0]
    n_plot = Y_orig_plot.shape[1]
    for j in range(n_plot):
        ax.plot(x, Y_orig_plot[:, j], linestyle="-", linewidth=1.2, alpha=0.55, color="C0")
    for j in range(n_plot):
        ax.plot(x, Y_mod_plot[:, j], linestyle="--", linewidth=1.2, alpha=0.55, color="C1")

    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], linestyle="-", linewidth=2.5, color="C0"),
        Line2D([0], [0], linestyle="--", linewidth=2.5, color="C1"),
    ]
    ax.legend(
        legend_lines,
        ["Original GP", "Modified GP"],
        fontsize=legend_fs,
        loc="best",
        frameon=True,
    )
    style_axis(
        ax,
        xlabel=r"$t$",
        ylabel=r"$g(t)$",
        label_fs=label_fs,
        tick_fs=tick_fs,
    )

    # (d) PSDs
    ax = axes[1, 1]
    positive_freqs = freqs > 0.0
    f = freqs[positive_freqs]

    for j in range(psd_orig_plot.shape[1]):
        ax.loglog(
            f,
            np.maximum(psd_orig_plot[positive_freqs, j], 1e-16),
            linestyle="-",
            linewidth=1.0,
            alpha=0.22,
            color="C0",
        )
    for j in range(psd_mod_plot.shape[1]):
        ax.loglog(
            f,
            np.maximum(psd_mod_plot[positive_freqs, j], 1e-16),
            linestyle="--",
            linewidth=1.0,
            alpha=0.22,
            color="C1",
        )

    ax.loglog(
        f,
        np.maximum(psd_orig_mean[positive_freqs], 1e-16),
        linestyle="-",
        linewidth=2.6,
        color="C0",
        label="Original mean PSD",
    )
    ax.loglog(
        f,
        np.maximum(psd_mod_mean[positive_freqs], 1e-16),
        linestyle="--",
        linewidth=2.6,
        color="C1",
        label="Modified mean PSD",
    )

    ax.legend(fontsize=legend_fs, loc="best", frameon=True)
    style_axis(
        ax,
        xlabel=r"$\omega$",
        ylabel=r"$\mathcal{S}(\omega)$",
        label_fs=label_fs,
        tick_fs=tick_fs,
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2 with larger fonts and labeled subplots.")
    parser.add_argument("--N", type=int, default=100, help="Number of grid points on [-1, 1].")
    parser.add_argument("--variance", type=float, default=1.0, help="Kernel variance.")
    parser.add_argument("--ell", type=float, default=0.3, help="Matérn lengthscale.")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of realizations to plot.")
    parser.add_argument("--n_psd_samples", type=int, default=500, help="Number of realizations used to average PSDs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--fs_factor", type=float, default=0.2, help="Sampling-frequency multiplier used in the original script.")
    parser.add_argument("--output", type=str, default="figure2_dashboard.png", help="Output image path.")

    # Font controls
    parser.add_argument("--label_fs", type=int, default=20, help="Axis-label font size.")
    parser.add_argument("--tick_fs", type=int, default=19, help="Tick-label font size.")
    parser.add_argument("--legend_fs", type=int, default=19, help="Legend font size.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Grid and covariance
    x = np.linspace(-1.0, 1.0, args.N)
    Sigma = matern52_covariance(x, variance=args.variance, lengthscale=args.ell)

    # Original basis
    evals, U, B_orig = compute_basis_from_covariance(Sigma)

    # Modified basis: scale all components
    k = np.arange(1, len(evals) + 1)
    scaled_sqrt_evals = np.sqrt(evals) * scaling_rule(k)
    B_mod = U * scaled_sqrt_evals

    # Paired realizations for plotting
    Z_plot = rng.standard_normal((B_orig.shape[1], args.n_samples))
    Y_orig_plot = B_orig @ Z_plot
    Y_mod_plot = B_mod @ Z_plot

    # More realizations for PSD averaging
    Z_psd = rng.standard_normal((B_orig.shape[1], args.n_psd_samples))
    Y_orig_psd = B_orig @ Z_psd
    Y_mod_psd = B_mod @ Z_psd

    # PSDs
    dt = (x[-1] - x[0]) / (args.N - 1)
    fs = args.fs_factor / dt

    freqs, psd_orig_plot = compute_psd_matrix(Y_orig_plot, fs=fs)
    _, psd_mod_plot = compute_psd_matrix(Y_mod_plot, fs=fs)
    _, psd_orig_all = compute_psd_matrix(Y_orig_psd, fs=fs)
    _, psd_mod_all = compute_psd_matrix(Y_mod_psd, fs=fs)

    psd_orig_mean = psd_orig_all.mean(axis=1)
    psd_mod_mean = psd_mod_all.mean(axis=1)

    plot_figure2(
        x=x,
        B_orig=B_orig[:, :10],
        B_mod=B_mod[:, :10],
        Y_orig_plot=Y_orig_plot,
        Y_mod_plot=Y_mod_plot,
        freqs=freqs,
        psd_orig_plot=psd_orig_plot,
        psd_mod_plot=psd_mod_plot,
        psd_orig_mean=psd_orig_mean,
        psd_mod_mean=psd_mod_mean,
        outpath=args.output,
        label_fs=args.label_fs,
        tick_fs=args.tick_fs,
        legend_fs=args.legend_fs,
    )

    print(f"Figure saved to: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()