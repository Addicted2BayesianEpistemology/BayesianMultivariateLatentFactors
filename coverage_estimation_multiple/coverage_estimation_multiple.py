# -*- coding: utf-8 -*-
"""
Version 2: Refactored simulation -> fit -> plot -> coverage + quadratic error pipeline.

Key change vs Version 1 (your script above):
- Truth/data-generating basis is now **B-spline**, not GP.
- The B-spline basis used for generation has **fewer basis functions than grid points**.
- Everything else stays the same:
  - Two fitted BLF models are still run: one with GP basis, one with B-spline basis.
  - PLS + FPLS (GP and B-spline FunctionalData) with bootstrap are still run.
  - Coverage + quadratic error (ISE via trapz) are still produced.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple, Sequence, Optional

import numpy as np
import numpy.linalg as npl
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import MultivariateBLF.MultivariateBLF as mvblf
from MultivariateBLF.visualization import align_true_factors_to_varimaxrsp
from MultivariateBLF.PartialLeastSquares import (
    PartialLeastSquares,
    FunctionalPartialLeastSquares,
    bootstrap_prediction_uncertainty,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def _default_x1() -> np.ndarray:
    return np.linspace(0.0, 1.0, 15)

def _default_x2() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 25)

def _default_seeds() -> tuple[int, ...]:
    return tuple(range(1, 6))  # change to 1..100 for full runs

@dataclass(frozen=True)
class ExperimentConfig:
    # Scenario identity / output
    output_root: Path = Path("Figures")
    scenario_name: str = "default"

    # Grids
    x1: np.ndarray = field(default_factory=_default_x1)
    x2: np.ndarray = field(default_factory=_default_x2)

    # Dimensions
    N: int = 30
    K: int = 3

    # Seeds
    seeds: Sequence[int] = field(default_factory=_default_seeds)

    # Data generating hyperparameters
    a1: float = 2.0
    a2: float = 2.1
    dfLambda: float = 7.0
    taus_alpha1: float = 5.0
    taus_scales1: float = 1.0
    taus_alpha2: float = 5.0
    taus_scales2: float = 1.0
    psi12_alpha: float = 11.0
    psi12_beta: float = 0.1
    psi22_alpha: float = 11.0
    psi22_beta: float = 0.1

    # Truth (generation) basis: B-spline with fewer basis fns than points
    bspline_true_degree: int = 3
    bspline_true_n_basis_mode1: Optional[int] = None  # filled automatically if None
    bspline_true_n_basis_mode2: Optional[int] = None  # filled automatically if None

    # GP basis configuration (used for one of the fitted models, and for GP FunctionalData)
    gp_type: str = "matern5/2"
    gp_lengthscale: float = 0.2
    gp_variance: float = 1.0

    # BLF model configuration
    MODEL_D: int = 2
    heteroscedastic_thetas: bool = True
    MGPS: bool = True
    local_shrinkage_LatentFactors: bool = True
    absence_of_psi2: List[bool] = field(default_factory=lambda: [False, False])

    # Stan config
    n_chains: int = 16
    iter_warmup: int = 3000
    iter_sampling: int = 500
    thin: int = 1
    max_treedepth: int = 12

    # Interval / coverage settings
    ci_level: float = 80.0

    # Bootstrap settings for PLS/FPLS
    n_boot_pls: int = 200


def _choose_default_true_bspline_n_basis(n_points: int, degree: int) -> int:
    """
    Pick a sensible default strictly less than n_points and > degree.
    """
    # Start with about half the points (but at least degree+2), then cap at n_points-1
    cand = max(degree + 2, int(np.floor(n_points / 2)))
    cand = min(cand, n_points - 1)
    if cand <= degree:
        cand = degree + 2
    if cand >= n_points:
        cand = n_points - 1
    return cand

def _cfg_with_defaults(cfg: ExperimentConfig) -> ExperimentConfig:
    d = dict(cfg.__dict__)

    deg = int(d["bspline_true_degree"])
    n1 = d.get("bspline_true_n_basis_mode1", None)
    n2 = d.get("bspline_true_n_basis_mode2", None)

    if n1 is None:
        n1 = _choose_default_true_bspline_n_basis(len(cfg.x1), deg)
    if n2 is None:
        n2 = _choose_default_true_bspline_n_basis(len(cfg.x2), deg)

    # Ensure valid
    n1 = int(n1)
    n2 = int(n2)
    if not (n1 < len(cfg.x1)):
        raise ValueError("bspline_true_n_basis_mode1 must be < len(x1)")
    if not (n2 < len(cfg.x2)):
        raise ValueError("bspline_true_n_basis_mode2 must be < len(x2)")
    if n1 <= deg:
        raise ValueError("bspline_true_n_basis_mode1 must be > bspline_true_degree")
    if n2 <= deg:
        raise ValueError("bspline_true_n_basis_mode2 must be > bspline_true_degree")

    d["bspline_true_n_basis_mode1"] = n1
    d["bspline_true_n_basis_mode2"] = n2

    # absence_of_psi2 default safety
    if d.get("absence_of_psi2", None) is None:
        d["absence_of_psi2"] = [False, False]

    return ExperimentConfig(**d)

# ---------------------------------------------------------------------
# Basis utilities
# ---------------------------------------------------------------------

def make_gp_basis(coords: np.ndarray, gp_type: str, gp_lengthscale: float, gp_variance: float) -> np.ndarray:
    f = mvblf.FunctionalData(coords=coords, data=np.zeros((len(coords), 1)))
    f.set_GP_basis(
        n_basis=len(coords),
        gaussian_process=gp_type,
        gp_lengthscale=gp_lengthscale,
        gp_variance=gp_variance,
    )
    return f._basis_matrix


def _bspline_knots(domain: Tuple[float, float], n_basis: int, degree: int) -> np.ndarray:
    a, b = map(float, domain)
    k = int(degree)
    n = int(n_basis)

    if not a < b:
        raise ValueError("domain must satisfy a < b")
    if k < 0:
        raise ValueError("degree must be >= 0")
    if n <= k:
        raise ValueError("n_basis must be > degree")

    m = n + k + 1
    interior = m - 2 * (k + 1)
    if interior > 0:
        interior_knots = np.linspace(a, b, interior + 2)[1:-1]
        t = np.concatenate((np.full(k + 1, a), interior_knots, np.full(k + 1, b)))
    else:
        t = np.concatenate((np.full(k + 1, a), np.full(k + 1, b)))
    return t


def make_bspline_basis_matrix(coords: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """
    Returns B-spline basis matrix evaluated at coords:
      B has shape (M, n_basis), where M = len(coords).
    """
    x = np.asarray(coords, dtype=float).reshape(-1)
    M = x.size
    a, b = float(x.min()), float(x.max())
    n = int(n_basis)
    k = int(degree)

    t = _bspline_knots((a, b), n, k)

    # Zeroth-degree
    N0 = np.zeros((n, M), dtype=float)
    for i in range(n):
        mask = (t[i] <= x) & (x < t[i + 1])
        if i == n - 1:
            mask |= (x == b)
        N0[i] = mask.astype(float)

    N = N0
    for p in range(1, k + 1):
        Np = np.zeros_like(N)
        for i in range(n):
            d1 = t[i + p] - t[i]
            if d1 != 0:
                Np[i] += ((x - t[i]) / d1) * N[i]
            d2 = t[i + p + 1] - t[i + 1]
            if d2 != 0 and i + 1 < n:
                Np[i] += ((t[i + p + 1] - x) / d2) * N[i + 1]
        N = Np

    # Return as (M, n)
    return N.T


def bspline_basis_functions(
    domain: Tuple[float, float],
    n_basis: int,
    degree: int,
) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Kept for the FunctionalData B-spline fit, as in Version 1.
    """
    a, b = map(float, domain)
    if not a < b:
        raise ValueError("domain must satisfy a < b")

    k = int(degree)
    if k < 0:
        raise ValueError("degree must be >= 0")

    n = int(n_basis)
    if n <= k:
        raise ValueError("n_basis must be > degree")

    t = _bspline_knots((a, b), n, k)

    def eval_all(x):
        x_arr = np.asarray(x, dtype=float)
        x_flat = x_arr.reshape(-1)
        m_pts = x_flat.size

        N0 = np.zeros((n, m_pts))
        for i in range(n):
            mask = (t[i] <= x_flat) & (x_flat < t[i + 1])
            if i == n - 1:
                mask |= (x_flat == b)
            N0[i] = mask.astype(float)

        N = N0
        for p in range(1, k + 1):
            Np = np.zeros_like(N)
            for i in range(n):
                d1 = t[i + p] - t[i]
                if d1 != 0:
                    Np[i] += ((x_flat - t[i]) / d1) * N[i]
                d2 = t[i + p + 1] - t[i + 1]
                if d2 != 0 and i + 1 < n:
                    Np[i] += ((t[i + p + 1] - x_flat) / d2) * N[i + 1]
            N = Np

        return N.reshape((n,) + x_arr.shape)

    basis: List[Callable] = []
    for i in range(n):
        basis.append(lambda x, i=i: eval_all(x)[i])
    return basis


def build_functional_data_gp(coords: np.ndarray, data: np.ndarray, cfg: ExperimentConfig) -> mvblf.FunctionalData:
    fd = mvblf.FunctionalData(coords=coords, data=data)
    fd.set_GP_basis(
        n_basis=len(coords),
        gaussian_process=cfg.gp_type,
        gp_lengthscale=cfg.gp_lengthscale,
        gp_variance=cfg.gp_variance,
    )
    return fd


def build_functional_data_bspline(coords: np.ndarray, data: np.ndarray, degree: int = 3) -> mvblf.FunctionalData:
    fd = mvblf.FunctionalData(coords=coords, data=data)
    fd.set_functional_basis(
        basis_functions=bspline_basis_functions(
            domain=(coords.min(), coords.max()),
            n_basis=len(coords),
            degree=degree,
        )
    )
    return fd


# ---------------------------------------------------------------------
# Simulation (truth generated from B-splines with n_basis_true < n_points)
# ---------------------------------------------------------------------

def simulate_once(seed: int, cfg: ExperimentConfig, B1_true: np.ndarray, B2_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    N, K = cfg.N, cfg.K
    nb1, nb2 = B1_true.shape[1], B2_true.shape[1]

    eta = rng.normal(0.0, 1.0, size=(N, K))

    delta1 = rng.gamma(shape=cfg.a1, scale=1.0, size=(1,))
    delta = rng.gamma(shape=cfg.a2, scale=1.0, size=(K - 1,))

    tau = [delta1]
    for k in range(K - 1):
        tau.append(tau[k] * delta[k])
    tau = np.array(tau).T  # (1, K)

    Lambda1 = rng.standard_t(df=cfg.dfLambda, size=(nb1, K)) * (1.0 / np.sqrt(tau))
    Lambda2 = rng.standard_t(df=cfg.dfLambda, size=(nb2, K)) * (1.0 / np.sqrt(tau))

    precisions_thetas_1 = rng.gamma(shape=cfg.taus_alpha1, scale=cfg.taus_scales1, size=(nb1,))
    precisions_thetas_2 = rng.gamma(shape=cfg.taus_alpha2, scale=cfg.taus_scales2, size=(nb2,))

    u1 = rng.normal(0.0, 1.0 / np.sqrt(precisions_thetas_1[:, None]), size=(nb1, N))
    u2 = rng.normal(0.0, 1.0 / np.sqrt(precisions_thetas_2[:, None]), size=(nb2, N))

    # Domain data generated from B-spline truth basis
    Y1 = B1_true @ (Lambda1 @ eta.T + u1)
    Y2 = B2_true @ (Lambda2 @ eta.T + u2)

    psi12 = 1.0 / rng.gamma(shape=cfg.psi12_alpha, scale=1.0 / cfg.psi12_beta)
    psi22 = 1.0 / rng.gamma(shape=cfg.psi22_alpha, scale=1.0 / cfg.psi22_beta)

    Y1 += rng.normal(0.0, np.sqrt(psi12), size=Y1.shape)
    Y2 += rng.normal(0.0, np.sqrt(psi22), size=Y2.shape)

    truth = {"Lambda1": Lambda1, "Lambda2": Lambda2, "B1": B1_true, "B2": B2_true}
    return Y1, Y2, truth


# ---------------------------------------------------------------------
# Alignment + metrics helpers (PLS/FPLS)
# ---------------------------------------------------------------------

def _orthogonal_procrustes_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A0 = np.nan_to_num(A, nan=0.0)
    B0 = np.nan_to_num(B, nan=0.0)
    M = A0.T @ B0
    U, _, Vt = npl.svd(M, full_matrices=False)
    return U @ Vt


def _align_truth_to_reference(truth_mat: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    R = _orthogonal_procrustes_rotation(truth_mat, ref_mat)
    return truth_mat @ R


def _align_stack_to_reference(stack: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    out = np.full_like(stack, np.nan)
    for b in range(stack.shape[0]):
        Ab = stack[b]
        if np.isnan(Ab).all():
            continue
        Rb = _orthogonal_procrustes_rotation(Ab, ref_mat)
        out[b] = Ab @ Rb
    return out


def _compute_curve_metrics_from_stack(
    seed: int,
    coords: np.ndarray,
    true_aligned: np.ndarray,     # (M,K)
    stack_aligned: np.ndarray,    # (B,M,K)
    mode_id: int,
    method_name: str,
    lower_q: float,
    upper_q: float,
    ci_level: float,
) -> list[dict]:
    records = []
    K = true_aligned.shape[1]
    M = true_aligned.shape[0]
    for k in range(K):
        draws = stack_aligned[:, :, k]  # (B,M)
        lower = np.nanpercentile(draws, lower_q, axis=0)
        upper = np.nanpercentile(draws, upper_q, axis=0)
        mean_curve = np.nanmean(draws, axis=0)

        true_curve = true_aligned[:, k]
        within = np.logical_and(true_curve >= lower, true_curve <= upper)

        within_count = int(np.nansum(within))
        coverage_pct = 100.0 * within_count / M

        quad_error = float(np.trapz((mean_curve - true_curve) ** 2, x=coords))
        mse = float(np.nanmean((mean_curve - true_curve) ** 2))
        rmse = float(np.sqrt(mse))

        records.append(
            dict(
                seed=seed,
                method=method_name,
                mode=mode_id,
                factor=k + 1,
                n_coords=int(M),
                ci_level=ci_level,
                lower_q=lower_q,
                upper_q=upper_q,
                within_count=within_count,
                coverage_pct=coverage_pct,
                quad_error=quad_error,
                mse=mse,
                rmse=rmse,
            )
        )
    return records


def _extract_fpls_domain_weights_singleX(
    fpls_model: FunctionalPartialLeastSquares,
    X_fd: mvblf.FunctionalData,
) -> tuple[np.ndarray, np.ndarray]:
    raw_x_weights = fpls_model.model.x_weights_  # (p_x, K)
    raw_y_weights = fpls_model.model.y_weights_  # (p_y, K)

    Bx = X_fd._basis_matrix
    Wx = fpls_model._Wx_list[0]
    w_x_domain = Bx @ (Wx @ raw_x_weights)

    By = fpls_model._Y_basis
    Wy = fpls_model._compute_gram_sqrt(By)
    w_y_domain = By @ (Wy @ raw_y_weights)

    return w_x_domain, w_y_domain


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def _write_method_summaries(df: pd.DataFrame, out_prefix: Path) -> None:
    if df.empty:
        return

    grouped = df.groupby(["mode", "factor"], as_index=False).agg(
        mean_coverage_pct=("coverage_pct", "mean"),
        std_coverage_pct=("coverage_pct", "std"),
        pooled_within=("within_count", "sum"),
        pooled_coords=("n_coords", "sum"),
        n_seeds=("seed", "nunique"),
        mean_quad_error=("quad_error", "mean"),
        std_quad_error=("quad_error", "std"),
        pooled_quad_error=("quad_error", "sum"),
    )
    grouped["pooled_coverage_pct"] = 100.0 * grouped["pooled_within"] / grouped["pooled_coords"]
    grouped["pooled_mse"] = grouped["pooled_quad_error"] / grouped["pooled_coords"]

    overall = pd.DataFrame(
        [{
            "mean_coverage_pct_over_cells": df["coverage_pct"].mean(),
            "std_coverage_pct_over_cells": df["coverage_pct"].std(),
            "pooled_coverage_pct_overall": 100.0 * df["within_count"].sum() / df["n_coords"].sum(),
            "n_cells": len(df),
            "n_seeds_completed": df["seed"].nunique(),
            "mean_quad_error_over_cells": df["quad_error"].mean(),
            "std_quad_error_over_cells": df["quad_error"].std(),
            "pooled_quad_error_overall": df["quad_error"].sum(),
            "pooled_mse_overall": df["quad_error"].sum() / df["n_coords"].sum(),
        }]
    )

    grouped.to_csv(Path(str(out_prefix) + "_summary_by_mode_factor.csv"), index=False)
    overall.to_csv(Path(str(out_prefix) + "_summary_overall.csv"), index=False)


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------

def _plot_band_vs_truth(
    ax: plt.Axes,
    coords: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    truth: np.ndarray,
    title: str,
    ylabel: str,
    show_legend: bool,
    ci_level: float,
    center_label: str,
):
    ax.fill_between(coords, lower, upper, alpha=0.25, label=f"PI {ci_level}%")
    ax.plot(coords, center, linewidth=2, label=center_label)
    ax.plot(coords, truth, color="red", linestyle="--", linewidth=3, label="True (aligned)")
    ax.set_title(title)
    ax.set_xlabel("coord")
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(loc="best")


# ---------------------------------------------------------------------
# Model fitting blocks
# ---------------------------------------------------------------------

def _fit_blf_model(seed: int, cfg: ExperimentConfig, modes: list[mvblf.FunctionalData]) -> mvblf.MvtBLF:
    model = mvblf.MvtBLF(
        n_components=cfg.K,
        D=cfg.MODEL_D,
        heteroscedastic_thetas=cfg.heteroscedastic_thetas,
        MGPS=cfg.MGPS,
        local_shrinkage_LatentFactors=cfg.local_shrinkage_LatentFactors,
        absence_of_psi2=cfg.absence_of_psi2,
        n_chains=cfg.n_chains,
        iter_warmup=cfg.iter_warmup,
        iter_sampling=cfg.iter_sampling,
        thin=cfg.thin,
        max_treedepth=cfg.max_treedepth,
        stan_seed=seed,
        a_psi=[cfg.psi12_alpha, cfg.psi22_alpha],
        b_psi=[cfg.psi12_beta, cfg.psi22_beta],
        a_tau=[cfg.taus_alpha1, cfg.taus_alpha2],
        b_tau=[1.0 / cfg.taus_scales1, 1.0 / cfg.taus_scales2],
        a1=cfg.a1,
        a2=cfg.a2,
        b_lambdavar=1.0,
        nu=cfg.dfLambda,
    )
    model.fit(X=np.zeros((0, cfg.N)), modes=modes)
    return model


def _blf_posterior_records_and_plot(
    seed: int,
    cfg: ExperimentConfig,
    seed_dir: Path,
    posteriorBLambda1,
    posteriorBLambda2,
    BLambda1_true_aligned: np.ndarray,
    BLambda2_true_aligned: np.ndarray,
    fig_name: str,
    csv_name: str,
):
    lower_q = (100 - cfg.ci_level) / 2
    upper_q = 100 - lower_q

    fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
    records: list[dict] = []

    for d in range(2):
        coords = cfg.x1 if d == 0 else cfg.x2
        post = posteriorBLambda1 if d == 0 else posteriorBLambda2
        true_aligned = BLambda1_true_aligned if d == 0 else BLambda2_true_aligned

        for k in range(cfg.K):
            ax = axs[k, d]

            post_k = (
                post.sel(K_rsp=k)
                .stack(sample=("chain", "draw"))
                .transpose("sample", "M")
                .values
            )

            lower = np.percentile(post_k, lower_q, axis=0)
            upper = np.percentile(post_k, upper_q, axis=0)
            median = np.percentile(post_k, 50, axis=0)

            trueBL = true_aligned[:, k]
            post_mean = post_k.mean(axis=0)

            quad_error = float(np.trapz((post_mean - trueBL) ** 2, x=coords))
            mse = float(np.mean((post_mean - trueBL) ** 2))
            rmse = float(np.sqrt(mse))

            within = np.logical_and(trueBL >= lower, trueBL <= upper)
            within_count = int(within.sum())
            n_coords = int(len(trueBL))
            coverage_pct = 100.0 * within_count / n_coords

            records.append(
                dict(
                    seed=seed,
                    mode=d + 1,
                    factor=k + 1,
                    n_coords=n_coords,
                    ci_level=cfg.ci_level,
                    lower_q=lower_q,
                    upper_q=upper_q,
                    within_count=within_count,
                    coverage_pct=coverage_pct,
                    quad_error=quad_error,
                    mse=mse,
                    rmse=rmse,
                )
            )

            _plot_band_vs_truth(
                ax=ax,
                coords=coords,
                lower=lower,
                upper=upper,
                center=median,
                truth=trueBL,
                title=f"Mode {d+1}, Factor {k+1} | Coverage={coverage_pct:.1f}%",
                ylabel="BLambda",
                show_legend=(k == 0),
                ci_level=cfg.ci_level,
                center_label="Posterior median",
            )

    fig.tight_layout()
    fig.savefig(seed_dir / fig_name, dpi=200)
    plt.close(fig)

    pd.DataFrame(records).to_csv(seed_dir / csv_name, index=False)
    return records


def _pls_fpls_records_and_plots(
    seed: int,
    cfg: ExperimentConfig,
    seed_dir: Path,
    f1_gp: mvblf.FunctionalData,
    f2_gp: mvblf.FunctionalData,
    f1_bs: mvblf.FunctionalData,
    f2_bs: mvblf.FunctionalData,
    truth: dict,
):
    lower_q = (100 - cfg.ci_level) / 2
    upper_q = 100 - lower_q
    bootstrap_alpha = 1.0 - (cfg.ci_level / 100.0)

    trueBL1_domain = truth["B1"] @ truth["Lambda1"]  # (len(x1), K)
    trueBL2_domain = truth["B2"] @ truth["Lambda2"]  # (len(x2), K)

    # ------------ PLS ------------
    pls_records: list[dict] = []
    try:
        pls_ref = PartialLeastSquares(n_components=cfg.K, scale=True)
        pls_ref.fit(f1_gp, f2_gp)

        ref_x = pls_ref.model.x_loadings_  # (len(x1), K)
        ref_y = pls_ref.model.y_loadings_  # (len(x2), K)

        true1_aligned = _align_truth_to_reference(trueBL1_domain, ref_x)
        true2_aligned = _align_truth_to_reference(trueBL2_domain, ref_y)

        boot_pls = bootstrap_prediction_uncertainty(
            model_class=PartialLeastSquares,
            model_params={"n_components": cfg.K, "scale": True},
            X_train=f1_gp,
            Y_train=f2_gp,
            X_test=f1_gp,
            n_boot=cfg.n_boot_pls,
            alpha=bootstrap_alpha,
            random_state=seed,
            verbose=False,
        )

        x_stack = _align_stack_to_reference(boot_pls.all_x_weights_stack, ref_x)
        y_stack = _align_stack_to_reference(boot_pls.all_y_weights_stack, ref_y)

        pls_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x1, true_aligned=true1_aligned, stack_aligned=x_stack,
            mode_id=1, method_name="PLS", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))
        pls_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x2, true_aligned=true2_aligned, stack_aligned=y_stack,
            mode_id=2, method_name="PLS", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))

        fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
        for k in range(cfg.K):
            draws = x_stack[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 0], coords=cfg.x1, lower=lo, upper=hi, center=mu, truth=true1_aligned[:, k],
                title=f"PLS | Mode 1 (X) | Factor {k+1}", ylabel="loading/weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

            draws = y_stack[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 1], coords=cfg.x2, lower=lo, upper=hi, center=mu, truth=true2_aligned[:, k],
                title=f"PLS | Mode 2 (Y) | Factor {k+1}", ylabel="loading/weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

        fig.tight_layout()
        fig.savefig(seed_dir / "pls_bootstrap_weights_vs_true.png", dpi=200)
        plt.close(fig)

        pd.DataFrame(pls_records).to_csv(seed_dir / "coverage_pls.csv", index=False)

    except Exception as e:
        pd.DataFrame([{
            "seed": seed, "method": "PLS", "mode": np.nan, "factor": np.nan,
            "coverage_pct": np.nan, "quad_error": np.nan, "error": repr(e)
        }]).to_csv(seed_dir / "coverage_pls.csv", index=False)

    # ------------ FPLS (GP) ------------
    fpls_gp_records: list[dict] = []
    try:
        fpls_ref = FunctionalPartialLeastSquares(n_components=cfg.K, scale=True)
        fpls_ref.fit(f1_gp, f2_gp)
        ref_x_gp, ref_y_gp = _extract_fpls_domain_weights_singleX(fpls_ref, f1_gp)

        true1_aligned_gp = _align_truth_to_reference(trueBL1_domain, ref_x_gp)
        true2_aligned_gp = _align_truth_to_reference(trueBL2_domain, ref_y_gp)

        boot_fpls_gp = bootstrap_prediction_uncertainty(
            model_class=FunctionalPartialLeastSquares,
            model_params={"n_components": cfg.K, "scale": True},
            X_train=f1_gp,
            Y_train=f2_gp,
            X_test=f1_gp,
            n_boot=cfg.n_boot_pls,
            alpha=bootstrap_alpha,
            random_state=seed,
            verbose=False,
        )

        x_stack_gp = _align_stack_to_reference(boot_fpls_gp.all_x_weights_stack, ref_x_gp)
        y_stack_gp = _align_stack_to_reference(boot_fpls_gp.all_y_weights_stack, ref_y_gp)

        fpls_gp_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x1, true_aligned=true1_aligned_gp, stack_aligned=x_stack_gp,
            mode_id=1, method_name="FPLS_GP", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))
        fpls_gp_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x2, true_aligned=true2_aligned_gp, stack_aligned=y_stack_gp,
            mode_id=2, method_name="FPLS_GP", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))

        fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
        for k in range(cfg.K):
            draws = x_stack_gp[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 0], coords=cfg.x1, lower=lo, upper=hi, center=mu, truth=true1_aligned_gp[:, k],
                title=f"FPLS (GP basis) | Mode 1 | Factor {k+1}", ylabel="weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

            draws = y_stack_gp[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 1], coords=cfg.x2, lower=lo, upper=hi, center=mu, truth=true2_aligned_gp[:, k],
                title=f"FPLS (GP basis) | Mode 2 | Factor {k+1}", ylabel="weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

        fig.tight_layout()
        fig.savefig(seed_dir / "fpls_gp_bootstrap_weights_vs_true.png", dpi=200)
        plt.close(fig)

        pd.DataFrame(fpls_gp_records).to_csv(seed_dir / "coverage_fpls_gp.csv", index=False)

    except Exception as e:
        pd.DataFrame([{
            "seed": seed, "method": "FPLS_GP", "mode": np.nan, "factor": np.nan,
            "coverage_pct": np.nan, "quad_error": np.nan, "error": repr(e)
        }]).to_csv(seed_dir / "coverage_fpls_gp.csv", index=False)

    # ------------ FPLS (B-spline) ------------
    fpls_bs_records: list[dict] = []
    try:
        fpls_ref_bs = FunctionalPartialLeastSquares(n_components=cfg.K, scale=True)
        fpls_ref_bs.fit(f1_bs, f2_bs)
        ref_x_bs, ref_y_bs = _extract_fpls_domain_weights_singleX(fpls_ref_bs, f1_bs)

        true1_aligned_bs = _align_truth_to_reference(trueBL1_domain, ref_x_bs)
        true2_aligned_bs = _align_truth_to_reference(trueBL2_domain, ref_y_bs)

        boot_fpls_bs = bootstrap_prediction_uncertainty(
            model_class=FunctionalPartialLeastSquares,
            model_params={"n_components": cfg.K, "scale": True},
            X_train=f1_bs,
            Y_train=f2_bs,
            X_test=f1_bs,
            n_boot=cfg.n_boot_pls,
            alpha=bootstrap_alpha,
            random_state=seed,
            verbose=False,
        )

        x_stack_bs = _align_stack_to_reference(boot_fpls_bs.all_x_weights_stack, ref_x_bs)
        y_stack_bs = _align_stack_to_reference(boot_fpls_bs.all_y_weights_stack, ref_y_bs)

        fpls_bs_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x1, true_aligned=true1_aligned_bs, stack_aligned=x_stack_bs,
            mode_id=1, method_name="FPLS_BSPLINE", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))
        fpls_bs_records.extend(_compute_curve_metrics_from_stack(
            seed=seed, coords=cfg.x2, true_aligned=true2_aligned_bs, stack_aligned=y_stack_bs,
            mode_id=2, method_name="FPLS_BSPLINE", lower_q=lower_q, upper_q=upper_q, ci_level=cfg.ci_level
        ))

        fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
        for k in range(cfg.K):
            draws = x_stack_bs[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 0], coords=cfg.x1, lower=lo, upper=hi, center=mu, truth=true1_aligned_bs[:, k],
                title=f"FPLS (B-spline basis) | Mode 1 | Factor {k+1}", ylabel="weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

            draws = y_stack_bs[:, :, k]
            lo = np.nanpercentile(draws, lower_q, axis=0)
            hi = np.nanpercentile(draws, upper_q, axis=0)
            mu = np.nanmean(draws, axis=0)
            _plot_band_vs_truth(
                ax=axs[k, 1], coords=cfg.x2, lower=lo, upper=hi, center=mu, truth=true2_aligned_bs[:, k],
                title=f"FPLS (B-spline basis) | Mode 2 | Factor {k+1}", ylabel="weight",
                show_legend=(k == 0), ci_level=cfg.ci_level, center_label="Bootstrap mean"
            )

        fig.tight_layout()
        fig.savefig(seed_dir / "fpls_bspline_bootstrap_weights_vs_true.png", dpi=200)
        plt.close(fig)

        pd.DataFrame(fpls_bs_records).to_csv(seed_dir / "coverage_fpls_bspline.csv", index=False)

    except Exception as e:
        pd.DataFrame([{
            "seed": seed, "method": "FPLS_BSPLINE", "mode": np.nan, "factor": np.nan,
            "coverage_pct": np.nan, "quad_error": np.nan, "error": repr(e)
        }]).to_csv(seed_dir / "coverage_fpls_bspline.csv", index=False)

    return pls_records, fpls_gp_records, fpls_bs_records


# ---------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------

def run_seed(seed: int, cfg: ExperimentConfig, scenario_dir: Path, B1_true: np.ndarray, B2_true: np.ndarray):
    seed_dir = scenario_dir / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    Y1, Y2, truth = simulate_once(seed, cfg, B1_true, B2_true)

    # Build FunctionalData (GP)
    f1_gp = build_functional_data_gp(cfg.x1, Y1, cfg)
    f2_gp = build_functional_data_gp(cfg.x2, Y2, cfg)

    # Build FunctionalData (B-spline)
    f1_bs = build_functional_data_bspline(cfg.x1, Y1, degree=3)
    f2_bs = build_functional_data_bspline(cfg.x2, Y2, degree=3)

    # Fit BLF models (both are kept, as requested)
    model_gp = _fit_blf_model(seed, cfg, modes=[f1_gp, f2_gp])
    model_bs = _fit_blf_model(seed, cfg, modes=[f1_bs, f2_bs])

    # True domain BLambda curves from truth generation basis
    trueBL1_domain = truth["B1"] @ truth["Lambda1"]  # (len(x1), K)
    trueBL2_domain = truth["B2"] @ truth["Lambda2"]  # (len(x2), K)

    # Align true factors to VarimaxRSP for GP-fit model:
    # map true domain curves into GP coefficient space
    Lambda1_true_in_gp = npl.pinv(f1_gp._basis_matrix) @ trueBL1_domain
    Lambda2_true_in_gp = npl.pinv(f2_gp._basis_matrix) @ trueBL2_domain
    out_gp = align_true_factors_to_varimaxrsp(
        model=model_gp,
        Lambda_true_list=[Lambda1_true_in_gp, Lambda2_true_in_gp],
        B_true_list=[f1_gp._basis_matrix, f2_gp._basis_matrix],
    )
    BL1_true_gp = out_gp["BLambda_true_aligned_per_mode"][0]
    BL2_true_gp = out_gp["BLambda_true_aligned_per_mode"][1]

    # Align true factors to VarimaxRSP for B-spline-fit model:
    # map true domain curves into B-spline coefficient space
    Lambda1_true_in_bs = npl.pinv(f1_bs._basis_matrix) @ trueBL1_domain
    Lambda2_true_in_bs = npl.pinv(f2_bs._basis_matrix) @ trueBL2_domain
    out_bs = align_true_factors_to_varimaxrsp(
        model=model_bs,
        Lambda_true_list=[Lambda1_true_in_bs, Lambda2_true_in_bs],
        B_true_list=[f1_bs._basis_matrix, f2_bs._basis_matrix],
    )
    BL1_true_bs = out_bs["BLambda_true_aligned_per_mode"][0]
    BL2_true_bs = out_bs["BLambda_true_aligned_per_mode"][1]

    # Posterior BLambda
    postBL1_gp = model_gp.get_BLambda_d_VarimaxRSP(1)
    postBL2_gp = model_gp.get_BLambda_d_VarimaxRSP(2)
    postBL1_bs = model_bs.get_BLambda_d_VarimaxRSP(1)
    postBL2_bs = model_bs.get_BLambda_d_VarimaxRSP(2)

    # BLF coverage + quad error + plot
    rec_blf_gp = _blf_posterior_records_and_plot(
        seed, cfg, seed_dir,
        posteriorBLambda1=postBL1_gp, posteriorBLambda2=postBL2_gp,
        BLambda1_true_aligned=BL1_true_gp, BLambda2_true_aligned=BL2_true_gp,
        fig_name="posterior_vs_true.png", csv_name="coverage.csv"
    )
    rec_blf_bs = _blf_posterior_records_and_plot(
        seed, cfg, seed_dir,
        posteriorBLambda1=postBL1_bs, posteriorBLambda2=postBL2_bs,
        BLambda1_true_aligned=BL1_true_bs, BLambda2_true_aligned=BL2_true_bs,
        fig_name="posterior_vs_true_bspline.png", csv_name="coverage_bspline.csv"
    )

    # PLS / FPLS (unchanged)
    rec_pls, rec_fpls_gp, rec_fpls_bs = _pls_fpls_records_and_plots(
        seed=seed, cfg=cfg, seed_dir=seed_dir,
        f1_gp=f1_gp, f2_gp=f2_gp, f1_bs=f1_bs, f2_bs=f2_bs,
        truth=truth,
    )

    # Cleanup
    del model_gp, model_bs
    gc.collect()

    return rec_blf_gp, rec_blf_bs, rec_pls, rec_fpls_gp, rec_fpls_bs


# ---------------------------------------------------------------------
# Experiment runner (call this in your loop over N,K)
# ---------------------------------------------------------------------

def run_experiment(cfg_in: ExperimentConfig) -> Path:
    cfg = _cfg_with_defaults(cfg_in)

    scenario_dir = cfg.output_root / cfg.scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Precompute truth B-spline bases for this scenario (generation uses fewer basis fns than points)
    B1_true = make_bspline_basis_matrix(cfg.x1, n_basis=cfg.bspline_true_n_basis_mode1, degree=cfg.bspline_true_degree)
    B2_true = make_bspline_basis_matrix(cfg.x2, n_basis=cfg.bspline_true_n_basis_mode2, degree=cfg.bspline_true_degree)

    all_blf_gp, all_blf_bs = [], []
    all_pls, all_fpls_gp, all_fpls_bs = [], [], []
    failures = []

    for seed in tqdm(list(cfg.seeds), desc=f"{cfg.scenario_name} | seeds"):
        try:
            rec_blf_gp, rec_blf_bs, rec_pls, rec_fpls_gp, rec_fpls_bs = run_seed(
                seed=seed, cfg=cfg, scenario_dir=scenario_dir, B1_true=B1_true, B2_true=B2_true
            )
            all_blf_gp.extend(rec_blf_gp)
            all_blf_bs.extend(rec_blf_bs)
            all_pls.extend(rec_pls)
            all_fpls_gp.extend(rec_fpls_gp)
            all_fpls_bs.extend(rec_fpls_bs)

        except Exception as e:
            failures.append({"seed": seed, "error": repr(e)})

            seed_dir = scenario_dir / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "seed": seed, "mode": np.nan, "factor": np.nan,
                "coverage_pct": np.nan, "quad_error": np.nan, "error": repr(e)
            }]).to_csv(seed_dir / "coverage.csv", index=False)

    # Raw all-seeds tables
    df_blf_gp = pd.DataFrame(all_blf_gp)
    df_blf_bs = pd.DataFrame(all_blf_bs)
    df_pls = pd.DataFrame(all_pls)
    df_fpls_gp = pd.DataFrame(all_fpls_gp)
    df_fpls_bs = pd.DataFrame(all_fpls_bs)

    df_blf_gp.to_csv(scenario_dir / "coverage_all_seeds.csv", index=False)
    df_blf_bs.to_csv(scenario_dir / "coverage_all_seeds_bspline.csv", index=False)
    df_pls.to_csv(scenario_dir / "coverage_all_seeds_pls.csv", index=False)
    df_fpls_gp.to_csv(scenario_dir / "coverage_all_seeds_fpls_gp.csv", index=False)
    df_fpls_bs.to_csv(scenario_dir / "coverage_all_seeds_fpls_bspline.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(scenario_dir / "failures.csv", index=False)

    # Summaries
    _write_method_summaries(df_blf_gp, scenario_dir / "blf_gp")
    _write_method_summaries(df_blf_bs, scenario_dir / "blf_bspline")
    _write_method_summaries(df_pls, scenario_dir / "pls")
    _write_method_summaries(df_fpls_gp, scenario_dir / "fpls_gp")
    _write_method_summaries(df_fpls_bs, scenario_dir / "fpls_bspline")

    # Manifest
    manifest = {
        "scenario_name": cfg.scenario_name,
        "N": int(cfg.N),
        "K": int(cfg.K),
        "n_seeds": int(len(cfg.seeds)),
        "ci_level": float(cfg.ci_level),
        "n_boot_pls": int(cfg.n_boot_pls),
        "truth_generation": {
            "basis": "bspline",
            "degree": int(cfg.bspline_true_degree),
            "n_basis_mode1": int(cfg.bspline_true_n_basis_mode1),
            "n_basis_mode2": int(cfg.bspline_true_n_basis_mode2),
            "n_points_mode1": int(len(cfg.x1)),
            "n_points_mode2": int(len(cfg.x2)),
        },
        "fitted_models": {
            "BLF_GP_basis": {"gp_type": cfg.gp_type, "gp_lengthscale": float(cfg.gp_lengthscale), "gp_variance": float(cfg.gp_variance)},
            "BLF_BSPLINE_basis": {"degree": 3, "n_basis": "len(grid)"},
        },
        "stan": {
            "n_chains": int(cfg.n_chains),
            "iter_warmup": int(cfg.iter_warmup),
            "iter_sampling": int(cfg.iter_sampling),
            "thin": int(cfg.thin),
            "max_treedepth": int(cfg.max_treedepth),
        },
    }
    (scenario_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return scenario_dir


# ---------------------------------------------------------------------
# Example: run multiple scenarios in a loop
# ---------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        (10, 2),
        (25, 2),
        (50, 2),
        (75, 2),
        (100, 2),
        (10, 3),
        (25, 3),
        (50, 3),
        (75, 3),
        (100, 3),
        (10, 4),
        (25, 4),
        (50, 4),
        (75, 4),
        (100, 4),
    ]

    n_experiments = 50

    i = 1
    for N, K in scenarios:
        cfg = ExperimentConfig(
            output_root=Path("Figures"),
            scenario_name=f"N{N}_K{K}",
            N=N,
            K=K,
            seeds=tuple(range(i, i + n_experiments)),
            n_boot_pls=200,
            ci_level=80.0,
            # Optional overrides for truth generation basis sizes:
            bspline_true_n_basis_mode1=8,
            bspline_true_n_basis_mode2=12,
            bspline_true_degree=3,
        )
        i += n_experiments
        out = run_experiment(cfg)
        print(f"Wrote results to: {out}")
