# -*- coding: utf-8 -*-
"""
Bayesian-only pipeline (MvtBLF) with pluggable:

1) Data-generating mechanism (truth) chosen by a string.
2) Second MvtBLF basis chosen by a string.
   - First MvtBLF is ALWAYS the GP-basis solution.

Removed:
- PLS / Functional PLS + bootstrap (per request).

Extension points (add your own):
- TRUTH_MECHANISMS: add new sampling mechanisms (truth basis builders).
- FIT_BASIS_SETTERS: add new FunctionalData basis setters for the 2nd MvtBLF.

Fourier basis is implemented (truth + fit).

Usage (CLI):
  python run_blf.py --truth bspline --alt-basis fourier --scenario N30_K3
  python run_blf.py --truth fourier --alt-basis bspline --scenario N30_K3

Programmatic:
  run_experiment(cfg, truth_mechanism="bspline", alt_basis="fourier")
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import numpy.linalg as npl
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import MultivariateBLF.MultivariateBLF as mvblf
from MultivariateBLF.visualization import align_true_factors_to_varimaxrsp




def _blambda_posterior_mean_stacked(model: mvblf.MvtBLF) -> np.ndarray:
    """
    Stack posterior mean BLambda (VarimaxRSP) across modes:
      returns (M_total, K)
    """
    D = model.D
    out = []
    for d in range(1, D + 1):
        da = model.get_BLambda_d_VarimaxRSP(d)  # expected dims: chain, draw, M, K_rsp (or similar)
        reduce_dims = [dim for dim in ("chain", "draw") if dim in da.dims]
        da_mean = da.mean(dim=reduce_dims)

        # try to standardize dims ordering -> (M, K)
        m_dim = "M" if "M" in da_mean.dims else da_mean.dims[0]
        k_dim = "K_rsp" if "K_rsp" in da_mean.dims else ("K" if "K" in da_mean.dims else da_mean.dims[-1])
        da_mean = da_mean.transpose(m_dim, k_dim)

        out.append(np.asarray(da_mean.values, dtype=float))  # (M_d, K)
    return np.vstack(out)  # (M_total, K)


def align_true_factors_to_varimaxrsp_B_version(
    model: mvblf.MvtBLF,
    BLambda_true_list: List[np.ndarray],
    method: str = "varimax",
) -> Dict[str, Any]:
    """
    Align *true* BLambda curves (domain-level) to the Varimax-RSP orientation used by `model`.

    This version NEVER touches Lambda or projections. It only works with BLambda = B @ Lambda.

    Steps
    -----
    1. Stack true BLambda_d over modes (M_total x K).
    2. Apply orthogonal rotation (default Varimax) to stacked true BLambda.
    3. Build posterior reference in the *same space*: posterior mean BLambda (VarimaxRSP), stacked (M_total x K).
    4. Find best signed permutation from true->reference via Hungarian assignment.
    5. Return aligned true BLambda per mode.

    Returns
    -------
    dict with:
      - "BLambda_true_varimax_stacked": (M_total, K)
      - "BLambda_true_aligned_stacked": (M_total, K)
      - "BLambda_true_aligned_per_mode": list of (M_d, K)
      - "perm": (K,) int
      - "signs": (K,) float
      - "R_true": (K, K) rotation used on true BLambda
    """
    from MultivariateBLF.helpers import _ortho_rotation
    from scipy.optimize import linear_sum_assignment

    D = model.D
    K = model.n_components

    if len(BLambda_true_list) != D:
        raise ValueError(f"expected {D} true BLambda arrays, got {len(BLambda_true_list)}")

    BLambda_true_list = [np.asarray(Bd, dtype=float) for Bd in BLambda_true_list]
    M_list = []
    for d, Bd in enumerate(BLambda_true_list):
        if Bd.ndim != 2 or Bd.shape[1] != K:
            raise ValueError(f"BLambda_true_list[{d}] must have shape (M_d, K={K}), got {Bd.shape}")
        M_list.append(Bd.shape[0])
    M_total = int(sum(M_list))

    # 1) Stack true BLambda over modes
    BLambda_true_stacked = np.vstack(BLambda_true_list)  # (M_total, K)

    # 2) Varimax (or other orthogonal) rotation of true stacked BLambda
    if method is not None:
        BLambda_true_varimax_stacked, R_true = _ortho_rotation(
            BLambda_true_stacked, method=method
        )
    else:
        BLambda_true_varimax_stacked = BLambda_true_stacked.copy()
        R_true = np.eye(K)

    # 3) Posterior reference (mean BLambda in VarimaxRSP orientation)
    BLambda_post_ref_stacked = _blambda_posterior_mean_stacked(model)  # (M_total, K)
    if BLambda_post_ref_stacked.shape != (M_total, K):
        raise RuntimeError(
            f"Shape mismatch: true BLambda stacked {(M_total, K)} vs posterior ref {BLambda_post_ref_stacked.shape}"
        )

    # 4) Signed permutation alignment: true_varimax -> posterior reference
    A = BLambda_post_ref_stacked                 # reference (M_total x K)
    B = BLambda_true_varimax_stacked             # candidate (M_total x K)

    A_norm_sq = np.sum(A ** 2, axis=0)
    B_norm_sq = np.sum(B ** 2, axis=0)

    cost = np.empty((K, K), dtype=float)
    sign_matrix = np.empty((K, K), dtype=float)

    for i in range(K):
        a = A[:, i]
        for j in range(K):
            b = B[:, j]
            dot = float(np.dot(a, b))
            s = 1.0 if dot >= 0.0 else -1.0
            sign_matrix[i, j] = s
            cost[i, j] = A_norm_sq[i] + B_norm_sq[j] - 2.0 * abs(dot)

    row_ind, col_ind = linear_sum_assignment(cost)

    BLambda_true_aligned_stacked = np.zeros_like(B)
    signs = np.zeros(K, dtype=float)
    perm = np.zeros(K, dtype=int)

    for idx in range(len(row_ind)):
        i = int(row_ind[idx])
        j = int(col_ind[idx])
        s = float(sign_matrix[i, j])
        BLambda_true_aligned_stacked[:, i] = s * B[:, j]
        signs[i] = s
        perm[i] = j

    # 5) Split aligned BLambda back into per-mode pieces
    BLambda_true_aligned_per_mode: List[np.ndarray] = []
    start = 0
    for d in range(D):
        end = start + M_list[d]
        BLambda_true_aligned_per_mode.append(BLambda_true_aligned_stacked[start:end, :])
        start = end

    return {
        "BLambda_true_varimax_stacked": BLambda_true_varimax_stacked,
        "BLambda_true_aligned_stacked": BLambda_true_aligned_stacked,
        "BLambda_true_aligned_per_mode": BLambda_true_aligned_per_mode,
        "perm": perm,
        "signs": signs,
        "R_true": R_true,
    }



# =============================================================================
# Config
# =============================================================================

def _default_x1() -> np.ndarray:
    return np.linspace(0.0, 1.0, 15)

def _default_x2() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 25)

def _default_seeds() -> tuple[int, ...]:
    return tuple(range(1, 6))  # adjust as needed


@dataclass(frozen=True)
class ExperimentConfig:
    # Output
    output_root: Path = Path("Figures")
    scenario_name: str = "default"

    # Grids
    x1: np.ndarray = field(default_factory=_default_x1)
    x2: np.ndarray = field(default_factory=_default_x2)

    # Dimensions
    N: int = 30
    K: int = 3
    seeds: Sequence[int] = field(default_factory=_default_seeds)

    # Data-generating hyperparameters (same structure as your previous script)
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

    # Truth basis params (mechanism-specific may use these)
    # B-spline truth
    bspline_true_degree: int = 3
    bspline_true_n_basis_mode1: Optional[int] = None  # if None -> default < n_points
    bspline_true_n_basis_mode2: Optional[int] = None

    # Fourier truth
    # n_basis = 1 + 2*h; if None -> default < n_points (odd)
    fourier_true_n_basis_mode1: Optional[int] = None
    fourier_true_n_basis_mode2: Optional[int] = None

    # GP fit basis (fixed for model #1)
    gp_type: str = "matern5/2"
    gp_lengthscale: float = 0.2
    gp_variance: float = 1.0

    # Alt (model #2) basis sizes (used for bspline/fourier fit setters)
    # If None -> use len(grid) (and made odd for Fourier if needed).
    alt_n_basis_mode1: Optional[int] = None
    alt_n_basis_mode2: Optional[int] = None
    alt_bspline_degree: int = 3  # for alt-basis="bspline"

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

    # Interval / coverage
    ci_level: float = 80.0


def _choose_default_true_n_basis(n_points: int, min_required: int) -> int:
    """
    Pick a default number of truth basis functions strictly less than n_points,
    and >= min_required.
    """
    cand = max(min_required, int(np.floor(n_points / 2)))
    cand = min(cand, n_points - 1)
    if cand >= n_points:
        cand = n_points - 1
    return cand


def _ensure_truth_defaults(cfg: ExperimentConfig) -> ExperimentConfig:
    d = dict(cfg.__dict__)

    # B-spline truth defaults
    deg = int(d["bspline_true_degree"])
    if d.get("bspline_true_n_basis_mode1") is None:
        d["bspline_true_n_basis_mode1"] = _choose_default_true_n_basis(len(cfg.x1), deg + 2)
    if d.get("bspline_true_n_basis_mode2") is None:
        d["bspline_true_n_basis_mode2"] = _choose_default_true_n_basis(len(cfg.x2), deg + 2)

    # Fourier truth defaults: n_basis must be odd and < n_points
    if d.get("fourier_true_n_basis_mode1") is None:
        cand = _choose_default_true_n_basis(len(cfg.x1), 3)  # at least 3 => 1 + 2*1
        d["fourier_true_n_basis_mode1"] = cand if cand % 2 == 1 else cand - 1
    if d.get("fourier_true_n_basis_mode2") is None:
        cand = _choose_default_true_n_basis(len(cfg.x2), 3)
        d["fourier_true_n_basis_mode2"] = cand if cand % 2 == 1 else cand - 1

    # Safety
    if d.get("absence_of_psi2") is None:
        d["absence_of_psi2"] = [False, False]

    return ExperimentConfig(**d)


# =============================================================================
# Basis utilities (B-spline + Fourier)
# =============================================================================

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

    return N.T  # (M, n)


def bspline_basis_functions(domain: Tuple[float, float], n_basis: int, degree: int) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Basis functions for mvblf.FunctionalData.set_functional_basis.
    """
    a, b = map(float, domain)
    if not a < b:
        raise ValueError("domain must satisfy a < b")

    k = int(degree)
    n = int(n_basis)
    if k < 0:
        raise ValueError("degree must be >= 0")
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

    return [lambda x, i=i: eval_all(x)[i] for i in range(n)]


def _normalize_to_unit_interval(x: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    a, b = map(float, domain)
    x = np.asarray(x, dtype=float)
    if not a < b:
        raise ValueError("domain must satisfy a < b")
    return (x - a) / (b - a)


def _fourier_n_basis_to_harmonics(n_basis: int) -> int:
    """
    Fourier basis: [1, sin(2π t), cos(2π t), ..., sin(2π h t), cos(2π h t)]
    => n_basis must be odd: n_basis = 1 + 2*h
    """
    n_basis = int(n_basis)
    if n_basis < 1:
        raise ValueError("fourier n_basis must be >= 1")
    if n_basis % 2 == 0:
        raise ValueError("fourier n_basis must be odd (1 + 2*h).")
    return (n_basis - 1) // 2


def make_fourier_basis_matrix(coords: np.ndarray, n_basis: int) -> np.ndarray:
    """
    Fourier basis matrix evaluated at coords.
    Shape: (M, n_basis). n_basis must be odd: 1 + 2*h.
    """
    x = np.asarray(coords, dtype=float).reshape(-1)
    a, b = float(x.min()), float(x.max())
    t = _normalize_to_unit_interval(x, (a, b))

    h = _fourier_n_basis_to_harmonics(n_basis)

    cols = [np.ones_like(t)]
    for m in range(1, h + 1):
        cols.append(np.sin(2.0 * np.pi * m * t))
        cols.append(np.cos(2.0 * np.pi * m * t))
    return np.column_stack(cols)


def fourier_basis_functions(domain: Tuple[float, float], n_basis: int) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Fourier basis functions for mvblf.FunctionalData.set_functional_basis.
    """
    a, b = map(float, domain)
    h = _fourier_n_basis_to_harmonics(n_basis)

    def basis_vec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        t = _normalize_to_unit_interval(x_arr, (a, b))
        out = [np.ones_like(t)]
        for m in range(1, h + 1):
            out.append(np.sin(2.0 * np.pi * m * t))
            out.append(np.cos(2.0 * np.pi * m * t))
        return np.stack(out, axis=0)  # (n_basis, ...)

    return [lambda x, i=i: basis_vec(x)[i] for i in range(1 + 2 * h)]


# =============================================================================
# Fit basis setters (FunctionalData)
# =============================================================================

def build_functional_data_gp(coords: np.ndarray, data: np.ndarray, cfg: ExperimentConfig) -> mvblf.FunctionalData:
    fd = mvblf.FunctionalData(coords=coords, data=data)
    fd.set_GP_basis(
        n_basis=len(coords),
        gaussian_process=cfg.gp_type,
        gp_lengthscale=cfg.gp_lengthscale,
        gp_variance=cfg.gp_variance,
    )
    return fd


def _alt_n_basis(cfg: ExperimentConfig, mode: int, coords: np.ndarray, basis_name: str) -> int:
    """
    Decide number of basis functions for alt model (model #2).
    For bspline default = len(coords). For fourier default = len(coords) (must be odd).
    """
    n = cfg.alt_n_basis_mode1 if mode == 1 else cfg.alt_n_basis_mode2
    if n is None:
        n = int(len(coords))
        if basis_name == "fourier" and n % 2 == 0:
            n -= 1
    return int(n)


def build_functional_data_bspline_alt(coords: np.ndarray, data: np.ndarray, cfg: ExperimentConfig, mode: int) -> mvblf.FunctionalData:
    n_basis = _alt_n_basis(cfg, mode, coords, "bspline")
    fd = mvblf.FunctionalData(coords=coords, data=data)
    fd.set_functional_basis(
        basis_functions=bspline_basis_functions(
            domain=(float(coords.min()), float(coords.max())),
            n_basis=n_basis,
            degree=int(cfg.alt_bspline_degree),
        )
    )
    return fd


def build_functional_data_fourier_alt(coords: np.ndarray, data: np.ndarray, cfg: ExperimentConfig, mode: int) -> mvblf.FunctionalData:
    n_basis = _alt_n_basis(cfg, mode, coords, "fourier")
    if n_basis % 2 == 0:
        raise ValueError(f"Fourier n_basis must be odd, got {n_basis} for mode={mode}.")
    fd = mvblf.FunctionalData(coords=coords, data=data)
    fd.set_functional_basis(
        basis_functions=fourier_basis_functions(
            domain=(float(coords.min()), float(coords.max())),
            n_basis=n_basis,
        )
    )
    return fd


# Register fit-basis setters for model #2 here.
# To add a new fitted basis:
# 1) implement build_functional_data_<name>_alt(...)
# 2) add it to FIT_BASIS_SETTERS with key "<name>"
FIT_BASIS_SETTERS = {
    "bspline": build_functional_data_bspline_alt,
    "fourier": build_functional_data_fourier_alt,
    # "gp": (optional) could reuse GP for debugging if you want:
    # "gp": lambda coords, data, cfg, mode: build_functional_data_gp(coords, data, cfg),
}


# =============================================================================
# Truth (data-generating) mechanisms
# =============================================================================

def truth_basis_bspline(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Truth = low-rank B-spline basis (n_basis_true < n_points).
    """
    n1 = int(cfg.bspline_true_n_basis_mode1)
    n2 = int(cfg.bspline_true_n_basis_mode2)
    deg = int(cfg.bspline_true_degree)

    if not (n1 < len(cfg.x1) and n2 < len(cfg.x2)):
        raise ValueError("B-spline truth n_basis must be < number of grid points per mode.")
    if n1 <= deg or n2 <= deg:
        raise ValueError("B-spline truth n_basis must be > degree.")

    B1 = make_bspline_basis_matrix(cfg.x1, n_basis=n1, degree=deg)
    B2 = make_bspline_basis_matrix(cfg.x2, n_basis=n2, degree=deg)
    meta = {"basis": "bspline", "degree": deg, "n_basis_mode1": n1, "n_basis_mode2": n2}
    return B1, B2, meta


def truth_basis_fourier(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Truth = low-rank Fourier basis (n_basis_true < n_points, odd).
    """
    n1 = int(cfg.fourier_true_n_basis_mode1)
    n2 = int(cfg.fourier_true_n_basis_mode2)

    if n1 % 2 == 0 or n2 % 2 == 0:
        raise ValueError("Fourier truth n_basis must be odd (1 + 2*h).")
    if not (n1 < len(cfg.x1) and n2 < len(cfg.x2)):
        raise ValueError("Fourier truth n_basis must be < number of grid points per mode.")
    if n1 < 3 or n2 < 3:
        raise ValueError("Fourier truth n_basis should be >= 3 (constant + first sin/cos).")

    B1 = make_fourier_basis_matrix(cfg.x1, n_basis=n1)
    B2 = make_fourier_basis_matrix(cfg.x2, n_basis=n2)
    meta = {"basis": "fourier", "n_basis_mode1": n1, "n_basis_mode2": n2}
    return B1, B2, meta


def truth_basis_gp_matern(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Truth = the SAME GP basis matrix used by the BLF GP-basis model.

    This uses mvblf.FunctionalData.set_GP_basis with:
      - gaussian_process = cfg.gp_type (should be a Matern variant)
      - gp_lengthscale   = cfg.gp_lengthscale
      - gp_variance      = cfg.gp_variance

    Returns:
      B1_true: (M1, M1)
      B2_true: (M2, M2)
    """
    # Optional: enforce that this truth mechanism is actually Matern
    if "matern" not in str(cfg.gp_type).lower():
        raise ValueError(
            f"truth 'gp_matern' requires cfg.gp_type to be a Matern GP (got {cfg.gp_type!r})."
        )

    # Dummy data just to initialize FunctionalData; basis depends on coords + GP params.
    Z1 = np.zeros((len(cfg.x1), 1), dtype=float)
    Z2 = np.zeros((len(cfg.x2), 1), dtype=float)

    fd1 = build_functional_data_gp(cfg.x1, Z1, cfg)
    fd2 = build_functional_data_gp(cfg.x2, Z2, cfg)

    B1 = np.asarray(fd1._basis_matrix, dtype=float)
    B2 = np.asarray(fd2._basis_matrix, dtype=float)

    meta = {
        "basis": "gp_matern",
        "gp_type": cfg.gp_type,
        "gp_lengthscale": float(cfg.gp_lengthscale),
        "gp_variance": float(cfg.gp_variance),
        "n_basis_mode1": int(B1.shape[1]),
        "n_basis_mode2": int(B2.shape[1]),
    }
    return B1, B2, meta





# Register truth mechanisms here.
# To add a new sampling mechanism:
# 1) implement truth_basis_<name>(cfg) -> (B1, B2, meta)
# 2) add it to TRUTH_MECHANISMS with key "<name>"
TRUTH_MECHANISMS = {
    "bspline": truth_basis_bspline,
    "fourier": truth_basis_fourier,
    "gp_matern": truth_basis_gp_matern,
}


# =============================================================================
# Simulation core (shared across truth mechanisms)
# =============================================================================

def simulate_once(seed: int, cfg: ExperimentConfig, B1_true: np.ndarray, B2_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Given truth bases B1_true, B2_true (possibly low-rank), simulate Y1,Y2
    with your existing Bayesian factor-generation scheme.
    """
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

    # Domain data from truth basis
    Y1 = B1_true @ (Lambda1 @ eta.T + u1)
    Y2 = B2_true @ (Lambda2 @ eta.T + u2)

    psi12 = 1.0 / rng.gamma(shape=cfg.psi12_alpha, scale=1.0 / cfg.psi12_beta)
    psi22 = 1.0 / rng.gamma(shape=cfg.psi22_alpha, scale=1.0 / cfg.psi22_beta)

    Y1 += rng.normal(0.0, np.sqrt(psi12), size=Y1.shape)
    Y2 += rng.normal(0.0, np.sqrt(psi22), size=Y2.shape)

    truth = {"Lambda1": Lambda1, "Lambda2": Lambda2, "B1": B1_true, "B2": B2_true}
    return Y1, Y2, truth


# =============================================================================
# BLF Fit + evaluation
# =============================================================================

def _fit_blf_model(seed: int, cfg: ExperimentConfig, modes: List[mvblf.FunctionalData]) -> mvblf.MvtBLF:
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


def _evaluate_and_plot_blf(
    *,
    seed: int,
    cfg: ExperimentConfig,
    seed_dir: Path,
    model: mvblf.MvtBLF,
    f1: mvblf.FunctionalData,
    f2: mvblf.FunctionalData,
    truth: Dict,
    tag: str,
) -> List[Dict]:
    """
    Computes coverage + quadratic error (ISE via trapz) for BLambda posterior
    and saves a plot. 'tag' is used in filenames.
    """
    lower_q = (100 - cfg.ci_level) / 2
    upper_q = 100 - lower_q

    # True domain BLambda curves
    trueBL1_domain = truth["B1"] @ truth["Lambda1"]  # (len(x1), K)
    trueBL2_domain = truth["B2"] @ truth["Lambda2"]  # (len(x2), K)

    # Map truth into the model's coefficient space for alignment helper
    Lambda1_true_in_model = npl.pinv(f1._basis_matrix) @ trueBL1_domain
    Lambda2_true_in_model = npl.pinv(f2._basis_matrix) @ trueBL2_domain

    out = align_true_factors_to_varimaxrsp(
        model=model,
        Lambda_true_list=[Lambda1_true_in_model, Lambda2_true_in_model],
        B_true_list=[f1._basis_matrix, f2._basis_matrix],
    )
    BL1_true_aligned = out["BLambda_true_aligned_per_mode"][0]  # (len(x1), K)
    BL2_true_aligned = out["BLambda_true_aligned_per_mode"][1]  # (len(x2), K)

    # Posterior BLambda (VarimaxRSP)
    postBL1 = model.get_BLambda_d_VarimaxRSP(1)
    postBL2 = model.get_BLambda_d_VarimaxRSP(2)

    fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
    records: List[Dict] = []

    for d in range(2):
        coords = cfg.x1 if d == 0 else cfg.x2
        post = postBL1 if d == 0 else postBL2
        true_aligned = BL1_true_aligned if d == 0 else BL2_true_aligned

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
            post_mean = post_k.mean(axis=0)

            true_curve = true_aligned[:, k]
            within = np.logical_and(true_curve >= lower, true_curve <= upper)
            within_count = int(within.sum())
            n_coords = int(len(true_curve))
            coverage_pct = 100.0 * within_count / n_coords

            quad_error = float(np.trapz((post_mean - true_curve) ** 2, x=coords))
            mse = float(np.mean((post_mean - true_curve) ** 2))
            rmse = float(np.sqrt(mse))

            records.append(
                dict(
                    seed=seed,
                    model_tag=tag,
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
                    eval_version="lambda_projected_optimum",
                )
            )

            _plot_band_vs_truth(
                ax=ax,
                coords=coords,
                lower=lower,
                upper=upper,
                center=median,
                truth=true_curve,
                title=f"{tag} | Mode {d+1}, Factor {k+1} | Coverage={coverage_pct:.1f}%",
                ylabel="BLambda",
                show_legend=(k == 0),
                ci_level=cfg.ci_level,
                center_label="Posterior median",
            )

    fig.tight_layout()
    fig.savefig(seed_dir / f"posterior_vs_true_{tag}.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(records).to_csv(seed_dir / f"coverage_{tag}.csv", index=False)
    return records



def _evaluate_and_plot_blf_B_version(
    *,
    seed: int,
    cfg: ExperimentConfig,
    seed_dir: Path,
    model: mvblf.MvtBLF,
    truth: Dict,
    tag: str,
) -> List[Dict]:
    """
    BLambda-only evaluation:
      - aligns and compares using TRUE domain curves BLambda = (B_true @ Lambda_true)
      - NEVER projects into model coefficient space
    """
    lower_q = (100 - cfg.ci_level) / 2
    upper_q = 100 - lower_q

    # TRUE domain BLambda curves (actual truth on the observation grid)
    trueBL1_domain = truth["B1"] @ truth["Lambda1"]  # (len(x1), K)
    trueBL2_domain = truth["B2"] @ truth["Lambda2"]  # (len(x2), K)

    # Align using BLambda-only alignment to the model's VarimaxRSP BLambda reference
    outB = align_true_factors_to_varimaxrsp_B_version(
        model=model,
        BLambda_true_list=[trueBL1_domain, trueBL2_domain],
        method="varimax",
    )
    BL1_true_aligned = outB["BLambda_true_aligned_per_mode"][0]  # (len(x1), K)
    BL2_true_aligned = outB["BLambda_true_aligned_per_mode"][1]  # (len(x2), K)

    # Posterior BLambda (VarimaxRSP)
    postBL1 = model.get_BLambda_d_VarimaxRSP(1)
    postBL2 = model.get_BLambda_d_VarimaxRSP(2)

    fig, axs = plt.subplots(cfg.K, 2, figsize=(15, 4.5 * cfg.K), squeeze=False)
    records: List[Dict] = []

    for d in range(2):
        coords = cfg.x1 if d == 0 else cfg.x2
        post = postBL1 if d == 0 else postBL2
        true_aligned = BL1_true_aligned if d == 0 else BL2_true_aligned

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
            post_mean = post_k.mean(axis=0)

            true_curve = true_aligned[:, k]
            within = np.logical_and(true_curve >= lower, true_curve <= upper)
            within_count = int(within.sum())
            n_coords = int(len(true_curve))
            coverage_pct = 100.0 * within_count / n_coords

            quad_error = float(np.trapz((post_mean - true_curve) ** 2, x=coords))
            mse = float(np.mean((post_mean - true_curve) ** 2))
            rmse = float(np.sqrt(mse))

            records.append(
                dict(
                    seed=seed,
                    model_tag=tag,
                    eval_version="blambda_actual",
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
                truth=true_curve,
                title=f"{tag} | BLambda-ACTUAL | Mode {d+1}, Factor {k+1} | Coverage={coverage_pct:.1f}%",
                ylabel="BLambda",
                show_legend=(k == 0),
                ci_level=cfg.ci_level,
                center_label="Posterior median",
            )

    fig.tight_layout()
    fig.savefig(seed_dir / f"posterior_vs_true_B_{tag}.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(records).to_csv(seed_dir / f"coverage_B_{tag}.csv", index=False)
    return records











# def _write_method_summaries(df: pd.DataFrame, out_prefix: Path) -> None:
#     if df.empty:
#         return

#     grouped = df.groupby(["model_tag", "mode", "factor"], as_index=False).agg(
#         mean_coverage_pct=("coverage_pct", "mean"),
#         std_coverage_pct=("coverage_pct", "std"),
#         pooled_within=("within_count", "sum"),
#         pooled_coords=("n_coords", "sum"),
#         n_seeds=("seed", "nunique"),
#         mean_quad_error=("quad_error", "mean"),
#         std_quad_error=("quad_error", "std"),
#         pooled_quad_error=("quad_error", "sum"),
#     )
#     grouped["pooled_coverage_pct"] = 100.0 * grouped["pooled_within"] / grouped["pooled_coords"]
#     grouped["pooled_mse"] = grouped["pooled_quad_error"] / grouped["pooled_coords"]

#     overall = pd.DataFrame(
#         [{
#             "mean_coverage_pct_over_cells": df["coverage_pct"].mean(),
#             "std_coverage_pct_over_cells": df["coverage_pct"].std(),
#             "pooled_coverage_pct_overall": 100.0 * df["within_count"].sum() / df["n_coords"].sum(),
#             "n_cells": len(df),
#             "n_seeds_completed": df["seed"].nunique(),
#             "mean_quad_error_over_cells": df["quad_error"].mean(),
#             "std_quad_error_over_cells": df["quad_error"].std(),
#             "pooled_quad_error_overall": df["quad_error"].sum(),
#             "pooled_mse_overall": df["quad_error"].sum() / df["n_coords"].sum(),
#         }]
#     )

#     grouped.to_csv(Path(str(out_prefix) + "_summary_by_mode_factor.csv"), index=False)
#     overall.to_csv(Path(str(out_prefix) + "_summary_overall.csv"), index=False)



def _write_method_summaries(df: pd.DataFrame, out_prefix: Path) -> None:
    if df.empty:
        return

    # If older records exist without eval_version, tag them
    if "eval_version" not in df.columns:
        df = df.copy()
        df["eval_version"] = "lambda_projected_optimum"

    group_cols = ["eval_version", "model_tag", "mode", "factor"]

    grouped = df.groupby(group_cols, as_index=False).agg(
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

    overall = df.groupby(["eval_version"], as_index=False).agg(
        mean_coverage_pct_over_cells=("coverage_pct", "mean"),
        std_coverage_pct_over_cells=("coverage_pct", "std"),
        pooled_within=("within_count", "sum"),
        pooled_coords=("n_coords", "sum"),
        n_cells=("coverage_pct", "size"),
        n_seeds_completed=("seed", "nunique"),
        mean_quad_error_over_cells=("quad_error", "mean"),
        std_quad_error_over_cells=("quad_error", "std"),
        pooled_quad_error_overall=("quad_error", "sum"),
    )
    overall["pooled_coverage_pct_overall"] = 100.0 * overall["pooled_within"] / overall["pooled_coords"]
    overall["pooled_mse_overall"] = overall["pooled_quad_error_overall"] / overall["pooled_coords"]

    grouped.to_csv(Path(str(out_prefix) + "_summary_by_eval_mode_factor.csv"), index=False)
    overall.to_csv(Path(str(out_prefix) + "_summary_overall_by_eval.csv"), index=False)



# =============================================================================
# Per-seed runner
# =============================================================================

def run_seed(
    *,
    seed: int,
    cfg: ExperimentConfig,
    seed_dir: Path,
    B1_true: np.ndarray,
    B2_true: np.ndarray,
    alt_bases: Sequence[str],
) -> List[Dict]:
    """
    Runs 2 BLF models:
      - model_tag="gp": always GP basis
      - model_tag=f"alt_{alt_basis}": alt basis
    Returns list of records for both models.
    """
    Y1, Y2, truth = simulate_once(seed, cfg, B1_true, B2_true)
    all_records: List[Dict] = []

    # Model #1: GP basis (fixed)
    f1_gp = build_functional_data_gp(cfg.x1, Y1, cfg)
    f2_gp = build_functional_data_gp(cfg.x2, Y2, cfg)
    model_gp = _fit_blf_model(seed, cfg, modes=[f1_gp, f2_gp])

    # # Model #2: alt basis (chosen by user)
    # if alt_basis not in FIT_BASIS_SETTERS:
    #     raise ValueError(f"Unknown alt_basis='{alt_basis}'. Choices: {sorted(FIT_BASIS_SETTERS)}")

    # f1_alt = FIT_BASIS_SETTERS[alt_basis](cfg.x1, Y1, cfg, 1)
    # f2_alt = FIT_BASIS_SETTERS[alt_basis](cfg.x2, Y2, cfg, 2)
    # model_alt = _fit_blf_model(seed, cfg, modes=[f1_alt, f2_alt])

    # Evaluate + plot
    rec_gp = _evaluate_and_plot_blf(
        seed=seed, cfg=cfg, seed_dir=seed_dir,
        model=model_gp, f1=f1_gp, f2=f2_gp,
        truth=truth, tag="gp",
    )
    all_records.extend(rec_gp)

    rec_gp_B = _evaluate_and_plot_blf_B_version(
        seed=seed, cfg=cfg, seed_dir=seed_dir,
        model=model_gp,
        truth=truth,
        tag="gp",
    )
    all_records.extend(rec_gp_B)


    # rec_alt = _evaluate_and_plot_blf(
    #     seed=seed, cfg=cfg, seed_dir=seed_dir,
    #     model=model_alt, f1=f1_alt, f2=f2_alt,
    #     truth=truth, tag=f"alt_{alt_basis}",
    # )

    # # Cleanup
    # del model_gp, model_alt, f1_gp, f2_gp, f1_alt, f2_alt
    # gc.collect()

    # return rec_gp + rec_alt

    for alt_basis in alt_bases:
        if alt_basis not in FIT_BASIS_SETTERS:
            raise ValueError(f"Unknown alt_basis='{alt_basis}'. Choices: {sorted(FIT_BASIS_SETTERS)}")

        f1_alt = FIT_BASIS_SETTERS[alt_basis](cfg.x1, Y1, cfg, 1)
        f2_alt = FIT_BASIS_SETTERS[alt_basis](cfg.x2, Y2, cfg, 2)
        model_alt = _fit_blf_model(seed, cfg, modes=[f1_alt, f2_alt])

        rec_alt = _evaluate_and_plot_blf(
            seed=seed, cfg=cfg, seed_dir=seed_dir,
            model=model_alt, f1=f1_alt, f2=f2_alt,
            truth=truth, tag=f"alt_{alt_basis}",
        )
        all_records.extend(rec_alt)

        rec_alt_B = _evaluate_and_plot_blf_B_version(
            seed=seed, cfg=cfg, seed_dir=seed_dir,
            model=model_alt,
            truth=truth,
            tag=f"alt_{alt_basis}",
        )
        all_records.extend(rec_alt_B)

        # cleanup per alt
        del model_alt, f1_alt, f2_alt
        gc.collect()

    return all_records



# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment(
    cfg_in: ExperimentConfig,
    *,
    truth_mechanism: str,
    alt_bases: Sequence[str],
) -> Path:
    """
    Main entry point (programmatic):
      truth_mechanism: string selecting the data-generating mechanism.
      alt_basis: string selecting the basis for the second BLF model.
      GP-basis BLF is always run in addition to alt_basis BLF.
    """
    cfg = _ensure_truth_defaults(cfg_in)

    alt_bases = tuple(alt_bases)
    if len(alt_bases) != 2:
        raise ValueError(f"Expected exactly two alternative bases, got {len(alt_bases)}: {alt_bases}")

    if truth_mechanism not in TRUTH_MECHANISMS:
        raise ValueError(f"Unknown truth_mechanism='{truth_mechanism}'. Choices: {sorted(TRUTH_MECHANISMS)}")

    scenario_dir = cfg.output_root / cfg.scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Precompute truth bases for the chosen mechanism (shared across seeds)
    B1_true, B2_true, truth_meta = TRUTH_MECHANISMS[truth_mechanism](cfg)

    all_records: List[Dict] = []
    failures: List[Dict] = []

    for seed in tqdm(list(cfg.seeds), desc=f"{cfg.scenario_name} | truth={truth_mechanism} | alt={','.join(alt_bases)}"):
        seed_dir = scenario_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        try:
            rec = run_seed(
                seed=seed, cfg=cfg, seed_dir=seed_dir,
                B1_true=B1_true, B2_true=B2_true,
                alt_bases=alt_bases,
            )
            all_records.extend(rec)

        except Exception as e:
            failures.append({"seed": seed, "error": repr(e)})
            pd.DataFrame([{
                "seed": seed,
                "model_tag": np.nan,
                "eval_version": np.nan,
                "mode": np.nan,
                "factor": np.nan,
                "coverage_pct": np.nan,
                "quad_error": np.nan,
                "error": repr(e),
            }]).to_csv(seed_dir / "failure.csv", index=False)


    df = pd.DataFrame(all_records)

    if not df.empty and "eval_version" not in df.columns:
        df["eval_version"] = "lambda_projected_optimum"
        
    df.to_csv(scenario_dir / "coverage_all_seeds.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(scenario_dir / "failures.csv", index=False)

    _write_method_summaries(df, scenario_dir / "blf")

    manifest = {
        "scenario_name": cfg.scenario_name,
        "N": int(cfg.N),
        "K": int(cfg.K),
        "n_seeds": int(len(cfg.seeds)),
        "ci_level": float(cfg.ci_level),
        "truth_mechanism": truth_mechanism,
        "truth_meta": truth_meta,
        "evaluation_versions": ["lambda_projected_optimum", "blambda_actual"],
        "models": {
            "model_1": {
                "tag": "gp",
                "basis": "gp",
                "gp_type": cfg.gp_type,
                "gp_lengthscale": float(cfg.gp_lengthscale),
                "gp_variance": float(cfg.gp_variance),
            },
            "model_2": {
                "tag": f"alt_{alt_bases[0]}",
                "basis": alt_bases[0],
                "alt_n_basis_mode1": cfg.alt_n_basis_mode1,
                "alt_n_basis_mode2": cfg.alt_n_basis_mode2,
                "alt_bspline_degree": int(cfg.alt_bspline_degree),
            },
            "model_3": {
                "tag": f"alt_{alt_bases[1]}",
                "basis": alt_bases[1],
                "alt_n_basis_mode1": cfg.alt_n_basis_mode1,
                "alt_n_basis_mode2": cfg.alt_n_basis_mode2,
                "alt_bspline_degree": int(cfg.alt_bspline_degree),
            },
        },
        "stan": {
            "n_chains": int(cfg.n_chains),
            "iter_warmup": int(cfg.iter_warmup),
            "iter_sampling": int(cfg.iter_sampling),
            "thin": int(cfg.thin),
            "max_treedepth": int(cfg.max_treedepth),
        },
        "available_truth_mechanisms": sorted(TRUTH_MECHANISMS),
        "available_alt_bases": sorted(FIT_BASIS_SETTERS),
    }
    (scenario_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return scenario_dir


# =============================================================================
# CLI main
# =============================================================================

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Run Bayesian MvtBLF comparison: GP vs alternative basis, with chosen truth mechanism.")
    p.add_argument("--truth", required=True, choices=sorted(TRUTH_MECHANISMS), help="Truth mechanism (data-generating).")
    # p.add_argument("--alt-basis", required=True, choices=sorted(FIT_BASIS_SETTERS), help="Alt basis for the second MvtBLF.")
    p.add_argument(
        "--alt-basis",
        action="append",
        required=True,
        choices=sorted(FIT_BASIS_SETTERS),
        help="Alternative basis. Provide this flag twice to fit two alternative models.",
    )
    p.add_argument("--scenario", default="default", help="Scenario name (output folder).")

    # Common scenario knobs
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--seeds-from", type=int, default=1)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--ci", type=float, default=80.0)

    # Truth knobs (optional)
    p.add_argument("--bspline-true-n1", type=int, default=None)
    p.add_argument("--bspline-true-n2", type=int, default=None)
    p.add_argument("--bspline-true-degree", type=int, default=3)
    p.add_argument("--fourier-true-n1", type=int, default=None)
    p.add_argument("--fourier-true-n2", type=int, default=None)

    # Alt knobs
    p.add_argument("--alt-n1", type=int, default=None)
    p.add_argument("--alt-n2", type=int, default=None)
    p.add_argument("--alt-bspline-degree", type=int, default=3)

    # Output root
    p.add_argument("--out", type=str, default="Figures")

    return p


def main():
    args = _build_arg_parser().parse_args()

    if len(args.alt_basis) != 2:
        raise ValueError(f"Expected exactly two --alt-basis flags, got {len(args.alt_basis)}: {args.alt_basis}")

    seeds = tuple(range(args.seeds_from, args.seeds_from + args.n_seeds))
    cfg = ExperimentConfig(
        output_root=Path(args.out),
        scenario_name=args.scenario,
        N=args.N,
        K=args.K,
        seeds=seeds,
        ci_level=float(args.ci),
        bspline_true_degree=int(args.bspline_true_degree),
        bspline_true_n_basis_mode1=args.bspline_true_n1,
        bspline_true_n_basis_mode2=args.bspline_true_n2,
        fourier_true_n_basis_mode1=args.fourier_true_n1,
        fourier_true_n_basis_mode2=args.fourier_true_n2,
        alt_n_basis_mode1=args.alt_n1,
        alt_n_basis_mode2=args.alt_n2,
        alt_bspline_degree=int(args.alt_bspline_degree),
        gp_lengthscale=0.7 # added but to remove later
    )

    # out = run_experiment(cfg, truth_mechanism=args.truth, alt_basis=args.alt_basis)
    out = run_experiment(cfg, truth_mechanism=args.truth, alt_bases=tuple(args.alt_basis))
    print(f"Wrote results to: {out}")


if __name__ == "__main__":
    main()
