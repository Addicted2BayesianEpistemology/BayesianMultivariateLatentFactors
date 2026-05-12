

from typing import List, Tuple, Optional, Union, Literal
from xml.parsers.expat import model
from .stan_helper import _assembleMBLF_stan_code_for_given_D
from .stan_helper import _get_or_compile_cmdstan_model

import numpy as np
import arviz as az
import xarray as xr
import xarray_einstats as xe

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from scipy.spatial.distance import pdist

import pickle
import os


class FunctionalData:
    @staticmethod
    def _coords_to_matrix_if_not(
        coords: Union[np.ndarray, List[np.ndarray]],
        data: np.ndarray,
    ) -> np.ndarray:
        if isinstance(coords, np.ndarray):
            if not isinstance(data, np.ndarray):
                raise ValueError("if coords is an array, data must also be an array")
            # check that coords is 2D otherwise make it 2D
            if coords.ndim == 1:
                coords = coords[:, np.newaxis]
            elif coords.ndim > 2:
                raise ValueError("coords must be 1D or 2D array")

            # data must be 2D
            if data.ndim != 2:
                raise ValueError("data must be 2D array")
            
            # first dimension of coords must match first dimension of data
            if coords.shape[0] != data.shape[0]:
                raise ValueError("first dimension of coords must match first dimension of data")

            # change data so that any Inf is converted to NaN, BUT raise a warning if there is some Inf
            if np.isinf(data).any():
                import warnings
                warnings.warn("data contains Inf values, converting them to NaN")
            mask = ~np.isfinite(data)
            data[mask] = np.nan

            return coords, data
        else:
            # both must be lists of same length
            if not isinstance(data, list):
                raise ValueError("if coords is a list, data must also be a list")
            if not isinstance(coords, list):
                raise ValueError("if data is a list, coords must also be a list")
            if len(coords) != len(data):
                raise ValueError("coords and data must be lists of same length")

            ddim = None
            for c, d in zip(coords, data):
                if c.ndim == 1:
                    c = c[:, np.newaxis]
                elif c.ndim > 2:
                    raise ValueError("each coords array must be 1D or 2D array")

                if ddim is None:
                    ddim = c.shape[1]
                elif c.shape[1] != ddim:
                    raise ValueError("all coords arrays must have same second dimension")

                if d.ndim != 1:
                    raise ValueError("each data array must be 1D array")
                if c.shape[0] != d.shape[0]:
                    raise ValueError("first dimension of each coords array must match length of corresponding data array")
                if ~np.isfinite(d).all():
                    raise ValueError("data arrays must not contain NaN or Inf values")

            # make a list of unique coords across all dimensions and sort them using lexicographical order, then create a data matrix with NaN for missing values
            all_coords_list = set()
            for c in coords:
                for row in c:
                    all_coords_list.add(tuple(row))
            all_coords = np.array(sorted(list(all_coords_list)))
            all_data = np.full((all_coords.shape[0], len(data)), np.nan)
            for smpl_idx, (c, d) in enumerate(zip(coords, data)):
                for coord_row, data_value in zip(c, d):
                    coord_tuple = tuple(coord_row)
                    coord_idx = np.where((all_coords == coord_tuple).all(axis=1))[0][0]
                    all_data[coord_idx, smpl_idx] = data_value

            return all_coords, all_data

    def __init__(
            self,
            coords: Union[np.ndarray, List[np.ndarray]],
            data: Union[np.ndarray, List[np.ndarray]],
    ):
        self._coords, self._data = self._coords_to_matrix_if_not(coords, data)
    


    def set_raw_basis(self, basis_matrix: np.ndarray):
        # TO-DO add here some checks
        self._basis_matrix = basis_matrix

    def set_functional_basis(self, basis_functions: List[callable]):
        basis_matrix = np.column_stack(
            [f(self._coords) for f in basis_functions]
        )
        self._basis_matrix = basis_matrix

    def set_GP_basis(self, n_basis: int, gaussian_process: Union[Literal['quad_cov_kernel', 'matern5/2'], str], gp_lengthscale: Optional[float] = 1.0, gp_variance: Optional[float] = 1.0):
        """
        Construct the Gaussian Process basis matrix using the Singular Value Decomposition (SVD)
        of the covariance matrix defined by the RBF kernel with the specified lengthscale and variance.
        """
        if gaussian_process == 'quad_cov_kernel':
            from .gp_kernels import quad_cov_kernel
            cov_matrix = quad_cov_kernel(self._coords, self._coords, lengthscale=gp_lengthscale, variance=gp_variance)
        elif gaussian_process == 'matern5/2':
            from .gp_kernels import matern52_cov_kernel
            cov_matrix = matern52_cov_kernel(self._coords, self._coords, lengthscale=gp_lengthscale, variance=gp_variance)
        else:
            # assume it is a formula
            from .gp_kernels import formula_eval
            cov_matrix = formula_eval(gaussian_process, self._coords, self._coords)

        U, S, Vt = np.linalg.svd(cov_matrix)
        basis_matrix = U[:, :n_basis] * np.sqrt(S[:n_basis])
        self._basis_matrix = basis_matrix

    @property
    def domainDim(self) -> int:
        return self.coords.shape[1]
    
    @property
    def missingnessMask(self) -> np.ndarray:
        return np.isnan(self.data)
    
    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def empirical_basis_coefficients(self) -> np.ndarray:
        from numpy.linalg import lstsq
        coeffs, _, _, _ = lstsq(self._basis_matrix, np.where(np.isnan(self._data), 0.0, self._data), rcond=None)
        return coeffs

    @property
    def stan_friendly_data(self) -> dict:
        """
        Returns a dict with keys matching the Stan data block for ONE mode (without the mode suffix):
          M (int), p (int), B: (M, p), T: (p, N), Delta: (M, N),
          sum_rss (float), sum_Mdn (int)
        """
        if not hasattr(self, "_basis_matrix"):
            raise ValueError("Basis matrix not set. Please set the basis matrix using one of the set_basis methods.")
        B = self._basis_matrix  # (M, p)
        Y = self._data          # (M, N)
        if Y.ndim != 2:
            raise ValueError("data must be 2D (M x N)")
        if B.shape[0] != Y.shape[0]:
            raise ValueError("basis rows (M) must match data rows (M)")
        T = self.empirical_basis_coefficients  # (p, N)
        fit_Y = B @ T                           # (M, N)
        obs_mask = ~np.isnan(Y)                 # (M, N), 1 where observed

        resid = np.zeros_like(Y)
        resid[obs_mask] = (Y - fit_Y)[obs_mask]
        sum_rss = float(np.sum(resid[obs_mask] ** 2))
        sum_Mdn = int(np.sum(obs_mask))

        stan_data = {
            "M": B.shape[0],
            "N": Y.shape[1],
            "p": B.shape[1],
            "B": B,
            "T": T,
            "Delta": obs_mask.astype(int),
            "sum_rss": sum_rss,
            "sum_Mdn": sum_Mdn,
        }
        return stan_data

    def compute_semivariogram(
        self,
        n_lags: int = 15,
        max_range: Optional[float] = None,
        bin_edges: Optional[np.ndarray] = None,
        min_pairs: int = 30,
        estimator: Literal["matheron", "cressie"] = "matheron",
        mode: Literal["pooled", "per_sample"] = "pooled",
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],                 # pooled: (bins, gamma, counts)
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]      # per_sample: (bins, gamma_mean, counts_sum, gamma_per_sample)
    ]:
        """
        Empirical isotropic semivariogram using coords (M,d) and data (M,N).

        Parameters
        ----------
        n_lags : int
            Number of lag bins (ignored if `bin_edges` is provided).
        max_range : float, optional
            Maximum distance included. Default = 0.5 * max pairwise distance.
        bin_edges : array, optional
            Explicit bin edges (overrides `n_lags`/`max_range`).
        min_pairs : int
            Minimum total pairs required to report a bin (pooled) or per-sample bin (per_sample).
        estimator : {"matheron", "cressie"}
            Matheron: 0.5 * mean((z_i - z_j)^2). Cressie–Hawkins robust estimator.
        mode : {"pooled", "per_sample"}
            - "pooled": combine pairs across all samples (columns) into a single semivariogram.
            - "per_sample": compute a semivariogram for each sample; also returns their mean.

        Returns
        -------
        If mode == "pooled":
            bin_centers : (K,)
            gamma       : (K,)
            pairs_count : (K,)
        If mode == "per_sample":
            bin_centers : (K,)
            gamma_mean  : (K,)  # mean across samples (ignoring NaNs)
            pairs_sum   : (K,)  # total contributing pairs across samples
            gamma_ps    : (K, N)  # per-sample semivariograms (NaN where < min_pairs)
        """
        coords = self.coords
        Y = self.data  # (M, N) with NaNs allowed
        if Y.ndim != 2:
            raise ValueError("data must be 2D (M x N)")

        M, N = Y.shape

        # 1) Pairwise distances for coords (condensed form length M*(M-1)/2)
        h = pdist(coords, metric="euclidean")

        # 2) Bin setup
        if bin_edges is None:
            if max_range is None:
                max_range = 0.5 * float(h.max()) if h.size else 0.0
            bin_edges = np.linspace(0.0, max_range, n_lags + 1)
        else:
            bin_edges = np.asarray(bin_edges)
            if bin_edges.ndim != 1 or bin_edges.size < 2:
                raise ValueError("bin_edges must be a 1D array with at least two values")

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        K = bin_centers.size

        # Which bin each pair falls into
        bin_idx = np.digitize(h, bin_edges, right=False) - 1
        in_range = (bin_idx >= 0) & (bin_idx < K)

        # Precompute pair indices that correspond to pdist's condensed order
        iu, ju = np.triu_indices(M, k=1)  # shape (P,), matches pdist / squareform order

        # Utility lambdas for estimators
        def _matheron(dv):
            # gamma(h) = 0.5 * mean((dv)**2)
            return 0.5 * (dv ** 2)

        def _cressie_gamma_from_absdiff(absdiff):
            # Robust Cressie-Hawkins (1980):
            # gamma(h) = [ mean(|z_i - z_j|^{1/2}) ]^4 / c(n); with
            # c(n) = 0.457 + 0.494/n + 0.045/n^2
            # We compute this *after* aggregating absdiff^{1/2} per-bin.
            sqrt_abs = np.sqrt(absdiff)
            return sqrt_abs

        if mode == "pooled":
            # Accumulators across *all* samples for each bin
            if estimator == "matheron":
                num_sum = np.zeros(K, dtype=float)
                count = np.zeros(K, dtype=int)

                for n in range(N):
                    obs = ~np.isnan(Y[:, n])
                    if obs.sum() < 2:
                        continue
                    # pairs where both observed
                    valid_pairs = obs[iu] & obs[ju] & in_range
                    if not np.any(valid_pairs):
                        continue
                    dv = (Y[ju, n] - Y[iu, n])[valid_pairs]
                    b = bin_idx[valid_pairs]
                    contrib = _matheron(dv)
                    # accumulate per bin
                    for k in range(K):
                        sel = (b == k)
                        if np.any(sel):
                            num_sum[k] += contrib[sel].sum()
                            count[k]  += sel.sum()

                gamma = np.full(K, np.nan)
                ok = count >= min_pairs
                gamma[ok] = num_sum[ok] / count[ok]
                return bin_centers, gamma, count

            elif estimator == "cressie":
                # We'll aggregate sqrt(|dv|) to then apply the ^4 / c(n) correction.
                sqrtabs_sum = np.zeros(K, dtype=float)
                count = np.zeros(K, dtype=int)

                for n in range(N):
                    obs = ~np.isnan(Y[:, n])
                    if obs.sum() < 2:
                        continue
                    valid_pairs = obs[iu] & obs[ju] & in_range
                    if not np.any(valid_pairs):
                        continue
                    dv = (Y[ju, n] - Y[iu, n])[valid_pairs]
                    b = bin_idx[valid_pairs]
                    contrib = _cressie_gamma_from_absdiff(np.abs(dv))  # == sqrt(|dv|)
                    for k in range(K):
                        sel = (b == k)
                        if np.any(sel):
                            sqrtabs_sum[k] += contrib[sel].sum()
                            count[k]       += sel.sum()

                gamma = np.full(K, np.nan)
                ok = count >= min_pairs
                # mean(sqrt(|dv|))^4 / c(n)
                c = np.zeros_like(count, dtype=float)
                c[ok] = 0.457 + 0.494 / count[ok] + 0.045 / (count[ok] ** 2)
                mean_sqrt = np.zeros_like(sqrtabs_sum)
                mean_sqrt[ok] = sqrtabs_sum[ok] / count[ok]
                gamma[ok] = (mean_sqrt[ok] ** 4) / c[ok]
                return bin_centers, gamma, count

            else:
                raise ValueError("estimator must be 'matheron' or 'cressie'")

        elif mode == "per_sample":
            gamma_ps = np.full((K, N), np.nan)
            pairs_ps = np.zeros((K, N), dtype=int)

            for n in range(N):
                obs = ~np.isnan(Y[:, n])
                if obs.sum() < 2:
                    continue
                valid_pairs = obs[iu] & obs[ju] & in_range
                if not np.any(valid_pairs):
                    continue
                dv = (Y[ju, n] - Y[iu, n])[valid_pairs]
                b = bin_idx[valid_pairs]

                if estimator == "matheron":
                    contrib = _matheron(dv)
                    for k in range(K):
                        sel = (b == k)
                        pairs_ps[k, n] = int(sel.sum())
                        if pairs_ps[k, n] >= min_pairs:
                            gamma_ps[k, n] = contrib[sel].mean()

                elif estimator == "cressie":
                    contrib = _cressie_gamma_from_absdiff(np.abs(dv))  # sqrt|dv|
                    for k in range(K):
                        sel = (b == k)
                        pairs_ps[k, n] = int(sel.sum())
                        if pairs_ps[k, n] >= min_pairs:
                            # per-sample c(n) depends on *this* sample's N(h)
                            nn = pairs_ps[k, n]
                            c = 0.457 + 0.494 / nn + 0.045 / (nn ** 2)
                            mean_sqrt = contrib[sel].mean()
                            gamma_ps[k, n] = (mean_sqrt ** 4) / c
                else:
                    raise ValueError("estimator must be 'matheron' or 'cressie'")

            # Aggregate: mean across samples where defined
            with np.errstate(invalid="ignore"):
                gamma_mean = np.nanmean(gamma_ps, axis=1)
            pairs_sum = np.sum(pairs_ps, axis=1)
            return bin_centers, gamma_mean, pairs_sum, gamma_ps

        else:
            raise ValueError("mode must be 'pooled' or 'per_sample'")

    # ------- Semivariogram: plot -------
    def plot_semivariogram(
        self,
        *,
        n_lags: int = 15,
        max_range: Optional[float] = None,
        bin_edges: Optional[np.ndarray] = None,
        min_pairs: int = 30,
        estimator: Literal["matheron", "cressie"] = "matheron",
        mode: Literal["pooled", "per_sample"] = "pooled",
        show_pairs: bool = True,
        per_sample_alpha: float = 0.2,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Convenience plotter for the empirical semivariogram.

        Parameters mirror `compute_semivariogram`. If `mode="per_sample"`,
        thin gray lines show each sample's semivariogram and the thick line is
        the across-sample mean.
        """
        if mode == "pooled":
            bins, gamma, counts = self.compute_semivariogram(
                n_lags=n_lags, max_range=max_range, bin_edges=bin_edges,
                min_pairs=min_pairs, estimator=estimator, mode="pooled"
            )
        else:
            bins, gamma, counts, gamma_ps = self.compute_semivariogram(
                n_lags=n_lags, max_range=max_range, bin_edges=bin_edges,
                min_pairs=min_pairs, estimator=estimator, mode="per_sample"
            )

        if ax is None:
            fig, ax = plt.subplots()

        # Per-sample traces (if requested)
        if mode == "per_sample":
            for n in range(gamma_ps.shape[1]):
                ax.plot(bins, gamma_ps[:, n], linewidth=1, alpha=per_sample_alpha)

        # Main curve
        ax.plot(bins, gamma, marker="o", linewidth=2)
        ax.set_xlabel("Lag distance h")
        ax.set_ylabel("Semivariance γ(h)")
        title_mode = "Pooled" if mode == "pooled" else "Per-sample"
        ax.set_title(f"Empirical semivariogram ({title_mode}, {estimator})")
        ax.grid(True, alpha=0.3)

        # Optional pair counts on a secondary y-axis
        if show_pairs:
            ax2 = ax.twinx()
            ax2.bar(bins, counts, width=(bins[1]-bins[0]) if bins.size > 1 else 0.9,
                    alpha=0.15, align="center")
            ax2.set_ylabel("Pair count N(h)")
            ax2.set_zorder(ax.get_zorder() - 1)

        return ax



import numpy as np
from typing import Optional

# assuming FunctionalData is already defined above / imported
# from .functional_data import FunctionalData   # if in another module
# and OALS_PCA is in .oals_pca (adjust if needed)
# from .oals_pca import OALS_PCA


class VectorialData(FunctionalData):
    """
    Vector-valued data viewed as 'functional' over p abstract points.

    - data: (p, N) matrix (p variables/features, N samples).
    - coords: automatically set to I_p, so all p points are equally distant
      from each other in Euclidean space.

    Basis options:
    --------------
    1. Identity basis (default in __init__).
    2. PCA basis via OALS_PCA (handles missing values).
    """

    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        data : np.ndarray, shape (p, N)
            Matrix of p-dimensional observations (columns are samples).
            Missing entries should be encoded as np.nan.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("VectorialData expects data to be 2D (p x N).")

        p, N = data.shape

        # Coordinates: p abstract 'points' as the standard basis in R^p
        coords = np.eye(p)

        # Use FunctionalData machinery (coords, missingness handling, etc.)
        super().__init__(coords=coords, data=data)

        # Default basis: identity (option 1)
        self.set_identity_basis()

    # -------- Basis option 1: identity --------
    def set_identity_basis(self) -> None:
        """
        Set the basis matrix to the p x p identity matrix.

        This is the default used in __init__.
        """
        p = self.data.shape[0]
        self._basis_matrix = np.eye(p)

    # -------- Basis option 2: PCA via OALS_PCA --------
    def set_PCA_basis(
        self,
        n_components: Optional[int] = None,
        *,
        center: bool = False,
        max_iter: int = 2500,
        tol_percent: float = 1e-11,
        n_init: int = 10,
        init_max_iter: int = 100,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Set the basis matrix to the first q PCA components using OALS_PCA,
        which supports missing values (NaNs) in the data.

        Parameters
        ----------
        n_components : int or None, optional
            Number of principal components q to use (q <= min(p, N)).
            If None, use q = min(p, N).
        center : bool, default False
            Passed to OALS_PCA. If True, column-wise centering is applied
            (on the sample x feature matrix).
        max_iter, tol_percent, n_init, init_max_iter, random_state, verbose :
            Passed directly to OALS_PCA.

        Notes
        -----
        - `self.data` is of shape (p, N): p variables, N samples (columns).
        - OALS_PCA expects X as (n_samples, n_features), so we pass X = data.T,
          i.e. X has shape (N, p).
        - The resulting loadings `components_` have shape (p, q), which is then
          used as the basis matrix B (M=p, p_basis=q).
        """
        from .helpers import OALS_PCA  # adjust path to where OALS_PCA lives

        Y = self.data  # (p, N) possibly with NaNs
        if Y.ndim != 2:
            raise ValueError("data must be 2D (p x N) for PCA basis.")

        p, N = Y.shape

        # Ensure there is at least some observed data
        if not np.isfinite(Y).any():
            raise ValueError("PCA basis cannot be computed: data has no finite values.")

        if n_components is None:
            n_components = min(p, N)

        if n_components < 1 or n_components > min(p, N):
            raise ValueError(
                f"n_components must be between 1 and {min(p, N)}, got {n_components}"
            )

        # OALS_PCA expects (n_samples, n_features)
        X = Y.T  # shape (N, p): N samples, p features

        model = OALS_PCA(
            n_components=n_components,
            max_iter=max_iter,
            tol_percent=tol_percent,
            n_init=n_init,
            init_max_iter=init_max_iter,
            center=center,
            random_state=random_state,
            verbose=verbose,
        )
        model.fit(X)

        # Loadings matrix: (p, n_components) – exactly the basis we need
        self._basis_matrix = model.components_

        # Optionally store the PCA model if you want to reuse it later
        self._pca_model = model






# # sklearn-like interface for Multivariate Bayesian Latent Factor model
# class MvtBLF:
#     def __init__(
#         self,
#         n_components: int,
#         n_process: int = 4,
#         iter_warmup: int = 1000,
#         iter_sampling: int = 1000,
#         stan_seed: int = 12345,
#     ):
#         pass




class MvtBLF:
    """
    sklearn-like interface for the Multivariate Bayesian Latent Factor model.

    Notes
    -----
    * The Stan *code* is generated and compiled in `__init__` using the knobs
      that affect the program structure (D, heteroscedasticity pattern, MGPS,
      and whether Lambda has local t-shrinkage). This yields a cached
      CmdStan executable stored on disk.
    * Model *data* (X, per-mode B/T/Delta) are provided at `.fit(...)`.
    * All prior hyperparameters are explicit `__init__` arguments (no **kwargs)
      and are stored on the instance.
    """

    def __init__(
        self,
        n_components: int,
        D: int,
        heteroscedastic_thetas: Union[bool, List[bool], None] = None,
        MGPS: bool = True,
        local_shrinkage_LatentFactors: bool = True,
        absence_of_psi2: Optional[List[bool]] = None,
        # sampling / build config
        n_chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 1000,
        thin: int = 1,
        max_treedepth: int = 12,
        stan_seed: int = 12345,
        stanc_options: Union[dict, None] = None,
        cpp_options: Union[dict, None] = None,
        stanfilename: Union[str, None] = None,
        # ---- explicit prior hyperparameters (scalars or length-D vectors) ----
        a_psi: Union[float, List[float]] = 2.0,
        b_psi: Union[float, List[float]] = 2.0,
        a_tau: Union[float, List[float]] = 2.0,
        b_tau: Union[float, List[float]] = 2.0,
        # MGPS (used if MGPS=True)
        a1: float = 2.0,
        a2: float = 2.0,
        # Non-MGPS alternative (used if MGPS=False)
        a_lambdavar: float = 2.0,
        # rate parameter is used in both cases, the classical MGPS has b = 1.0
        b_lambdavar: Union[float, List[float]] = 1.0,
        # Student-t df for Lambda (used if local_shrinkage_LatentFactors=True)
        nu: float = 3.0,
        # Regression prior scale
        sigma_regr_coeffs: float = 10.0,
        # if not None, it's a list of length D, of lists of length K, each element is either None or a np.ndarray with shape (p_d,) representing the prior mean for Lambda_d[,k]
        decenter_Lambda: Optional[List[List[Union[None, np.ndarray]]]] = None, 
    ) -> None:
        from .stan_helper import (
            _assembleMBLF_stan_code_for_given_D,
            _get_or_compile_cmdstan_model,
        )

        self.n_components = int(n_components)  # K
        self.D = int(D)
        self.MGPS = bool(MGPS)
        self.local_shrinkage_LatentFactors = bool(local_shrinkage_LatentFactors)
        self.absence_of_psi2 = [bool(absence) for absence in absence_of_psi2] if absence_of_psi2 is not None else None
        self.n_chains = int(n_chains)
        self.iter_warmup = int(iter_warmup)
        self.iter_sampling = int(iter_sampling)
        self.thin = int(thin)
        self.max_treedepth = int(max_treedepth)
        self.stan_seed = int(stan_seed)
        self.stanc_options = stanc_options or {}
        self.cpp_options = cpp_options or {}

        # Normalize heteroscedastic flags
        if heteroscedastic_thetas is None:
            hetero_list = [True] * self.D
        elif isinstance(heteroscedastic_thetas, bool):
            hetero_list = [heteroscedastic_thetas] * self.D
        else:
            if len(heteroscedastic_thetas) != self.D:
                raise ValueError("length of heteroscedastic_thetas must equal D")
            if not all(isinstance(h, bool) for h in heteroscedastic_thetas):
                raise ValueError("heteroscedastic_thetas must be bools")
            hetero_list = list(heteroscedastic_thetas)
        self.heteroscedastic_thetas = hetero_list

        # ---- store priors explicitly on the instance (may be scalars or length-D) ----
        self.prior_a_psi = a_psi
        self.prior_b_psi = b_psi
        self.prior_a_tau = a_tau
        self.prior_b_tau = b_tau
        self.prior_a1 = float(a1)
        self.prior_a2 = float(a2)
        self.prior_a_lambdavar = float(a_lambdavar)
        # self.prior_b_lambdavar = float(b_lambdavar)
        # coerce b_lambdavar to length-D array
        if isinstance(b_lambdavar, float) or isinstance(b_lambdavar, int):
            self.prior_b_lambdavar = np.repeat(float(b_lambdavar), self.D)
        else:
            b_lambdavar_arr = np.asarray(b_lambdavar, dtype=float)
            if b_lambdavar_arr.shape != (self.D,):
                raise ValueError("b_lambdavar must be a scalar or length-D array")
            self.prior_b_lambdavar = b_lambdavar_arr
        # now self.prior_b_lambdavar is always a length-D array of floats make it to list
        self.prior_b_lambdavar = self.prior_b_lambdavar.tolist()
        self.prior_nu = float(nu)
        self.prior_sigma_regr_coeffs = float(sigma_regr_coeffs)

        # Compose a stable filename used in the on-disk cache
        if stanfilename is None:
            het_tag = "".join("1" if h else "0" for h in hetero_list)
            if self.absence_of_psi2 is None:
                stanfilename = \
                f"MBLF_D{self.D}_MGPS{int(self.MGPS)}_t{int(self.local_shrinkage_LatentFactors)}_het{het_tag}"
            else:
                stanfilename = \
                f"MBLF_D{self.D}_MGPS{int(self.MGPS)}_t{int(self.local_shrinkage_LatentFactors)}_het{het_tag}_psi2{''.join('0' if a else '1' for a in self.absence_of_psi2)}"
        self._stanfilename = stanfilename

        # check that decenter_Lambda is well formed
        if decenter_Lambda is not None:
            if not isinstance(decenter_Lambda, list):
                raise ValueError("decenter_Lambda must be a list of length D")
            if len(decenter_Lambda) != self.D:
                raise ValueError("decenter_Lambda must be a list of length D")
            for d in range(self.D):
                if decenter_Lambda[d] is None:
                    decenter_Lambda[d] = [None] * self.n_components
                if not isinstance(decenter_Lambda[d], list):
                    raise ValueError(f"decenter_Lambda[{d}] must be a list of length K")
                if len(decenter_Lambda[d]) != self.n_components:
                    raise ValueError(f"decenter_Lambda[{d}] must be a list of length K")
                for k in range(self.n_components):
                    if decenter_Lambda[d][k] is not None:
                        arr = np.asarray(decenter_Lambda[d][k], dtype=float)
                        if arr.ndim != 1:
                            raise ValueError(f"decenter_Lambda[{d}][{k}] must be a 1D array or None")
                        # if len(arr) != self.modes[d].p:
                        #     raise ValueError(f"decenter_Lambda[{d}][{k}] must have length equal to p_d={self.modes[d].p}")
                        decenter_Lambda[d][k] = arr
                    # else:
                    #     arr = np.zeros(self.modes[d].p, dtype=float)
            # all checks passed
        self.decenter_Lambda = decenter_Lambda

        # Generate Stan program and compile (with cache)
        self._stan_code = _assembleMBLF_stan_code_for_given_D(
            D=self.D,
            heteroscedastic_thetas=self.heteroscedastic_thetas,
            MGPS=self.MGPS,
            local_shrinkage_LatentFactors=self.local_shrinkage_LatentFactors,
            absence_of_psi2=self.absence_of_psi2,
            decenter_Lambda=self.decenter_Lambda
        )

        self._model = _get_or_compile_cmdstan_model(
            stan_code=self._stan_code,
            stanfilename=self._stanfilename,
            stanc_options=self.stanc_options,
            cpp_options=self.cpp_options,
            quiet=True,
        )

        # Will be populated after fit
        self._fit = None
        self._dims = {}
        self.idata = None
        self._modes = None

    # -------------------------- utilities --------------------------
    def _coerce_vec(self, x, name: str) -> np.ndarray:
        """Accept a scalar or length-D sequence and return a length-D float array (>0)."""
        if np.isscalar(x):
            arr = np.repeat(float(x), self.D)
        else:
            arr = np.asarray(x, dtype=float)
            if arr.shape != (self.D,):
                raise ValueError(f"{name} must be a scalar or length-D array")
        if (arr <= 0).any():
            raise ValueError(f"{name} entries must be > 0")
        return arr

    def _pos(self, x, name: str) -> float:
        v = float(x)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return v

    def _build_stan_data(self, X: np.ndarray, modes: List["FunctionalData"]) -> dict:
        if len(modes) != self.D:
            raise ValueError(f"expected {self.D} FunctionalData modes, got {len(modes)}")

        # Pull per-mode stan-friendly payloads and validate N consistency
        per_mode = [m.stan_friendly_data for m in modes]
        N_vals = [pm["N"] for pm in per_mode]
        if len(set(N_vals)) != 1:
            raise ValueError(f"all modes must have the same N; got {N_vals}")
        N = int(N_vals[0])

        # Shape-check predictors X as (L, N); auto-transpose if (N, L)
        if not X is None:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be a 2D array of shape (L, N) or (N, L)")
            if X.shape[1] == N:
                L = int(X.shape[0])
            elif X.shape[0] == N:
                X = X.T
                L = int(X.shape[0])
            else:
                raise ValueError(f"X's length along one axis must equal N={N}; got {X.shape}")
        else:
            L = 0
            X = np.empty((0, N), dtype=float)

        data = {
            "N": N,
            "D": self.D,
            "K": self.n_components,
            "L": L,
            "X": X,
            # vectorized priors (allow scalar or length-D)
            "a_psi": self._coerce_vec(self.prior_a_psi, "a_psi"),
            "b_psi": self._coerce_vec(self.prior_b_psi, "b_psi"),
            "a_tau": self._coerce_vec(self.prior_a_tau, "a_tau"),
            "b_tau": self._coerce_vec(self.prior_b_tau, "b_tau"),
            "sigma_regr_coeffs": self._pos(self.prior_sigma_regr_coeffs, "sigma_regr_coeffs"),
        }
        if self.MGPS:
            data.update({
                "a1": self._pos(self.prior_a1, "a1"),
                "a2": self._pos(self.prior_a2, "a2"),
            })
        else:
            data.update({
                "a_lambdavar": self._pos(self.prior_a_lambdavar, "a_lambdavar"),
            })
        if self.local_shrinkage_LatentFactors:
            data.update({"nu": self._pos(self.prior_nu, "nu")})

        # Unroll per-mode blocks: p{d}, M{d}, B{d}, T{d}, Delta{d}, sum_rss{d}, sum_Mdn{d}
        for d, pm in enumerate(per_mode, start=1):
            # Cast Delta to real-valued matrix to match Stan's `matrix` type
            Delta_real = np.asarray(pm["Delta"], dtype=float)
            data.update(
                {
                    f"p{d}": int(pm["p"]),
                    f"M{d}": int(pm["M"]),
                    f"B{d}": np.asarray(pm["B"], dtype=float),
                    f"T{d}": np.asarray(pm["T"], dtype=float),
                    f"Delta{d}": Delta_real,
                    f"sum_rss{d}": float(pm["sum_rss"]),
                    f"sum_Mdn{d}": int(pm["sum_Mdn"]),
                    f"b_lambdavar{d}": float(self.prior_b_lambdavar[d-1]),
                }
            )
            if self.decenter_Lambda is not None:
                if self.decenter_Lambda[d-1] is not None:
                    matrice = np.zeros((pm["p"], self.n_components), dtype=float)
                    for k in range(self.n_components):
                        if self.decenter_Lambda[d-1][k] is not None:
                            matrice[:, k] = self.decenter_Lambda[d-1][k]
                    data[f"Lambda{d}_prior_means"] = matrice

        # Keep a small memo of shapes for downstream checks
        self._dims = {
            "N": N,
            "L": L,
            "D": self.D,
            "per_mode": [
                {"M": int(pm["M"]), "p": int(pm["p"]) } for pm in per_mode
            ],
        }
        return data

    # --------------------------- public API ---------------------------
    def fit(
        self,
        X: np.ndarray,
        modes: List["FunctionalData"],
        *,
        seed: Union[int, None] = None,
        show_progress: bool = True,
        show_console: bool = False,
        show_messages: bool = True,
        **sample_kwargs,
    ) -> "MvtBLF":
        """Build Stan data from `X` and `modes`, then sample.

        Parameters
        ----------
        X : array-like, shape (L, N) or (N, L)
            Predictor matrix. If (N, L) is given, it's auto-transposed.
        modes : list of FunctionalData, length D
            Each mode supplies B/T/Delta and sufficient stats via
            `.stan_friendly_data`.
        seed : int, optional
            RNG seed; defaults to `stan_seed` from the constructor.
        show_progress : bool
            Forwarded to CmdStan `show_progress`.
        **sample_kwargs :
            Extra kwargs supported by `CmdStanModel.sample` (e.g.,
            `adapt_delta`, `max_treedepth`, `thin`, etc.).
        """

        if self.decenter_Lambda is not None:
            if not isinstance(self.decenter_Lambda, list):
                raise ValueError("decenter_Lambda must be a list of length D")
            if len(self.decenter_Lambda) != self.D:
                raise ValueError("decenter_Lambda must be a list of length D")
            for d in range(self.D):
                if not isinstance(self.decenter_Lambda[d], list):
                    raise ValueError(f"decenter_Lambda[{d}] must be a list of length K")
                if len(self.decenter_Lambda[d]) != self.n_components:
                    raise ValueError(f"decenter_Lambda[{d}] must be a list of length K")
                for k in range(self.n_components):
                    if self.decenter_Lambda[d][k] is not None:
                        arr = np.asarray(self.decenter_Lambda[d][k], dtype=float)
                        if arr.ndim != 1:
                            raise ValueError(f"decenter_Lambda[{d}][{k}] must be a 1D array or None")
                        if len(arr) != modes[d].p:
                            raise ValueError(f"decenter_Lambda[{d}][{k}] must have length equal to p_d={modes[d].p}")
                        self.decenter_Lambda[d][k] = arr
                    else:
                        arr = np.zeros(modes[d].p, dtype=float)
                        self.decenter_Lambda[d][k] = arr

        data = self._build_stan_data(X, modes)

        self._modes = modes
        self._covariates = data['X']

        chains = self.n_chains
        if seed is None:
            seed = self.stan_seed

        if show_messages:
            import logging
            logging.getLogger("cmdstanpy").disabled = False
        else:
            import logging
            logging.getLogger("cmdstanpy").disabled = True

        # Sample
        self._fit = self._model.sample(
            data=data,
            chains=int(chains),
            parallel_chains=int(chains),
            max_treedepth=int(self.max_treedepth),
            seed=int(seed),
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling * self.thin,
            thin=self.thin,
            show_progress=bool(show_progress),
            show_console=bool(show_console),
            **sample_kwargs,
        )

        self.idata = az.from_cmdstanpy(self._fit)

        return self

    # ------------------------ convenience accessors ------------------------
    @property
    def model(self):
        """The compiled CmdStanModel."""
        return self._model

    @property
    def fit_result(self):
        """The CmdStanMCMC result returned by `.fit(...)` (or None if not fit)."""
        return self._fit

    @property
    def stan_code(self) -> str:
        """Return the exact Stan program used for this instance."""
        return self._stan_code
    
    def get_Lambda_d(self, d: int) -> xr.DataArray:
        """
        Return the posterior samples of the factor loadings for mode d
        """
        if d < 1 or d > self.D:
            raise ValueError(f"d must be between 1 and {self.D}")
        lambda_d = self.idata.posterior[f'Lambda{d}'].copy()
        lambda_d = lambda_d.rename({f'Lambda{d}_dim_0': 'p', f'Lambda{d}_dim_1': 'K'})
        return lambda_d
    
    def get_BLambda_d(self, d: int) -> xr.DataArray:
        """
        Return the posterior samples of the basis expansion of the factor loadings for mode d
        """
        Lambda_d_xr = self.get_Lambda_d(d)
        basis_matrix_d = self._modes[d-1]._basis_matrix  # (M, p)
        basis_matrix_d_xr = xr.DataArray(
            basis_matrix_d,
            dims=['M', 'p'],
        )
        BLambda_d = xe.matmul(
            basis_matrix_d_xr,
            Lambda_d_xr,
            dims=[['M', 'p'], ['p', 'K']],
        )
        return BLambda_d
    
    @property
    def LambdaStacked(self) -> xr.DataArray:
        """
        Return the posterior samples of the factor loadings stacked across Lambda{d}_dim_0 (notice that acutally Lambda{d}_dim_1 is the same across all d in theory)
        """
        lambdas_xr_list = []
        for d in range(1, self.D + 1):
            Lambda_d_xr = self.get_Lambda_d(d)
            lambdas_xr_list.append(Lambda_d_xr)
        lambda_stacked = xr.concat(lambdas_xr_list, dim='p')
        return lambda_stacked

    def _computeVarimax(self):
        from .helpers import _ortho_rotation
        LambdaStacked = self.LambdaStacked
        # apply the varimax for each chain and draw
        rotations_lists = []
        lambdas_lists = []
        for c in range(self.n_chains):
            rotations_d_list = []
            lambdas_d_list = []
            for d in range(self.iter_sampling):
                Lambda_sample = LambdaStacked.isel(chain=c, draw=d).values  # (p, K)
                Lambda_rotated, rotation_matrix = _ortho_rotation(Lambda_sample, method="varimax")
                lambdas_d_list.append(Lambda_rotated)
                rotations_d_list.append(rotation_matrix)
            rotations_lists.append(rotations_d_list)
            lambdas_lists.append(lambdas_d_list)

        rotated_lambdas = np.stack([np.stack(lst, axis=0) for lst in lambdas_lists], axis=0)  # (chain, draw, p, K)
        rotations = np.stack([np.stack(lst, axis=0) for lst in rotations_lists], axis=0)      # (chain, draw, K, K)

        chain_coords = LambdaStacked.coords["chain"] if "chain" in LambdaStacked.coords else np.arange(self.n_chains)
        draw_coords = LambdaStacked.coords["draw"] if "draw" in LambdaStacked.coords else np.arange(self.iter_sampling)
        p_coords = LambdaStacked.coords["p"] if "p" in LambdaStacked.coords else np.arange(LambdaStacked.sizes["p"])
        K_coords = LambdaStacked.coords["K"] if "K" in LambdaStacked.coords else np.arange(self.n_components)

        RotatedLambdas_xr = xr.DataArray(
            rotated_lambdas,
            dims=("chain", "draw", "p", "K"),
            coords={"chain": chain_coords, "draw": draw_coords, "p": p_coords, "K": K_coords},
        )

        Rotations_xr = xr.DataArray(
            rotations,
            dims=("chain", "draw", "K", "K_rot"),
            coords={"chain": chain_coords, "draw": draw_coords, "K": K_coords, "K_rot": K_coords.rename({'K': 'K_rot'})},
        )

        return RotatedLambdas_xr, Rotations_xr 
    
    from functools import cached_property
    @cached_property
    def computeVarimaxRSP(self):
        """
        Run the Varimax + Rotation–Sign–Permutation (Varimax-RSP) algorithm
        of Papastamoulis & Ntzoufras (2022, Algorithm 1) on the posterior
        draws of the stacked loading matrix.

        Returns
        -------
        lambdas_rsp : xr.DataArray
            Varimax-RSP–corrected loadings stacked over all modes, dims
            ("chain", "draw", "p", "K").
        rotations : xr.DataArray
            Varimax rotation matrices R^(t) as returned by `_computeVarimax`,
            dims ("chain", "draw", "K", "K_rot").
        signed_permutations : xr.DataArray
            Signed permutation matrices Q^(t), dims
            ("chain", "draw", "K", "K_rsp").
        reference_loadings : xr.DataArray
            Final reference loading matrix Λ* with dims ("p", "K").
        """
        import numpy as np
        import xarray as xr
        from itertools import product

        from scipy.optimize import linear_sum_assignment

        # Step 1 of Algorithm 1: varimax rotation per draw
        RotatedLambdas_xr, Rotations_xr = self._computeVarimax()  # note: call

        Lambda = RotatedLambdas_xr
        n_chain = Lambda.sizes["chain"]
        n_draw = Lambda.sizes["draw"]
        p = Lambda.sizes["p"]
        K = Lambda.sizes["K"]

        # We implement the exact Scheme A (Algorithm 2) for moderate K
        max_exact_K = 10
        if K > max_exact_K:
            raise ValueError(
                f"VarimaxRSP exact implementation is currently supported "
                f"for K <= {max_exact_K}; got K = {K}. "
                "For larger K you may want to add a simulated–annealing "
                "approximation as in Papastamoulis & Ntzoufras (2022)."
            )

        # Flatten (chain, draw) -> t = 0, ..., T-1
        lam_flat = Lambda.values.reshape(n_chain * n_draw, p, K)
        T = lam_flat.shape[0]

        # ----- helpers implementing Algorithm 1 / Scheme A -----
        def compute_ref(lam_flat, s, nu):
            """
            RLME step: given {s^(t), nu^(t)}, compute Λ* as
            λ*_rj = (1/T) sum_t s_j^(t) λ̃_{r,nu_j^(t)}^(t)
            """
            ref = np.zeros((p, K), dtype=float)
            for t in range(T):
                for j in range(K):
                    ref[:, j] += s[t, j] * lam_flat[t, :, nu[t, j]]
            ref /= float(T)
            return ref

        def total_loss(lam_flat, ref, s, nu):
            """
            Ψ(Λ*, s, ν) = sum_t L_s,ν^(t)
            where L_s,ν^(t) = sum_r sum_j (s_j λ̃_{r,nu_j}^(t) - λ*_rj)^2
            """
            loss = 0.0
            for t in range(T):
                for j in range(K):
                    col = s[t, j] * lam_flat[t, :, nu[t, j]]
                    diff = col - ref[:, j]
                    loss += float(np.dot(diff, diff))
            return loss

        def optimize_one_draw(lam_t, ref):
            """
            Exact SP step (Scheme A) for a single draw t:

            (s^(t), nu^(t)) = argmin_{s,ν} L_s,ν^(t)
            where we enumerate s ∈ {−1,1}^K and, for each s,
            solve the assignment problem over ν using a cost matrix.
            """
            best_loss = np.inf
            best_s = None
            best_nu = None

            for signs in product([-1.0, 1.0], repeat=K):
                s_vec = np.asarray(signs, dtype=float)

                # Build K x K cost matrix C[j, i] as in Eq. (16)
                # j: reference column index, i: column index of λ̃^(t)
                C = np.empty((K, K), dtype=float)
                for j in range(K):
                    ref_col = ref[:, j]                    # (p,)
                    diff = s_vec[j] * lam_t - ref_col[:, None]  # (p, K)
                    C[j, :] = np.sum(diff * diff, axis=0)

                row_ind, col_ind = linear_sum_assignment(C)
                perm = np.empty(K, dtype=int)
                perm[row_ind] = col_ind
                loss = float(C[row_ind, col_ind].sum())

                if loss < best_loss:
                    best_loss = loss
                    best_s = s_vec
                    best_nu = perm

            return best_s, best_nu, best_loss

        # ----- Step 2 of Algorithm 1: Signed permutation step -----

        # Step 2.1: initialization  s_j^(t) = 1, ν_j^(t) = j
        s = np.ones((T, K), dtype=float)
        nu = np.tile(np.arange(K, dtype=int), (T, 1))

        tol = 1e-6 * T * p * K  # same idea as in the paper
        max_iter = 100

        for _ in range(max_iter):
            # RLME step (2.2.1): update Λ* given current {s^(t), ν^(t)}
            ref = compute_ref(lam_flat, s, nu)

            # Current objective Ψ before we change s, ν at this iteration
            current_loss = total_loss(lam_flat, ref, s, nu)

            # SP step (2.2.2): for each t, solve (s^(t), ν^(t))
            new_loss = 0.0
            for t in range(T):
                s_t, nu_t, loss_t = optimize_one_draw(lam_flat[t], ref)
                s[t, :] = s_t
                nu[t, :] = nu_t
                new_loss += loss_t

            # Stopping rule: no relevant improvement in Ψ
            if current_loss - new_loss < tol:
                break

        # Build final Λ°(t) = Λ̃^(t) S^(t) P^(t) and Q^(t) = S^(t) P^(t)
        lam_rsp = np.zeros_like(lam_flat, dtype=float)
        Q = np.zeros((T, K, K), dtype=float)

        for t in range(T):
            for j in range(K):
                col_idx = int(nu[t, j])
                sign = float(s[t, j])
                lam_rsp[t, :, j] = sign * lam_flat[t, :, col_idx]
                Q[t, col_idx, j] = sign  # Q^(t) is signed permutation

        # Reshape back to (chain, draw, p, K) / (chain, draw, K, K)
        lam_rsp = lam_rsp.reshape(n_chain, n_draw, p, K)
        Q = Q.reshape(n_chain, n_draw, K, K)

        chain_coords = Lambda.coords["chain"]
        draw_coords = Lambda.coords["draw"]
        p_coords = Lambda.coords["p"]
        K_coords = Lambda.coords["K"]

        lambdas_rsp_xr = xr.DataArray(
            lam_rsp,
            dims=("chain", "draw", "p", "K"),
            coords={
                "chain": chain_coords,
                "draw": draw_coords,
                "p": p_coords,
                "K": K_coords,
            },
        )

        Q_xr = xr.DataArray(
            Q,
            dims=("chain", "draw", "K_rot", "K_rsp"),
            coords={
                "chain": chain_coords,
                "draw": draw_coords,
                "K_rot": K_coords.rename({'K': 'K_rot'}),
                "K_rsp": K_coords.rename({'K': 'K_rsp'}),
            },
        )

        ref_xr = xr.DataArray(
            ref,
            dims=("p", "K"),
            coords={"p": p_coords, "K": K_coords},
        )

        return lambdas_rsp_xr, Rotations_xr, Q_xr, ref_xr

    def _apply_rotation_sign_permutation_and_varimax_from_computed_VarimaxRSP(self, dataarray, name_of_coordinate_which_is_not_K: str) -> xr.DataArray:
        """
        dataarray is an xr.DataArray with K as one of the dimensions
        """        

        _, Rotations_xr, Q_xr, _ = self.computeVarimaxRSP

        # Apply to dataarray
        dataarray_rotated = xe.matmul(
            dataarray,
            xe.matmul(Rotations_xr, Q_xr, dims=[['K', 'K_rot'], ['K_rot', 'K_rsp']]),
            dims=[[name_of_coordinate_which_is_not_K, 'K'], ['K', 'K_rsp']],
        )

        return dataarray_rotated

    def get_BLambda_d_VarimaxRSP(self, d: int) -> xr.DataArray:
        """
        Get the B-Lambda for a specific draw d after applying Varimax rotation and signed permutation.
        """

        BLambda_d = self.get_BLambda_d(d)
        BLambda_d_rotated = self._apply_rotation_sign_permutation_and_varimax_from_computed_VarimaxRSP(
            BLambda_d,
            name_of_coordinate_which_is_not_K='M',
        )
        return BLambda_d_rotated

    def get_Lambda_d_VarimaxRSP(self, d: int) -> xr.DataArray:
        """
        Get the Lambda for a specific draw d after applying Varimax rotation and signed permutation.
        """

        Lambda_d = self.get_Lambda_d(d)
        Lambda_d_rotated = self._apply_rotation_sign_permutation_and_varimax_from_computed_VarimaxRSP(
            Lambda_d,
            name_of_coordinate_which_is_not_K='p',
        )
        return Lambda_d_rotated

    def is_mode_decentered(self, d: int) -> bool:
        """
        Check if mode d has decentered loadings
        """
        if self.decenter_Lambda is None:
            return False
        for k in range(self.n_components):
            if self.decenter_Lambda[d-1][k] is not None:
                return True
        return False

    def well_built_Lambda_decentering(self, d: int, p: int) -> np.ndarray:
        """
        Return the decentering matrix for mode d
        """
        if not self.is_mode_decentered(d):
            raise ValueError(f"mode {d} is not decentered")
        matrice = np.zeros((p, self.n_components), dtype=float)
        for k in range(self.n_components):
            if self.decenter_Lambda[d-1][k] is not None:
                matrice[:, k] = self.decenter_Lambda[d-1][k]
        return matrice

    def simulate_dataset(
        self,
        N: int,
        basis_matrices: Optional[List[np.ndarray]] = None,
        *,
        basis_functions_list: Optional[List[List[callable]]] = None,
        gp_specs_list: Optional[List[dict]] = None,
        coords_list: Optional[List[np.ndarray]] = None,
        L: Optional[int] = None,
        X: Optional[np.ndarray] = None,
        missing_prob: float = 0.0,
        seed: Optional[int] = None,
        return_latent: bool = True,
        return_stan_data: bool = False,
    ) -> dict:
        """
        Simulate a dataset from the generative model of this MvtBLF instance.

        Basis specification (exactly ONE of the following must be non-None)
        -------------------------------------------------------------------
        basis_matrices : list of np.ndarray, length D
            Each element is B_d, shape (M_d, p_d).
        basis_functions_list : list of list of callables, length D
            Each element is a list of basis functions [f1, ..., f_p_d] for
            mode d. Requires coords_list.
        gp_specs_list : list of dict, length D
            Each dict must at least contain 'n_basis', may contain:
              - 'gaussian_process' (str, default 'quad_cov_kernel')
              - 'gp_lengthscale' (float, default 1.0)
              - 'gp_variance' (float, default 1.0)
            Requires coords_list.

        Other parameters
        ----------------
        N : int
            Number of observations.
        coords_list : list of np.ndarray, optional
            Coordinates per mode; required for basis_functions_list or
            gp_specs_list. Each array is (M_d,) or (M_d, domainDim).
        L : int, optional
            Number of predictors (used if X is None).
        X : np.ndarray, optional
            Predictor matrix, shape (L, N) or (N, L). If None, generated
            i.i.d. N(0, 1) with shape (L, N).
        missing_prob : float in [0, 1]
            Per-entry probability of missingness in Y (NaN).
        seed : int, optional
            RNG seed; defaults to self.stan_seed.
        return_latent, return_stan_data : bool
            Whether to return latent variables and/or Stan data.

        Returns
        -------
        out : dict
            Always contains:
              - "X": predictor matrix, shape (L, N)
              - "modes": list[FunctionalData], length D
            Optionally:
              - "latent": dict of true parameters (if return_latent)
              - "stan_data": dict for Stan (if return_stan_data)
        """
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if not (0.0 <= missing_prob <= 1.0):
            raise ValueError("missing_prob must be in [0, 1]")

        if seed is None:
            seed = self.stan_seed
        rng = np.random.default_rng(int(seed))

        D = self.D
        K = self.n_components

        # ---- choose how to build the basis ----
        basis_flags = [
            basis_matrices is not None,
            basis_functions_list is not None,
            gp_specs_list is not None,
        ]
        if sum(basis_flags) != 1:
            raise ValueError(
                "Exactly one of basis_matrices, basis_functions_list, "
                "gp_specs_list must be provided."
            )

        # Build B_list and coords_arr_list
        B_list: List[np.ndarray] = []
        coords_arr_list: List[np.ndarray] = []

        # Helper to normalize coords_list against given Ms
        def _normalize_coords_with_M(coords_list_in, Ms_in):
            out = []
            for d, (coords, M_d) in enumerate(zip(coords_list_in, Ms_in)):
                c = np.asarray(coords, dtype=float)
                if c.ndim == 1:
                    c = c[:, np.newaxis]
                elif c.ndim > 2:
                    raise ValueError("each coords array must be 1D or 2D")
                if c.shape[0] != M_d:
                    raise ValueError(
                        f"coords_list[{d}] has first dimension {c.shape[0]}, "
                        f"expected {M_d}"
                    )
                out.append(c)
            return out

        if basis_matrices is not None:
            if len(basis_matrices) != D:
                raise ValueError(f"expected {D} basis_matrices, got {len(basis_matrices)}")
            B_list = [np.asarray(B, dtype=float) for B in basis_matrices]
            Ms = [B.shape[0] for B in B_list]
            ps = [B.shape[1] for B in B_list]

            if coords_list is not None:
                if len(coords_list) != D:
                    raise ValueError("coords_list must have length D")
                coords_arr_list = _normalize_coords_with_M(coords_list, Ms)
            else:
                # default coords: 1D grid on [0, 1]
                coords_arr_list = [
                    np.linspace(0.0, 1.0, M_d)[:, np.newaxis] for M_d in Ms
                ]

        elif basis_functions_list is not None:
            if coords_list is None:
                raise ValueError(
                    "coords_list must be provided when basis_functions_list is used."
                )
            if len(basis_functions_list) != D:
                raise ValueError(
                    "basis_functions_list must have length D (one list of "
                    "callables per mode)."
                )
            if len(coords_list) != D:
                raise ValueError("coords_list must have length D")

            # normalize coords
            for coords in coords_list:
                c = np.asarray(coords, dtype=float)
                if c.ndim == 1:
                    c = c[:, np.newaxis]
                elif c.ndim > 2:
                    raise ValueError("each coords array must be 1D or 2D")
                coords_arr_list.append(c)

            # build basis via FunctionalData.set_functional_basis
            for d in range(D):
                coords_d = coords_arr_list[d]
                M_d = coords_d.shape[0]
                tmp_data = np.zeros((M_d, 1), dtype=float)
                fd_tmp = FunctionalData(coords_d, tmp_data)
                fd_tmp.set_functional_basis(basis_functions_list[d])
                B_d = np.asarray(fd_tmp._basis_matrix, dtype=float)
                B_list.append(B_d)

            Ms = [B.shape[0] for B in B_list]
            ps = [B.shape[1] for B in B_list]

        else:  # gp_specs_list is not None
            if coords_list is None:
                raise ValueError(
                    "coords_list must be provided when gp_specs_list is used."
                )
            if len(gp_specs_list) != D:
                raise ValueError("gp_specs_list must have length D")
            if len(coords_list) != D:
                raise ValueError("coords_list must have length D")

            # normalize coords
            for coords in coords_list:
                c = np.asarray(coords, dtype=float)
                if c.ndim == 1:
                    c = c[:, np.newaxis]
                elif c.ndim > 2:
                    raise ValueError("each coords array must be 1D or 2D")
                coords_arr_list.append(c)

            # build basis via FunctionalData.set_GP_basis
            for d in range(D):
                spec = dict(gp_specs_list[d])  # copy
                if "n_basis" not in spec:
                    raise ValueError("each gp_specs_list[d] dict must contain 'n_basis'")

                n_basis = int(spec.pop("n_basis"))
                gaussian_process = spec.pop("gaussian_process", "quad_cov_kernel")
                gp_lengthscale = spec.pop("gp_lengthscale", 1.0)
                gp_variance = spec.pop("gp_variance", 1.0)

                coords_d = coords_arr_list[d]
                M_d = coords_d.shape[0]
                tmp_data = np.zeros((M_d, 1), dtype=float)
                fd_tmp = FunctionalData(coords_d, tmp_data)
                fd_tmp.set_GP_basis(
                    n_basis=n_basis,
                    gaussian_process=gaussian_process,
                    gp_lengthscale=gp_lengthscale,
                    gp_variance=gp_variance,
                )
                B_d = np.asarray(fd_tmp._basis_matrix, dtype=float)
                B_list.append(B_d)

            Ms = [B.shape[0] for B in B_list]
            ps = [B.shape[1] for B in B_list]

        # ---- handle predictors X ----
        if X is None:
            if L is None:
                raise ValueError("either X or L must be provided")
            L = int(L)
            if L < 0:
                raise ValueError("L must be a positive integer")
            if L == 0:
                X = np.empty((0, N), dtype=float)
            else:
                X = rng.normal(loc=0.0, scale=1.0, size=(L, N))
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] == N:
                L = int(X.shape[0])
            elif X.shape[0] == N:
                X = X.T
                L = int(X.shape[0])
            else:
                raise ValueError(f"X must have one dimension equal to N={N}, got {X.shape}")

        # ---- hyperparameters ----
        a_tau = self._coerce_vec(self.prior_a_tau, "a_tau")
        b_tau = self._coerce_vec(self.prior_b_tau, "b_tau")
        a_psi = self._coerce_vec(self.prior_a_psi, "a_psi")
        b_psi = self._coerce_vec(self.prior_b_psi, "b_psi")
        sigma_beta = self._pos(self.prior_sigma_regr_coeffs, "sigma_regr_coeffs")

        if self.MGPS:
            a1 = self._pos(self.prior_a1, "a1")
            a2 = self._pos(self.prior_a2, "a2")
            b_lambdavar = self._pos(self.prior_b_lambdavar, "b_lambdavar")
        else:
            a_lambdavar = self._pos(self.prior_a_lambdavar, "a_lambdavar")
            b_lambdavar = self._pos(self.prior_b_lambdavar, "b_lambdavar")

        if self.local_shrinkage_LatentFactors:
            nu = self._pos(self.prior_nu, "nu")

        # ---- 1) regression and factors ----
        regr_coeffs = rng.normal(loc=0.0, scale=sigma_beta, size=(K, L))  # (K, L)
        eta_mean = regr_coeffs @ X                                        # (K, N)
        eta = eta_mean + rng.normal(loc=0.0, scale=1.0, size=(K, N))      # (K, N)

        # ---- 2) global shrinkage ----
        delta = None
        omega = None
        lambda_var = None

        if self.MGPS:
            delta = np.empty(K, dtype=float)
            delta[0] = rng.gamma(shape=a1, scale=1.0)
            if K > 1:
                delta[1:] = rng.gamma(shape=a2, scale=1.0, size=K - 1)

            omega = np.empty(K, dtype=float)
            omega[0] = b_lambdavar
            for k in range(1, K):
                omega[k] = omega[k - 1] * delta[k]
            scale_vec = 1.0 / np.sqrt(omega)
        else:
            g = rng.gamma(shape=a_lambdavar, scale=1.0 / b_lambdavar)
            lambda_var = 1.0 / g
            scale_vec = np.full(K, np.sqrt(lambda_var), dtype=float)

        # ---- 3) per-mode parameters and Y ----
        psi2 = np.empty(D, dtype=float)
        Lambda_list: List[np.ndarray] = []
        theta_list: List[np.ndarray] = []
        tau_draws: List[Union[np.ndarray, float]] = []
        Y_list: List[np.ndarray] = []

        for d in range(D):
            p_d = ps[d]
            M_d = Ms[d]
            is_hetero = self.heteroscedastic_thetas[d]

            # tau_d ~ gamma(a_tau[d], b_tau[d]) (shape-rate)
            if is_hetero:
                tau_d = rng.gamma(shape=a_tau[d], scale=1.0 / b_tau[d], size=p_d)
            else:
                tau_d = float(rng.gamma(shape=a_tau[d], scale=1.0 / b_tau[d]))
            tau_draws.append(tau_d)

            # Lambda_d prior
            if self.local_shrinkage_LatentFactors:
                z = rng.normal(loc=0.0, scale=1.0, size=(p_d, K))
                v = rng.chisquare(df=nu, size=(p_d, K))
                if self.is_mode_decentered(d + 1):
                    decentering_matrix = self.well_built_Lambda_decentering(d+1, p_d)
                    Lambda_d = scale_vec[np.newaxis, :] * (decentering_matrix + z) / np.sqrt(v / nu)
                else:
                    Lambda_d = scale_vec[np.newaxis, :] * z / np.sqrt(v / nu)
            else:
                if self.is_mode_decentered(d + 1):
                    decentering_matrix = self.well_built_Lambda_decentering(d+1, p_d)
                    Lambda_d = decentering_matrix * scale_vec[np.newaxis, :] + rng.normal(
                        loc=0.0,
                        scale=scale_vec[np.newaxis, :],
                        size=(p_d, K),
                    )
                else:
                    Lambda_d = rng.normal(
                        loc=0.0,
                        scale=scale_vec[np.newaxis, :],
                        size=(p_d, K),
                    )
            Lambda_list.append(Lambda_d)

            # theta_d | Lambda_d, eta, tau_d
            mu_theta = Lambda_d @ eta  # (p_d, N)
            if is_hetero:
                sd_rows = np.sqrt(1.0 / np.asarray(tau_d))[:, np.newaxis]
            else:
                sd_rows = np.sqrt(float(tau_d))
            eps_theta = rng.normal(loc=0.0, scale=sd_rows, size=(p_d, N))
            theta_d = mu_theta + eps_theta
            theta_list.append(theta_d)

            # psi2[d] ~ InvGamma(a_psi[d], b_psi[d])
            g = rng.gamma(shape=a_psi[d], scale=1.0 / b_psi[d])
            psi2_d = 1.0 / g
            psi2[d] = psi2_d

            # Y_d = B_d * theta_d + noise
            mean_Y = B_list[d] @ theta_d
            eps_Y = rng.normal(
                loc=0.0,
                scale=np.sqrt(psi2_d),
                size=(M_d, N),
            )
            Y_d = mean_Y + eps_Y

            if missing_prob > 0.0:
                miss = rng.random(size=Y_d.shape) < missing_prob
                Y_d[miss] = np.nan

            Y_list.append(Y_d)

        # ---- 4) wrap into FunctionalData objects ----
        modes: List["FunctionalData"] = []
        for d in range(D):
            fd = FunctionalData(coords_arr_list[d], Y_list[d])
            fd.set_raw_basis(B_list[d])  # basis already built above
            modes.append(fd)

        out: dict = {"X": X, "modes": modes}

        if return_latent:
            out["latent"] = {
                "regr_coeffs": regr_coeffs,
                "eta": eta,
                "delta": delta,
                "omega": omega,
                "lambda_var": lambda_var,
                "tau": tau_draws,
                "theta": theta_list,
                "Lambda": Lambda_list,
                "psi2": psi2,
            }

        if return_stan_data:
            out["stan_data"] = self._build_stan_data(X, modes)

        return out

    def sample_prior_predictive(
        self,
        N: int,
        *,
        n_chains: Optional[int] = None,
        n_draws: int = 100,
        basis_matrices: Optional[List[np.ndarray]] = None,
        basis_functions_list: Optional[List[List[callable]]] = None,
        gp_specs_list: Optional[List[dict]] = None,
        coords_list: Optional[List[np.ndarray]] = None,
        L: Optional[int] = None,
        X: Optional[np.ndarray] = None,
        missing_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> az.InferenceData:
        """
        Draw multiple prior (parameter) and prior-predictive (Y) samples using
        `simulate_dataset`, and return them as an ArviZ InferenceData object.

        Parameters
        ----------
        N : int
            Number of observations (subjects).
        n_chains : int, optional
            Number of chains for the synthetic prior samples. Defaults to
            `self.n_chains`.
        n_draws : int, default 100
            Number of draws per chain.
        basis_matrices, basis_functions_list, gp_specs_list, coords_list :
            Passed through to `simulate_dataset` exactly as in that method.
            Exactly one of the three basis specifications must be non-None.
        L, X : optional
            Predictor dimension or matrix, forwarded to `simulate_dataset`.
            If X is None, each `simulate_dataset` call will generate its own X.
            If X is provided, the same X is used for all draws.
        missing_prob : float, default 0.0
            Missingness probability per Y entry, forwarded to `simulate_dataset`.
        seed : int, optional
            Base RNG seed; per-draw seeds are derived as `seed + offset`.

        Returns
        -------
        idata : az.InferenceData
            InferenceData object with groups:
              - prior:          parameters (regr_coeffs, eta, tau, Lambda, psi2, ...)
              - prior_predictive: synthetic Y_d for each mode (Y1, Y2, ...).
        """
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if not (0.0 <= missing_prob <= 1.0):
            raise ValueError("missing_prob must be in [0, 1]")

        if seed is None:
            seed = self.stan_seed
        base_seed = int(seed)

        n_chains = int(n_chains) if n_chains is not None else int(self.n_chains)
        n_draws = int(n_draws)

        # ---- First call: get shapes and use as (chain=0, draw=0) sample ----
        sim0 = self.simulate_dataset(
            N=N,
            basis_matrices=basis_matrices,
            basis_functions_list=basis_functions_list,
            gp_specs_list=gp_specs_list,
            coords_list=coords_list,
            L=L,
            X=X,
            missing_prob=missing_prob,
            seed=base_seed,
            return_latent=True,
            return_stan_data=False,
        )

        X0 = sim0["X"]
        modes0 = sim0["modes"]
        latent0 = sim0["latent"]

        K = self.n_components
        L_eff = X0.shape[0]
        N_eff = X0.shape[1]
        D = self.D

        # Per-mode dims
        Ms = [m.data.shape[0] for m in modes0]
        ps = [latent0["Lambda"][d].shape[0] for d in range(D)]

        # ---- Allocate arrays for prior group ----
        regr_coeffs_arr = np.empty((n_chains, n_draws, K, L_eff))
        eta_arr = np.empty((n_chains, n_draws, K, N_eff))
        psi2_arr = np.empty((n_chains, n_draws, D))

        delta_arr = None
        omega_arr = None
        lambda_var_arr = None

        if self.MGPS:
            delta_arr = np.empty((n_chains, n_draws, K))
            omega_arr = np.empty((n_chains, n_draws, K))
        else:
            lambda_var_arr = np.empty((n_chains, n_draws))

        # Per-mode parameter arrays
        Lambda_arr_list = []
        theta_arr_list = []
        tau_arr_list = []
        tau_is_scalar = []

        for d in range(D):
            p_d = ps[d]
            M_d = Ms[d]

            Lambda_arr_list.append(np.empty((n_chains, n_draws, p_d, K)))
            theta_arr_list.append(np.empty((n_chains, n_draws, p_d, N_eff)))

            tau0_d = latent0["tau"][d]
            if np.ndim(tau0_d) == 0:  # scalar tau
                tau_is_scalar.append(True)
                tau_arr_list.append(np.empty((n_chains, n_draws)))
            else:
                tau_is_scalar.append(False)
                tau_arr_list.append(np.empty((n_chains, n_draws, p_d)))

        # ---- Allocate arrays for prior_predictive group (Y_d) ----
        Y_arr_list = []
        for d in range(D):
            M_d = Ms[d]
            Y_arr_list.append(np.empty((n_chains, n_draws, M_d, N_eff)))

        # ---- Helper to assign one simulated dataset into arrays ----
        def _assign_sample(c_idx: int, d_idx: int, sim: dict) -> None:
            latent = sim["latent"]
            modes = sim["modes"]

            regr_coeffs_arr[c_idx, d_idx] = latent["regr_coeffs"]
            eta_arr[c_idx, d_idx] = latent["eta"]
            psi2_arr[c_idx, d_idx] = latent["psi2"]

            if self.MGPS:
                delta_arr[c_idx, d_idx] = latent["delta"]
                omega_arr[c_idx, d_idx] = latent["omega"]
            else:
                lambda_var_arr[c_idx, d_idx] = latent["lambda_var"]

            for dd in range(D):
                Lambda_arr_list[dd][c_idx, d_idx] = latent["Lambda"][dd]
                theta_arr_list[dd][c_idx, d_idx] = latent["theta"][dd]

                if tau_is_scalar[dd]:
                    tau_arr_list[dd][c_idx, d_idx] = float(latent["tau"][dd])
                else:
                    tau_arr_list[dd][c_idx, d_idx] = latent["tau"][dd]

                Y_arr_list[dd][c_idx, d_idx] = modes[dd].data

        # Assign first sample
        _assign_sample(0, 0, sim0)

        # ---- Remaining samples ----
        for c in range(n_chains):
            for d in range(n_draws):
                if c == 0 and d == 0:
                    continue
                draw_seed = base_seed + c * n_draws + d
                sim_cd = self.simulate_dataset(
                    N=N,
                    basis_matrices=basis_matrices,
                    basis_functions_list=basis_functions_list,
                    gp_specs_list=gp_specs_list,
                    coords_list=coords_list,
                    L=L,
                    X=X,
                    missing_prob=missing_prob,
                    seed=draw_seed,
                    return_latent=True,
                    return_stan_data=False,
                )
                _assign_sample(c, d, sim_cd)

        # ---- Build ArviZ InferenceData ----
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "K": np.arange(K),
            "L": np.arange(L_eff),
            "N": np.arange(N_eff),
            "D_mode": np.arange(D),
        }
        dims = {
            "regr_coeffs": ["K", "L"],
            "eta": ["K", "N"],
            "psi2": ["D_mode"],
        }

        if self.MGPS:
            dims["delta"] = ["K"]
            dims["omega"] = ["K"]

        for d in range(D):
            coords[f"p{d+1}"] = np.arange(ps[d])
            coords[f"M{d+1}"] = np.arange(Ms[d])

            dims[f"Lambda{d+1}"] = [f"p{d+1}", "K"]
            dims[f"theta{d+1}"] = [f"p{d+1}", "N"]
            dims[f"Y{d+1}"] = [f"M{d+1}", "N"]
            if not tau_is_scalar[d]:
                dims[f"tau{d+1}"] = [f"p{d+1}"]

        prior_dict = {
            "regr_coeffs": regr_coeffs_arr,
            "eta": eta_arr,
            "psi2": psi2_arr,
        }
        if self.MGPS:
            prior_dict["delta"] = delta_arr
            prior_dict["omega"] = omega_arr
        else:
            prior_dict["lambda_var"] = lambda_var_arr

        for d in range(D):
            prior_dict[f"Lambda{d+1}"] = Lambda_arr_list[d]
            prior_dict[f"theta{d+1}"] = theta_arr_list[d]
            prior_dict[f"tau{d+1}"] = tau_arr_list[d]

        prior_pred_dict = {}
        for d in range(D):
            prior_pred_dict[f"Y{d+1}"] = Y_arr_list[d]

        idata = az.from_dict(
            prior=prior_dict,
            prior_predictive=prior_pred_dict,
            coords=coords,
            dims=dims,
        )
        return idata

    def sample_posterior_predictive(
        self,
        *,
        seed: Optional[int] = None,
        replicate_missingness: bool = True,
    ) -> az.InferenceData:
        """
        Generate posterior predictive samples Y_rep for each mode using the
        fitted model (self.idata) and return an InferenceData that includes
        posterior, posterior_predictive, and observed_data groups.

        Parameters
        ----------
        seed : int, optional
            RNG seed for the posterior predictive noise. If None, uses
            self.stan_seed.
        replicate_missingness : bool, default True
            If True, positions that were originally missing (NaN) in the data
            of each FunctionalData mode are set to NaN in the simulated
            Y_rep as well.

        Returns
        -------
        idata : az.InferenceData
            self.idata with added posterior_predictive and observed_data
            groups. Also stored back into self.idata.
        """
        if self.idata is None or self._modes is None:
            raise RuntimeError(
                "Model must be fit (self.idata and self._modes set) "
                "before computing posterior predictive."
            )

        posterior = self.idata.posterior

        # Basic sizes
        n_chains = posterior.sizes.get("chain", posterior.dims["chain"])
        n_draws = posterior.sizes.get("draw", posterior.dims["draw"])
        D = self.D

        # Use psi2 (chain, draw, D)
        psi2_da = posterior["psi2"]
        psi2_np = psi2_da.values  # shape (chain, draw, D)

        # Per-mode dims from fitted FunctionalData
        Ms = [m.data.shape[0] for m in self._modes]
        Ns = [m.data.shape[1] for m in self._modes]
        # All modes share the same N
        N_eff = Ns[0]

        if seed is None:
            seed = self.stan_seed
        rng = np.random.default_rng(int(seed))

        # Containers for posterior predictive and observed data
        Y_rep_dict = {}
        Y_obs_dict = {}

        for d in range(1, D + 1):
            mode_idx = d - 1
            fd = self._modes[mode_idx]
            B_d = np.asarray(fd._basis_matrix, dtype=float)  # (M_d, p_d)
            M_d, p_d = B_d.shape

            # theta_d: (chain, draw, p_d, N)
            if self.absence_of_psi2[d-1]:
                Lambda_da = posterior[f"Lambda{d}"]
                eta_da = posterior["eta"]
                from xarray_einstats import matmul
                theta_da = matmul(Lambda_da, eta_da, dims=[[f"Lambda{d}_dim_0", f"Lambda{d}_dim_1"],["eta_dim_0", "eta_dim_1"]])
                theta_da = theta_da.rename({f"Lambda{d}_dim_0": f"theta{d}_dim_0", "eta_dim_1": f"theta{d}_dim_1"})
            else:
                theta_da = posterior[f"theta{d}"]
            theta_np = theta_da.values

            if theta_np.shape[0] != n_chains or theta_np.shape[1] != n_draws:
                raise RuntimeError(
                    f"Inconsistent theta{d} shape with chain/draw dims."
                )

            # Allocate Y_rep_d: (chain, draw, M_d, N)
            Y_rep_d = np.empty((n_chains, n_draws, M_d, N_eff), dtype=float)

            # Original missingness mask
            if replicate_missingness:
                miss_mask = np.isnan(fd.data)
            else:
                miss_mask = None

            for c in range(n_chains):
                for r in range(n_draws):
                    theta_cr = theta_np[c, r, :, :]   # (p_d, N)
                    mean_Y = B_d @ theta_cr           # (M_d, N)
                    if self.absence_of_psi2[d-1]:
                        Y_sample = mean_Y # no noise
                    else:
                        sd = np.sqrt(psi2_np[c, r, mode_idx])
                        eps = rng.normal(loc=0.0, scale=sd, size=(M_d, N_eff))
                        Y_sample = mean_Y + eps
                    if miss_mask is not None:
                        Y_sample = np.where(miss_mask, np.nan, Y_sample)
                    Y_rep_d[c, r] = Y_sample

            Y_rep_dict[f"Y{d}"] = Y_rep_d
            Y_obs_dict[f"Y{d}"] = fd.data  # (M_d, N)

        # Build coords/dims for ArviZ
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "N": np.arange(N_eff),
        }
        dims = {}

        for d in range(1, D + 1):
            M_d = Ms[d - 1]
            coords[f"M{d}"] = np.arange(M_d)
            dims[f"Y{d}"] = [f"M{d}", "N"]

        idata_ppc = az.from_dict(
            posterior_predictive=Y_rep_dict,
            observed_data=Y_obs_dict,
            coords=coords,
            dims=dims,
        )

        # Merge into a copy of self.idata
        idata_full = self.idata.copy()
        idata_full.posterior_predictive = idata_ppc.posterior_predictive
        idata_full.observed_data = idata_ppc.observed_data

        # Store and return
        self.idata = idata_full
        return self.idata

    def plot_BLambda_d_VarimaxRSP(
        self,
        d: int,
        *,
        ci: float = 0.95,
        factors: Optional[List[int]] = None,
        use_coords: bool = True,
        ax: Optional["plt.Axes"] = None,
    ) -> "plt.Axes":
        """
        Plot posterior percentile intervals of the Varimax-RSP BΛ for mode d
        using seaborn.

        Parameters
        ----------
        d : int
            Mode index (1-based, as in get_BLambda_d).
        ci : float, default 0.95
            Credible interval width (e.g. 0.95 -> 2.5% and 97.5% quantiles).
        factors : list of int, optional
            Indices of factors to plot along the K_rsp dimension.
            If None, all factors are plotted.
        use_coords : bool, default True
            If True, x-axis is the first coordinate of the mode's coords
            array; otherwise, x-axis is just the index 0..M-1.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot into. If None, a new figure and axes
            are created.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before plotting BLambda.")

        # Varimax-RSP rotated BΛ for mode d: dims ("chain", "draw", "M", "K_rsp")
        BLambda_d_rot = self.get_BLambda_d_VarimaxRSP(d)

        # Quantiles over chain & draw
        lower_q = (1.0 - ci) / 2.0
        upper_q = 1.0 - lower_q
        q = BLambda_d_rot.quantile([lower_q, 0.5, upper_q], dim=("chain", "draw"))

        q_lower = q.sel(quantile=lower_q)   # (M, K_rsp)
        q_med   = q.sel(quantile=0.5)
        q_upper = q.sel(quantile=upper_q)

        M_size = BLambda_d_rot.sizes["M"]
        K_size = BLambda_d_rot.sizes["K_rsp"]

        if factors is None:
            factors = list(range(K_size))
        else:
            factors = [f for f in factors if 0 <= f < K_size]
            if not factors:
                raise ValueError("No valid factor indices to plot.")

        # x-axis: either coords or simple index
        coords = self._modes[d - 1].coords  # (M, domainDim)
        if use_coords:
            x_vals = coords[:, 0]
        else:
            x_vals = np.arange(M_size)

        # Build a tidy DataFrame for seaborn
        rows = []
        for k in factors:
            for m in range(M_size):
                rows.append(
                    {
                        "x": float(x_vals[m]),
                        "factor": int(k),
                        "q_lower": float(q_lower.isel(M=m, K_rsp=k)),
                        "q_med": float(q_med.isel(M=m, K_rsp=k)),
                        "q_upper": float(q_upper.isel(M=m, K_rsp=k)),
                    }
                )
        df = pd.DataFrame(rows)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        for k in factors:
            sub = df[df["factor"] == k].sort_values("x")
            # median line
            sns.lineplot(
                data=sub,
                x="x",
                y="q_med",
                ax=ax,
                label=f"factor {k}",
            )
            # credible band
            ax.fill_between(
                sub["x"],
                sub["q_lower"],
                sub["q_upper"],
                alpha=0.2,
            )

        ax.set_xlabel(f"Domain coordinate (mode {d})" if use_coords else "M index")
        ax.set_ylabel("Varimax-RSP BΛ loading")
        ax.set_title(f"Posterior {int(ci*100)}% intervals of Varimaxed BΛ (mode {d})")
        ax.legend(title="Factor")

        return ax

    def get_Sigma_epsilon(self, d: Union[int, List[int]]) -> xr.DataArray:
        """
        Posterior samples of Σε (noise covariance).

        - If `d` is an int (1-based): return a scaled identity over
        (chain, draw, f"M{d}_left", f"M{d}_right").
        - If `d` is a list of ints: return a block-diagonal matrix over
        (chain, draw, "M_left", "M_right"), even if the list has length 1.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before getting Sigma_epsilon.")

        psi2 = self.idata.posterior["psi2"]  # dims: ("chain","draw","psi2_dim_0")

        # --- Case 1: single mode (int) -> use f"M{d}_left/right"
        if isinstance(d, int):
            if not (1 <= d <= self.D):
                raise ValueError(f"Mode index d={d} out of range.")
            p = len(self._modes[d - 1].coords)
            scale = psi2.sel(psi2_dim_0=d - 1)  # ("chain","draw")
            I = xr.DataArray(
                np.eye(p),
                dims=(f"M{d}_left", f"M{d}_right"),
                coords={f"M{d}_left": range(p), f"M{d}_right": range(p)},
            )
            if self.absence_of_psi2 is not None and self.absence_of_psi2[d - 1]:
                scale = 0.0
            return (scale * I).values  # broadcasts to ("chain","draw", f"M{d}_left", f"M{d}_right")

        # --- Case 2: list of modes -> block-diagonal over "M_left"/"M_right"
        if not isinstance(d, list) or len(d) == 0:
            raise ValueError("d must be an int or a non-empty List[int].")

        dims_list = [int(di) for di in d]
        for di in dims_list:
            if not (1 <= di <= self.D):
                raise ValueError(f"Mode index d={di} out of range.")

        sizes = [len(self._modes[di - 1].coords) for di in dims_list]
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        P = sum(sizes)

        out = xr.DataArray(
            np.zeros((psi2.sizes["chain"], psi2.sizes["draw"], P, P)),
            dims=("chain", "draw", "M_left", "M_right"),
            coords={
                "chain": psi2.coords.get("chain", range(psi2.sizes["chain"])),
                "draw":  psi2.coords.get("draw",  range(psi2.sizes["draw"])),
                "M_left": np.arange(P),
                "M_right": np.arange(P),
            },
        )

        for di, start, p in zip(dims_list, starts, sizes):
            if self.absence_of_psi2 is not None and self.absence_of_psi2[di - 1]:
                scale = 0.0
            else:
                scale = psi2.sel(psi2_dim_0=di - 1)  # ("chain","draw")
            I_block = xr.DataArray(
                np.eye(p),
                dims=("M_left", "M_right"),
                coords={
                    "M_left": np.arange(start, start + p),
                    "M_right": np.arange(start, start + p),
                },
            )
            out.loc[
                dict(M_left=slice(start, start + p - 1),
                    M_right=slice(start, start + p - 1))
            ] = (scale * I_block).values

        return out

    def get_Sigma_theta(self, d: Union[int, List[int]]) -> xr.DataArray:
        """
        Posterior samples of Σθ (latent factor covariance).

        - If `d` is an int (1-based): return a scaled identity over
        (chain, draw, f"p{d}_left", f"p{d}_right").
        - If `d` is a list of ints: return a block-diagonal matrix over
        (chain, draw, "p_left", "p_right"), even if the list has length 1.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before getting Sigma_theta.")

        tau = self.idata.posterior  # dims: ("chain","draw","tau_dim_0", ...)

        # --- Case 1: single mode (int) -> use f"p{d}_left/right"
        if isinstance(d, int):
            if not (1 <= d <= self.D):
                raise ValueError(f"Mode index d={d} out of range.")
            heteroschedastic_theta_d = self.heteroscedastic_thetas[d - 1]
            if heteroschedastic_theta_d:
                tau_d = tau[f"tau{d}"].copy()  # dims: ("chain","draw","tau{d}_dim_0")
                p = self.idata.posterior[f"tau{d}"].sizes[f"tau{d}_dim_0"]
                tau_d = tau_d.rename({f"tau{d}_dim_0": f"p{d}_left"})
                variance = 1.0 / tau_d  # dims: ("chain","draw","p{d}_left")
                I = xr.DataArray(
                    np.eye(p),
                    dims=(f"p{d}_left", f"p{d}_right"),
                    coords={f"p{d}_left": range(p), f"p{d}_right": range(p)},
                )
                return (variance * I).values
            else:
                tau_d = tau[f"tau{d}"]  # dims: ("chain","draw")
                # p = self.idata.posterior[f"theta{d}"].sizes[f"theta{d}_dim_0"]
                p = self._modes[d - 1]._basis_matrix.shape[1]
                scale = 1.0 / tau_d  # dims: ("chain","draw")
                I = xr.DataArray(
                    np.eye(p),
                    dims=(f"p{d}_left", f"p{d}_right"),
                    coords={f"p{d}_left": range(p), f"p{d}_right": range(p)},
                )
                return (scale * I).values  # broadcasts to ("chain","draw",f"p{d}_left",f"p{d}_right")

        # --- Case 2: list of modes -> block-diagonal over "p_left"/"p_right"
        if not isinstance(d, list) or len(d) == 0:
            raise ValueError("d must be an int or a non-empty List[int].")

        dims_list = [int(di) for di in d]
        for di in dims_list:
            if not (1 <= di <= self.D):
                raise ValueError(f"Mode index d={di} out of range.")

        sizes = []
        for di in dims_list:
            if self.heteroscedastic_thetas[di - 1]:
                p_di = self.idata.posterior[f"tau{di}"].sizes[f"tau{di}_dim_0"]
            else:
                # p_di = self.idata.posterior[f"theta{di}"].sizes[f"theta{di}_dim_0"]
                p_di = self._modes[di - 1]._basis_matrix.shape[1]
            sizes.append(p_di)
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        P = sum(sizes)

        out = xr.DataArray(
            np.zeros((tau.sizes["chain"], tau.sizes["draw"], P, P)),
            dims=("chain", "draw", "p_left", "p_right"),
            coords={
                "chain": tau.coords.get("chain", range(tau.sizes["chain"])),
                "draw":  tau.coords.get("draw",  range(tau.sizes["draw"])),
                "p_left": np.arange(P),
                "p_right": np.arange(P),
            },
        )

        for di, start, p in zip(dims_list, starts, sizes):
            heteroschedastic_theta_di = self.heteroscedastic_thetas[di - 1]
            if heteroschedastic_theta_di:
                tau_di = tau[f"tau{di}"].copy()  # dims: ("chain","draw","tau{di}_dim_0")
                tau_di = tau_di.rename({f"tau{di}_dim_0": f"p_left"})
                variance = 1.0 / tau_di  # dims: ("chain","draw","p{di}_left")
                I_block = xr.DataArray(
                    np.eye(p),
                    dims=("p_left", "p_right"),
                    coords={
                        "p_left": np.arange(p),
                        "p_right": np.arange(p),
                    },
                )
                out.loc[
                    dict(p_left=slice(start, start + p - 1),
                        p_right=slice(start, start + p - 1))
                ] = (variance * I_block).values
            else:
                tau_di = tau[f"tau{di}"]  # dims: ("chain","draw")
                scale = 1.0 / tau_di  # dims: ("chain","draw")
                I_block = xr.DataArray(
                    np.eye(p),
                    dims=("p_left", "p_right"),
                    coords={
                        "p_left": np.arange(start, start + p),
                        "p_right": np.arange(start, start + p),
                    },
                )
                out.loc[
                    dict(p_left=slice(start, start + p - 1),
                        p_right=slice(start, start + p - 1))
                ] = (scale * I_block).values

        return out
    
    def get_B_matrix_helper_regression_coeffs(
            self,
            modes: List[int],
        ) -> xr.DataArray:
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing B matrix for regression coeffs.")
        for d in modes:
            if d < 1 or d > self.D:
                raise ValueError(f"mode index d={d} out of range.")
        sizes = []
        for d in modes:
            M_d = self._modes[d - 1].data.shape[0]
            sizes.append(M_d)
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        M = sum(sizes)

        sizes = []
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes.append(p_d)
        startsP = np.cumsum([0] + sizes[:-1]).tolist()
        P = sum(sizes)

        B = xr.DataArray(
            np.zeros((M, P)),
            dims=("M_left", "p_left"),
            coords={
                "M_left": np.arange(M),
                "p_left": np.arange(P),
            },
        )

        for d in modes:
            M_d = self._modes[d - 1].data.shape[0]
            B_d = np.asarray(self._modes[d - 1]._basis_matrix, dtype=float)  # (M_d, p_d)
            p_d = B_d.shape[1]
            start_M = starts[modes.index(d)]
            start_p = startsP[modes.index(d)]
            B.loc[
                dict(M_left=slice(start_M, start_M + M_d - 1),
                     p_left=slice(start_p, start_p + p_d - 1))
            ] = B_d

        return B

    def get_E_matrix_helper_regression_coeffs(
        self,
        modes: List[int],
        missing_mask: Optional[List[np.ndarray]] = None,
    ) -> xr.DataArray:
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing E matrix for regression coeffs.")
        for d in modes:
            if d < 1 or d > self.D:
                raise ValueError(f"mode index d={d} out of range.")

        sizes = []
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes.append(p_d)
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        P = sum(sizes)

        L = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], P, self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(P),
                "K": np.arange(self.n_components),
            },
        )
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts[modes.index(d)]
            L.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        C = xe.linalg.matmul(
            L, L.rename({'p_left': 'p_right'}),
            dims=[["p_left", "K"], ["K", "p_right"]]
        )

        Sigma_theta = self.get_Sigma_theta(modes)
        C = C + Sigma_theta

        B = self.get_B_matrix_helper_regression_coeffs(modes)

        E = xe.linalg.matmul(
            B, C,
            dims=[["M_left", "p_left"], ["p_left", "p_right"]]
        )
        E = xe.linalg.matmul(
            E, B.rename({'p_left': 'p_right', 'M_left': 'M_right'}),
            dims=[["M_left", "p_right"], ["p_right", "M_right"]]
        )

        E = E + self.get_Sigma_epsilon(modes)

        if missing_mask is not None:
            # get the indices to be passed to E.sel taking into account the cumulative sizes
            present_indices = []
            start = 0
            for i,d in enumerate(modes):
                mode_idx = d - 1
                mask_d = missing_mask[i]
                if mask_d.shape[0] != self._modes[mode_idx].data.shape[0]:
                    raise ValueError(f"Missing mask for mode {d} has incorrect shape.")
                present_idx_d = np.where(~mask_d)[0]
                present_indices.extend((present_idx_d + start).tolist())
                start += len(mask_d)
            # Select the rows and columns corresponding to present indices
            E = E.sel(M_left=present_indices, M_right=present_indices)

        return E
    
    def get_invE_matrix_helper_regression_coeffs(
        self,
        modes: List[int],
        missing_mask: Optional[List[np.ndarray]] = None,
    ) -> xr.DataArray:
        raise RuntimeError("There is some error in this function, do not use it.")
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing E matrix for regression coeffs.")
        for d in modes:
            if d < 1 or d > self.D:
                raise ValueError(f"mode index d={d} out of range.")

        sizes = []
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes.append(p_d)
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        P = sum(sizes)

        L = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], P, self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(P),
                "K": np.arange(self.n_components),
            },
        )
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts[modes.index(d)]
            L.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        C = xe.linalg.matmul(
            L, L.rename({'p_left': 'p_right'}),
            dims=[["p_left", "K"], ["K", "p_right"]]
        )

        Sigma_theta = self.get_Sigma_theta(modes)
        C = C + Sigma_theta

        B = self.get_B_matrix_helper_regression_coeffs(modes)

        # Let U be the SVD of C
        U, S, _ = xe.linalg.svd(C, full_matrices=True, dims=["p_left", "p_right"])

        U = U.rename({
            'p_left2': 'p_right',
        })
        S = S.rename({
            'p_left': 'p_right',
        })
        U = U * np.sqrt(S)

        # E = Se + BCB^T
        # Assumiamo Se invertibile e diagonale, sia A = sqrt(Se)
        # Assumiamo C = UUt
        # inv(E) = At+ [ I - At BU (I + UtBt Se^{-1} BU)^{-1} (A+ BU)t ] A+ = 
        # = Se^{-1} - BU (I + UtBt Se^{-1} BU)^{-1} (BU)t Se^{-1} =
        # = [I - BU (I + UtBt Se^{-1} BU)^{-1} (BU)t] Se^{-1}
        
        Se = self.get_Sigma_epsilon(modes)
    
        n_diag = Se.sizes["M_left"]

        diag_idx = xr.DataArray(np.arange(n_diag), dims=("M_left",))
        Se_diag = Se.isel(M_left=diag_idx, M_right=diag_idx)

        invSe_diag = 1.0 / Se_diag
        invSe_diag = invSe_diag.drop_vars('M_right')

        retval = B * invSe_diag

        retval = retval.rename({
            'p_left': 'p_right',
        })

        retval = xe.linalg.matmul(
            B, retval,
            dims=[["p_left", "M_left"], ["M_left", "p_right"]]
        )

        retval = xe.linalg.matmul(
            retval, U,
            dims=[["p_left", "p_right"], ["p_left", "p_right"]]
        )

        retval = xe.linalg.matmul(
            U.rename({'p_left': 'p_right', 'p_right': 'p_left'}), retval,
            dims=[["p_left", "p_right"], ["p_left", "p_right"]]
        )
        # retval = UtBt Se^{-1} BU

        # Add identity
        identity = xr.DataArray(
            np.eye(retval.sizes["p_left"]),
            dims=("p_left", "p_right"),
            coords={
                "p_left": np.arange(retval.sizes["p_left"]),
                "p_right": np.arange(retval.sizes["p_right"]),
            }
        )

        retval = retval + identity

        retval = xe.linalg.inv(retval, dims=["p_left", "p_right"])

        retval = xe.linalg.matmul(
            U, retval,
            dims=[["p_left", "p_right"], ["p_left", "p_right"]]
        )
        retval = xe.linalg.matmul(
            retval, U.rename({'p_left': 'p_right', 'p_right': 'p_left'}),
            dims=[["p_left", "p_right"], ["p_left", "p_right"]]
        )

        retval = xe.linalg.matmul(
            B, retval,
            dims=[["M_left", "p_left"], ["p_left", "p_right"]]
        )

        retval = -xe.linalg.matmul(
            retval, B.rename({'p_left': 'p_right', 'M_left': 'M_right'}),
            dims=[["M_left", "p_right"], ["p_right", "M_right"]]
        )

        # now retval = -BU (I + UtBt Se^{-1} BU)^{-1} (BU)t

        # Now add the identity
        identity = xr.DataArray(
            np.eye(retval.sizes["M_left"]),
            dims=("M_left", "M_right"),
            coords={
                "M_left": np.arange(retval.sizes["M_left"]),
                "M_right": np.arange(retval.sizes["M_right"]),
            }
        )

        retval = retval + identity

        invSe_diag = invSe_diag.rename({
            'M_left': 'M_right',
        })

        E = retval.copy()
        E = E * invSe_diag

        if missing_mask is not None:
            # get the indices to be passed to E.sel taking into account the cumulative sizes
            present_indices = []
            for d in modes:
                mode_idx = d - 1
                mask_d = missing_mask[mode_idx]
                if mask_d.shape[0] != self._modes[mode_idx].data.shape[0]:
                    raise ValueError(f"Missing mask for mode {d} has incorrect shape.")
                present_idx_d = np.where(~mask_d)[0]
                start = starts[modes.index(d)]
                present_indices.extend((present_idx_d + start).tolist())
            # Select the rows and columns corresponding to present indices
            E = E.sel(M_left=present_indices, M_right=present_indices)

        return E

    def get_F_matrix_helper_regression_coeffs(
        self,
        source_modes: List[int],
        target_modes: List[int],
        source_missing_masks: Optional[List[np.ndarray]] = None, # if None, no missingness, 1 is missing, 0 is observed
    ) -> xr.DataArray:
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing F matrix for regression coeffs.")
        for d in source_modes:
            if d < 1 or d > self.D:
                raise ValueError(f"source mode index d={d} out of range.")
        for d in target_modes:
            if d < 1 or d > self.D:
                raise ValueError(f"target mode index d={d} out of range.")

        B_source = self.get_B_matrix_helper_regression_coeffs(source_modes)
        B_target = self.get_B_matrix_helper_regression_coeffs(target_modes)

        sizes_source = []
        for d in source_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes_source.append(p_d)
        starts_source = np.cumsum([0] + sizes_source[:-1]).tolist()
        P_source = sum(sizes_source)

        sizes_target = []
        for d in target_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes_target.append(p_d)
        starts_target = np.cumsum([0] + sizes_target[:-1]).tolist()
        P_target = sum(sizes_target)

        L_source = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], P_source, self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(P_source),
                "K": np.arange(self.n_components),
            },
        )
        for d in source_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts_source[source_modes.index(d)]
            L_source.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        L_target = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], P_target, self.n_components)),
            dims=("chain", "draw", "p_right", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_right": np.arange(P_target),
                "K": np.arange(self.n_components),
            },
        )
        for d in target_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts_target[target_modes.index(d)]
            L_target.loc[
                dict(p_right=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        C = xe.linalg.matmul(
            L_source, L_target,
            dims=[["p_left", "K"], ["K", "p_right"]]
        )

        F = xe.linalg.matmul(
            B_source, C,
            dims=[["M_left", "p_left"], ["p_left", "p_right"]]
        )
        F = xe.linalg.matmul(
            F, B_target.rename({'p_left': 'p_right', 'M_left': 'M_right'}),
            dims=[["M_left", "p_right"], ["p_right", "M_right"]]
        )

        if source_missing_masks is not None:
            # get the indices to be passed to F.sel taking into account the cumulative sizes
            present_indices_source = []
            start = 0
            for d in source_modes:
                mode_idx = d - 1
                mask_d = source_missing_masks[source_modes.index(d)]
                if mask_d.shape[0] != self._modes[mode_idx].data.shape[0]:
                    raise ValueError(f"Missing mask for source mode {d} has incorrect shape.")
                present_idx_d = np.where(~mask_d)[0]
                present_indices_source.extend((present_idx_d + start).tolist())
                start += len(mask_d)
            # Select the rows corresponding to present indices in source modes
            F = F.sel(M_left=present_indices_source)

        return F


    def get_transmodal_regression_coeffs(
        self,
        source_modes: List[int] = [1],
        source_missing_masks: Optional[List[np.ndarray]] = None, # if None, no missingness, 1 is missing, 0 is observed
        target_modes: List[int] = [2],
    ) -> xr.DataArray:
        """
        The targets are considered completely observed, while the sources have the missing values to be taken into account.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing transmodal regression coeffs.")

        for d in source_modes:
            if d < 1 or d > self.D:
                raise ValueError(f"source mode index d={d} out of range.")
        for d in target_modes:
            if d < 1 or d > self.D:
                raise ValueError(f"target mode index d={d} out of range.")

        if not source_missing_masks is None:
            if len(source_missing_masks) != len(source_modes):
                raise ValueError("Length of source_missing_masks must match length of source_modes.")
            # also the shapes must match to the observations
            for i, d in enumerate(source_modes):
                fd = self._modes[d - 1]
                if source_missing_masks[i].ndim != 1:
                    raise ValueError("Each source_missing_masks[i] must be 1D array.")
                if source_missing_masks[i].shape[0] != fd.data.shape[0]:
                    raise ValueError("Each source_missing_masks[i] must match the number of observed locations in the corresponding mode.")

        E = self.get_E_matrix_helper_regression_coeffs(source_modes, source_missing_masks)
        F = self.get_F_matrix_helper_regression_coeffs(source_modes, target_modes, source_missing_masks)
        G = self.get_E_matrix_helper_regression_coeffs(target_modes)

        B_source = self.get_B_matrix_helper_regression_coeffs(source_modes)
        B_target = self.get_B_matrix_helper_regression_coeffs(target_modes)
        if source_missing_masks is not None:
            # get the indices to be passed to B_source.sel taking into account the cumulative sizes
            present_indices_source = []
            sizes_source = []
            for d in source_modes:
                fd = self._modes[d - 1]
                sizes_source.append(fd.data.shape[0])
            starts_source = np.cumsum([0] + sizes_source[:-1]).tolist()
            for d in source_modes:
                mode_idx = d - 1
                mask_d = source_missing_masks[source_modes.index(d)]
                present_idx_d = np.where(~mask_d)[0]
                start = starts_source[source_modes.index(d)]
                present_indices_source.extend((present_idx_d + start).tolist())
            B_source = B_source.sel(M_left=present_indices_source)
        
        regr_coeffs_from_sources = xe.linalg.matmul(
            F, xe.linalg.inv(E, dims=['M_left','M_right']), # M_right of F is before, because we are multiplying F.transpose
            dims=[["M_right", "M_left"], ["M_right", "M_left"]] # remember that E is symmetric
        )

        Lsource = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], B_source.sizes["p_left"], self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(B_source.sizes["p_left"]),
                "K": np.arange(self.n_components),
            },
        )
        sizes_source = []
        for d in source_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes_source.append(p_d)
        starts_source = np.cumsum([0] + sizes_source[:-1]).tolist()
        for d in source_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts_source[source_modes.index(d)]
            Lsource.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        BLsource = xe.linalg.matmul(
            B_source, Lsource,
            dims=[["M_left", "p_left"], ["p_left", "K"]]
        )  # dims: ("chain", "draw", "M_left", "K")

        Ltarget = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], B_target.sizes["p_left"], self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(B_target.sizes["p_left"]),
                "K": np.arange(self.n_components),
            },
        )
        sizes_target = []
        for d in target_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            sizes_target.append(p_d)
        starts_target = np.cumsum([0] + sizes_target[:-1]).tolist()
        for d in target_modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = starts_target[target_modes.index(d)]
            Ltarget.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        BLtarget = xe.linalg.matmul(
            B_target, Ltarget,
            dims=[["M_left", "p_left"], ["p_left", "K"]]
        )  # dims: ("chain", "draw", "M_left", "K")
        BLtarget = BLtarget.rename({'M_left': 'M_right'})

        regr_coeffs_from_covariates = BLtarget - xe.linalg.matmul(
            regr_coeffs_from_sources, BLsource,
            dims=[["M_right", "M_left"], ["M_left", "K"]]
        )

        if 'regr_coeffs' not in self.idata.posterior:
            regr_coeffs_from_covariates = None # no covariates were used
        else:
            regr_coeffs_from_covariates = xe.linalg.matmul(
                regr_coeffs_from_covariates,
                self.idata.posterior['regr_coeffs'].rename({'regr_coeffs_dim_0': 'K', 'regr_coeffs_dim_1': 'R'}),
                dims=[["M_right", "K"], ["K", "R"]]
            )

        Cov = xe.linalg.matmul(
            regr_coeffs_from_sources, F,
            dims=[["M_right", "M_left"], ["M_left", "M_right"]]
        ).rename({'M_right2': 'M_left'})

        Cov = G - Cov

        return regr_coeffs_from_sources, regr_coeffs_from_covariates, Cov

    def get_projection_matrix(
            self,
            modes: List[int],
            missing_mask: Optional[List[np.ndarray]] = None,
    ):
        """
        Get the posterior mean projection matrix from the specified modes
        to the data space formed by those modes.

        Parameters
        ----------
        modes : list of int
            List of mode indices (1-based) to include in the projection.

        Returns
        -------
        P : xr.DataArray
            Projection matrix with dims ("M_left", "M_right").
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before computing projection matrix.")

        for d in modes:
            if d < 1 or d > self.D:
                raise ValueError(f"mode index d={d} out of range.")

        B = self.get_B_matrix_helper_regression_coeffs(modes)  # (M, p)

        if missing_mask is not None:
            # get the indices to be passed to B.sel taking into account the cumulative sizes
            present_indices = []
            for d in modes:
                mode_idx = d - 1
                mask_d = missing_mask[mode_idx]
                if mask_d.shape[0] != self._modes[mode_idx].data.shape[0]:
                    raise ValueError(f"Missing mask for mode {d} has incorrect shape.")
                present_idx_d = np.where(~mask_d)[0]
                sizes = []
                for di in modes:
                    M_di = self._modes[di - 1].data.shape[0]
                    sizes.append(M_di)
                starts = np.cumsum([0] + sizes[:-1]).tolist()
                start = starts[modes.index(d)]
                present_indices.extend((present_idx_d + start).tolist())
            # Select the rows corresponding to present indices
            B = B.sel(M_left=present_indices)

        Lambda = xr.DataArray(
            np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], B.sizes["p_left"], self.n_components)),
            dims=("chain", "draw", "p_left", "K"),
            coords={
                "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                "p_left": np.arange(B.sizes["p_left"]),
                "K": np.arange(self.n_components),
            },
        )
        for d in modes:
            p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
            Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
            Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
            start = sum(
                self.idata.posterior[f"Lambda{di}"].sizes[f"Lambda{di}_dim_0"]
                for di in modes if di < d
            )
            Lambda.loc[
                dict(p_left=slice(start, start + p_d - 1))
            ] = Lambda_d.values

        G = xe.linalg.matmul(
            B, Lambda,
            dims=[["M_left", "p_left"], ["p_left", "K"]]
        )  # dims: ("chain", "draw", "M_left", "K")

        E = self.get_E_matrix_helper_regression_coeffs(modes, missing_mask)

        regr_coeffs_modes = xe.linalg.matmul(
            G, xe.linalg.inv(E, dims=['M_left','M_right']),
            dims=[["K", "M_left"], ["M_right", "M_left"]]
        )  # dims: ("chain", "draw", "K", "M_left")

        Cov = xe.linalg.matmul(
            regr_coeffs_modes, G.rename({'K': 'K_right'}),
            dims=[["K", "M_left"], ["M_left", "K_right"]]
        )  # dims: ("chain", "draw", "K", "K_right")
        Cov = Cov.rename({'K': 'K_left'})

        I = xr.DataArray(
            np.eye(self.n_components),
            dims=("K_left", "K_right"),
            coords={"K_left": np.arange(self.n_components), "K_right": np.arange(self.n_components)},
        )

        Cov = I - Cov

        if 'regr_coeffs' not in self.idata.posterior:
            regr_coeffs_covariates = None # no covariates were used
        else:
            regr_coeffs_covariates = xe.linalg.matmul(
                Cov,
                self.idata.posterior['regr_coeffs'].rename({'regr_coeffs_dim_0': 'K_right', 'regr_coeffs_dim_1': 'R'}),
                dims=[["K_left", "K_right"], ["K_right", "R"]]
            )  # dims: ("chain", "draw", "K_left", "R")

        regr_coeffs_covariates = regr_coeffs_covariates.rename({'K_left': 'K'})

        return regr_coeffs_modes, regr_coeffs_covariates, Cov



    def save(self, path: str) -> None:
        """
        Save the full state of this MvtBLF instance.

        Files written (using `path` as a prefix, extension stripped if present):
        - <path>.pkl          : small Python state (init params, modes, etc.)
        - <path>_idata.nc     : InferenceData via az.to_netcdf (if present)
        - <path>_varimax.nc   : VarimaxRSP cache as xarray.Dataset (if present)

        The CmdStanModel executable is NOT serialized; it will be
        reconstructed on load via the constructor using saved init params.
        """
        # Normalize base prefix (strip any extension)
        base_root = os.path.splitext(os.path.abspath(path))[0]
        state_path = base_root + ".pkl"
        idata_path = base_root + "_idata.nc"
        varimax_path = base_root + "_varimax.nc"

        # 1) init parameters needed to reconstruct the object (mirror __init__)
        init_params = {
            "n_components": self.n_components,
            "D": self.D,
            "heteroscedastic_thetas": self.heteroscedastic_thetas,
            "MGPS": self.MGPS,
            "local_shrinkage_LatentFactors": self.local_shrinkage_LatentFactors,
            "n_chains": self.n_chains,
            "iter_warmup": self.iter_warmup,
            "iter_sampling": self.iter_sampling,
            "max_treedepth": self.max_treedepth,
            "stan_seed": self.stan_seed,
            "stanc_options": self.stanc_options,
            "cpp_options": self.cpp_options,
            "stanfilename": self._stanfilename,
            "a_psi": self.prior_a_psi,
            "b_psi": self.prior_b_psi,
            "a_tau": self.prior_a_tau,
            "b_tau": self.prior_b_tau,
            "a1": self.prior_a1,
            "a2": self.prior_a2,
            "a_lambdavar": self.prior_a_lambdavar,
            "b_lambdavar": self.prior_b_lambdavar,
            "nu": self.prior_nu,
            "sigma_regr_coeffs": self.prior_sigma_regr_coeffs,
        }

        # 2) serialize FunctionalData modes in a minimal form
        modes_payload = None
        if self._modes is not None:
            modes_payload = []
            for fd in self._modes:
                modes_payload.append(
                    {
                        "coords": np.asarray(fd.coords),
                        "data": np.asarray(fd.data),
                        "basis_matrix": (
                            np.asarray(getattr(fd, "_basis_matrix", None))
                            if hasattr(fd, "_basis_matrix")
                            and getattr(fd, "_basis_matrix") is not None
                            else None
                        ),
                    }
                )

        # 3) save InferenceData to NetCDF (if present)
        idata_file = None
        if self.idata is not None:
            az.to_netcdf(self.idata, idata_path)
            idata_file = idata_path

        # 4) cached VarimaxRSP result (cached_property) if present
        varimax_file = None
        varimax_cache = self.__dict__.get("computeVarimaxRSP", None)
        if varimax_cache is not None:
            lambdas_rsp_xr, Rotations_xr, Q_xr, ref_xr = varimax_cache
            varimax_ds = xr.Dataset(
                {
                    "lambdas_rsp": lambdas_rsp_xr,
                    "Rotations": Rotations_xr,
                    "Q": Q_xr,
                    "ref": ref_xr,
                }
            )
            varimax_ds.to_netcdf(varimax_path)
            varimax_file = varimax_path

        # 5) small state pickle (paths, dims, etc.)
        state = {
            "init_params": init_params,
            "_dims": getattr(self, "_dims", None),
            "modes_payload": modes_payload,
            "idata_file": idata_file,
            "varimax_file": varimax_file,
        }

        with open(state_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "MvtBLF":
        """
        Load a MvtBLF instance previously saved with `.save`.

        Uses:
        - pickle for small Python state (<path>.pkl)
        - az.from_netcdf for InferenceData (<path>_idata.nc)
        - xarray.load_dataset for VarimaxRSP cache (<path>_varimax.nc)
        """
        base_root = os.path.splitext(os.path.abspath(path))[0]
        state_path = base_root + ".pkl"

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        init_params = state["init_params"]
        obj = cls(**init_params)

        # restore simple attributes
        obj._dims = state.get("_dims", None)

        # restore modes
        modes_payload = state.get("modes_payload", None)
        if modes_payload is not None:
            modes = []
            for mp in modes_payload:
                fd = FunctionalData(mp["coords"], mp["data"])
                if mp.get("basis_matrix") is not None:
                    fd.set_raw_basis(mp["basis_matrix"])
                modes.append(fd)
            obj._modes = modes
        else:
            obj._modes = None

        # restore InferenceData via NetCDF
        idata_file = state.get("idata_file", None)
        if idata_file is not None:
            obj.idata = az.from_netcdf(idata_file)
        else:
            obj.idata = None

        # restore VarimaxRSP cache (if present)
        varimax_file = state.get("varimax_file", None)
        if varimax_file is not None and os.path.exists(varimax_file):
            varimax_ds = xr.load_dataset(varimax_file)
            lambdas_rsp_xr = varimax_ds["lambdas_rsp"]
            Rotations_xr = varimax_ds["Rotations"]
            Q_xr = varimax_ds["Q"]
            ref_xr = varimax_ds["ref"]
            # cached_property stores its value directly in __dict__
            obj.__dict__["computeVarimaxRSP"] = (
                lambdas_rsp_xr,
                Rotations_xr,
                Q_xr,
                ref_xr,
            )

        return obj


    def _predict_from_covariates(
            self,
            basis_matrix: xr.DataArray,
            LatentComponents: xr.DataArray,
            regr_coeffs: xr.DataArray,
            covariates: np.ndarray,
    ) -> xr.DataArray:
        """
        Helper function to compute the predicted data from covariates.

        Parameters
        ----------
        basis_matrix : xr.DataArray
            Basis matrix for the target mode, shape (M, p).
        LatentComponents : xr.DataArray
            Latent factors, shape (p, K).
        regr_coeffs : xr.DataArray
            Regression coefficients, shape (K, R).
        covariates : np.ndarray
            Covariates for regression coefficients, shape (R,).

        Returns
        -------
        predicted_data : xr.DataArray
            Predicted data array for the target mode, shape (M).
        """
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim != 1 or covariates.shape[0] != regr_coeffs.sizes["R"]:
            raise ValueError(
                f"covariates must be 1D of length R={regr_coeffs.sizes['R']}"
            )

        cov_da = xr.DataArray(
            covariates,
            dims=("R",),
            coords={"R": regr_coeffs.coords.get("R", np.arange(covariates.shape[0]))},
        )

        # (K x R) · covariates (R,) -> (K,)
        retval = regr_coeffs.dot(cov_da, dims=["R"])  # dims: ("K",)
        retval = LatentComponents.dot(retval, dims=["K"])  # dims: ("p",)
        retval = basis_matrix.dot(retval, dims=["p"])  # dims: ("M",)

        return retval


    def predict_from_covariates(
            self,
            target_mode: int,
            covariates: np.ndarray,
            num_samples: int = 1000,
            add_observation_noise: bool = True,
    ) -> np.ndarray:
        pass


    def prediction(
            self,
            target_mode: int,
            source_modes: List[int],
            source_data: List[np.ndarray],
            covariates: Optional[np.ndarray] = None,
            num_samples: int = 1000,
            add_observation_noise: bool = True,
    ) -> np.ndarray:
        """
        Predict the data in the target mode given observations in the source modes.

        Parameters
        ----------
        target_mode : int
            The mode index (1-based) to predict.
        source_modes : list of int
            List of mode indices (1-based) used as sources for prediction.
        source_data : list of np.ndarray
            List of observed data arrays corresponding to source_modes.
        covariates : np.ndarray, optional
            Array of covariates for regression coefficients, shape (R,).
            If not passed, either no covariates were used in the model or
            predictions will be made without covariate effects.
        num_samples : int
            Number of predicitons to draw from the posterior predictive distribution.
        add_observation_noise : bool
            Whether to add the noise term or to only use the regression part.

        Returns
        -------
        predicted_data : np.ndarray
            Predicted data array for the target mode.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before making predictions.")
        
        if target_mode < 1 or target_mode > self.D:
            raise ValueError(f"target_mode index {target_mode} out of range.")
        
        if len(source_modes) != len(source_data):
            raise ValueError("Length of source_modes must match length of source_data.")
        
        for i, mode in enumerate(source_modes):
            if mode < 1 or mode > self.D:
                raise ValueError(f"source_mode index {mode} out of range.")
            if source_data[i].ndim != 1:
                raise ValueError("Each source_data[i] must be a 1D array.")
            if source_data[i].shape[0] != self._modes[mode - 1].data.shape[0]:
                raise ValueError("Each source_data[i] must match the number of observed locations in the corresponding mode.")

        if covariates is not None:
            if covariates.ndim != 1:
                raise ValueError("Covariates must be a 1D array.")
            if covariates.shape[0] != self._covariates.shape[0]:
                raise ValueError("Covariates length must match the number of regression coefficients R.")

        # for each element in source data get the missing data mask
        source_missing_masks = []
        for i, mode in enumerate(source_modes):
            mask = np.isnan(source_data[i])
            source_missing_masks.append(mask)


        raise NotImplementedError("Prediction method is not yet implemented.")


    def predict_from_score(
            self,
            target_mode: int,
            scores: np.ndarray,
            num_samples: int = 1000,
            observational_noise: bool = True,
    ) -> np.ndarray:
        """
        Predict the data in the target mode given the latent factor scores.

        Parameters
        ----------
        target_mode : int
            The mode index (1-based) to predict.
        scores : np.ndarray
            Array of latent factor scores, shape (n_samples, n_components).
        num_samples : int
            Number of predicitons to draw from the posterior predictive distribution.
        observational_noise : bool
            Whether to add the noise term or to only use the regression part.

        Returns
        -------
        predicted_data : np.ndarray
            Predicted data array for the target mode.
        """
        if self.idata is None:
            raise RuntimeError("Model must be fit before making predictions.")
        
        if target_mode < 1 or target_mode > self.D:
            raise ValueError(f"target_mode index {target_mode} out of range.")
        
        if scores.ndim != 1:
            raise ValueError("Scores must be a 1D array.")
        if scores.shape[0] != self.n_components:
            raise ValueError("Scores length must match the number of components.")

        raise NotImplementedError("predict_from_score method is not yet implemented.")



    def predict_dumb_version(
            self,
            target_mode: int,
            source_modes: List[int],
            source_data: List[np.ndarray],
            covariates: Optional[np.ndarray] = None,
            add_observation_noise: bool = True,
    ):
        G = self.get_E_matrix_helper_regression_coeffs([target_mode])

        # from source_data get the source_modes missing masks
        source_missing_masks = []
        for i, mode in enumerate(source_modes):
            mask = np.isnan(source_data[i])
            source_missing_masks.append(mask)

        E = self.get_E_matrix_helper_regression_coeffs(source_modes, source_missing_masks).rename({'M_right': 'M_source'})
        F = self.get_F_matrix_helper_regression_coeffs(source_modes, [target_mode], source_missing_masks).rename({'M_left': 'M_source', 'M_right': 'M_target'})

        FEinv = xe.linalg.solve(
            E,
            F,
            dims=['M_left','M_source','M_target']
        ).rename({'M_left': 'M_source'}) # (chain, draw, M_source, M_target)

        Cov = G - xe.linalg.matmul(
            FEinv,
            F.rename({'M_target': 'M_right'}),
            dims=[['M_target','M_source'], ['M_source','M_right']]
        ).rename({'M_target': 'M_left'}) # (chain, draw, M_left, M_right)

        if covariates is None:
            # just predict zeros as Estimate_Target_from_Covariates and Estimate_Sources_from_Covariates
            Estimate_Target_from_Covariates = xr.DataArray(
                np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], self._modes[target_mode-1].data.shape[0])),
                dims=("chain", "draw", "M"),
                coords={
                    "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                    "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                    "M": np.arange(self._modes[target_mode-1].data.shape[0]),
                },
            )
            Estimate_Sources_from_Covariates = xr.DataArray(
                np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], sum(self._modes[d-1].data.shape[0] for d in source_modes))),
                dims=("chain", "draw", "M"),
                coords={
                    "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                    "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                    "M": np.arange(sum(self._modes[d-1].data.shape[0] for d in source_modes)),
                },
            )
        else:
            Estimate_Target_from_Covariates = self._predict_from_covariates(
                basis_matrix=xr.DataArray(self._modes[target_mode-1]._basis_matrix, dims=["M", "p"], coords={'M': np.arange(self._modes[target_mode-1]._basis_matrix.shape[0]), 'p': np.arange(self._modes[target_mode-1]._basis_matrix.shape[1])}),
                LatentComponents=self.idata.posterior[f"Lambda{target_mode}"].rename({f"Lambda{target_mode}_dim_0": "p", f"Lambda{target_mode}_dim_1": "K"}),
                regr_coeffs=self.idata.posterior['regr_coeffs'].rename({'regr_coeffs_dim_0': 'K', 'regr_coeffs_dim_1': 'R'}),
                covariates=covariates,
            )

            sizes = []
            for d in source_modes:
                p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
                sizes.append(p_d)
            starts = np.cumsum([0] + sizes[:-1]).tolist()
            P = sum(sizes)

            L = xr.DataArray(
                np.zeros((self.idata.posterior.sizes["chain"], self.idata.posterior.sizes["draw"], P, self.n_components)),
                dims=("chain", "draw", "p_left", "K"),
                coords={
                    "chain": self.idata.posterior.coords.get("chain", range(self.idata.posterior.sizes["chain"])),
                    "draw":  self.idata.posterior.coords.get("draw",  range(self.idata.posterior.sizes["draw"])),
                    "p_left": np.arange(P),
                    "K": np.arange(self.n_components),
                },
            )
            for d in source_modes:
                p_d = self.idata.posterior[f"Lambda{d}"].sizes[f"Lambda{d}_dim_0"]
                Lambda_d = self.idata.posterior[f"Lambda{d}"].copy()
                Lambda_d = Lambda_d.rename({f"Lambda{d}_dim_0": "p_left", f"Lambda{d}_dim_1": "K"})
                start = starts[source_modes.index(d)]
                L.loc[
                    dict(p_left=slice(start, start + p_d - 1))
                ] = Lambda_d.values

            Estimate_Sources_from_Covariates = self._predict_from_covariates(
                basis_matrix=self.get_B_matrix_helper_regression_coeffs(source_modes).rename({'M_left': 'M', 'p_left': 'p'}),
                LatentComponents=L.rename({'p_left': 'p', 'K': 'K'}),
                regr_coeffs=self.idata.posterior['regr_coeffs'].rename({'regr_coeffs_dim_0': 'K', 'regr_coeffs_dim_1': 'R'}),
                covariates=covariates,
            )

        # use missing data to drop rows in Estimate_Sources_from_Covariates
        sizes_source = []
        for d in source_modes:
            M_d = self._modes[d - 1].data.shape[0]
            sizes_source.append(M_d)
        starts_source = np.cumsum([0] + sizes_source[:-1]).tolist()
        present_indices_source = []
        for d in source_modes:
            mode_idx = d - 1
            mask_d = source_missing_masks[source_modes.index(d)]
            present_idx_d = np.where(~mask_d)[0]
            start = starts_source[source_modes.index(d)]
            present_indices_source.extend((present_idx_d + start).tolist())
        Estimate_Sources_from_Covariates = Estimate_Sources_from_Covariates.sel(M=present_indices_source)

        Residual_Sources = xr.DataArray(
            np.zeros(Estimate_Sources_from_Covariates.shape),
            dims=Estimate_Sources_from_Covariates.dims,
            coords=Estimate_Sources_from_Covariates.coords,
        )
        # fill Residual_Sources with the observed - estimated values
        idx = 0
        for i, d in enumerate(source_modes):
            M_d = self._modes[d - 1].data.shape[0]
            mask_d = source_missing_masks[i]
            present_idx_d = np.where(~mask_d)[0]
            for j, mi in enumerate(present_idx_d):
                Residual_Sources.loc[dict(M=idx + mi)] = source_data[i][mi] - Estimate_Sources_from_Covariates.loc[dict(M=idx + mi)]
            idx += len(mask_d)

        Estimate_Target_from_Sources = FEinv.dot(Residual_Sources.rename({'M': 'M_source'}), dims=['M_source'])  # dims: ("chain", "draw", "M_target")

        Predicted_Target = Estimate_Target_from_Covariates + Estimate_Target_from_Sources.rename({'M_target': 'M'})
        Predicted_Target = Predicted_Target.transpose("chain", "draw", "M")

        if add_observation_noise:
            # in this case we already have the covariance Cov computed above
            # sample from multivariate normal distribution with covariance Cov
            n_chains = self.idata.posterior.sizes["chain"]
            n_draws = self.idata.posterior.sizes["draw"]
            M = Predicted_Target.sizes["M"]
            noise_samples = np.zeros((n_chains, n_draws, M))
            for c in range(n_chains):
                for d in range(n_draws):
                    cov_cd = Cov.isel(chain=c, draw=d).values
                    noise_cd = np.random.multivariate_normal(
                        mean=np.zeros(M),
                        cov=cov_cd
                    )
                    noise_samples[c, d, :] = noise_cd
            noise_da = xr.DataArray(
                noise_samples,
                dims=Predicted_Target.dims,
                coords=Predicted_Target.coords,
            )
            return Predicted_Target + noise_da, Predicted_Target, Cov
        else:
            return Predicted_Target, Predicted_Target, Cov

