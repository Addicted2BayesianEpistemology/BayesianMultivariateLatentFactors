# MultivariateBLF — Bayesian Multimodal Latent Factor model for functional data

This repository contains the Python package **`MultivariateBLF`** and the
experimental code used in the paper

> Ursino B., Epifani I., Trovò F.
> *A Bayesian latent factors approach for multimodal functional data.*
> Submitted to *Journal of the Royal Statistical Society, Series B*.

The package implements the BMLF model proposed in the paper: a joint
probabilistic formulation for multiple functional modalities, with shared
latent factors, covariate-driven latent scores, modality-specific basis
expansions, multiplicative-gamma shrinkage on the loadings, and closed-form
cross-modal prediction. Posterior inference uses Stan (HMC/NUTS) via
`cmdstanpy`.

## Repository layout

```
.
├── MultivariateBLF/                 the Python package (importable as `MultivariateBLF`)
│   ├── pyproject.toml
│   └── src/MultivariateBLF/
│       ├── MultivariateBLF.py       FunctionalData, VectorialData, MvtBLF
│       ├── PartialLeastSquares.py   PLS / fPLS baselines + nonparametric bootstrap
│       ├── stan_helper.py           on-the-fly Stan program generation + caching
│       ├── visualization.py         alignment of true loadings to Varimax-RSP
│       ├── gp_kernels.py            squared-exponential and Matérn 5/2 kernels
│       ├── helpers.py               Varimax rotation, OALS-PCA for vectorial data
│       └── stan/                    cached compiled Stan programs (auto-generated)
│
├── figure2_confronto_processigaussiani_con_tau_riscalato/
│   └── code2.py                     Figure 2: effect of heteroscedastic basis scaling
│
├── coverage_estimation_multiple/
│   ├── coverage_estimation_multiple.py    Figures 5, 6, 7 raw data
│   └── boxplots.py                        the actual figures from the CSVs
│
└── prediction ability/
    ├── run_prediction_ability.py    Figure 4 raw data (cross-modal prediction MSE)
    └── plot.py                      the figure from the CSVs
```

Each experiment folder writes CSV files that are read back by a separate
plotting script — heavy computations were run on a server, plots produced
locally. See [Reproducing the figures](#reproducing-the-figures) below.

---

## Installation

The package targets Python ≥ 3.8 and relies on
[`cmdstanpy`](https://mc-stan.org/cmdstanpy/) for posterior inference. From
the repository root:

```bash
pip install -e ./MultivariateBLF
# cmdstanpy will compile programs the first time you fit a model:
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Dependencies pulled in transitively include `numpy`, `pandas`, `scipy`,
`scikit-learn`, `arviz`, `xarray`, `xarray-einstats`, `cmdstanpy`,
`matplotlib`, `seaborn`, and `tqdm`.

---

## The `MultivariateBLF` package

### Model overview

For each subject `n = 1, …, N` and modality `d = 1, …, D` the observed
functional target `Y^(d)_n ∈ R^{M_d}` admits the basis expansion

```
Y^(d)_n   =  B^(d) · ( Λ^(d) · η_n  +  u^(d)_n )  +  ε^(d)_n
η_n       =  β X_n  +  ζ_n,           ζ_n ~ N(0, I_K)
ε^(d)_n   ~  N(0, ψ_d^{-1} I_{M_d})
u^(d)_{j,n} ~ N(0, τ^(d)_j^{-1})
Λ^(d)_{j,k} ~ N(0, [φ^(d)_{j,k} · ω_k]^{-1})
```

where `B^(d) ∈ R^{M_d × p_d}` is a modality-specific basis matrix, `Λ^(d)`
the loadings, `η_n ∈ R^K` the shared latent scores, `β ∈ R^{K × L}` the
covariate regression, and `ω_k = ∏_{ℓ ≤ k} δ_ℓ` the cumulative
multiplicative-gamma shrinkage of Bhattacharya & Dunson (2011), with
`δ_1 ~ Gamma(a_1, 1)` and `δ_ℓ ~ Gamma(a_2, 1)` for `ℓ ≥ 2`. Heteroscedastic
basis precisions `τ^(d)_j` and local shrinkage `φ^(d)_{j,k}` can be enabled
independently per modality.

The basis matrix `B^(d)` is a model choice rather than a feature of the
data: typical options are Fourier, B-splines, or a kernel-induced
eigenbasis (Section 3.5 of the paper). The package supports all three
through the `FunctionalData` interface (next section).

### `FunctionalData` — observed targets and their basis

```python
import numpy as np
import MultivariateBLF.MultivariateBLF as mvblf

coords = np.linspace(0.0, 1.0, 80)
Y = ...  # shape (M=80, N=50), NaN where missing

fd = mvblf.FunctionalData(coords=coords, data=Y)
```

The constructor accepts either:

- an `(M,)` or `(M, domainDim)` coordinate array and an `(M, N)` data
  matrix (NaNs allowed; ±Inf is converted to NaN with a warning), or
- two parallel *lists* of one-array-per-subject, in which case unique
  coordinates are merged lexicographically and missing observations are
  filled with NaN.

A basis is then attached to the object via one of:

```python
# 1. supply the basis matrix directly
fd.set_raw_basis(B)                       # B : (M, p)

# 2. supply a list of basis functions f_1, …, f_p   (they are evaluated on `coords`)
fd.set_functional_basis([f1, f2, ..., fp])

# 3. kernel-induced eigenbasis (Section 3.5):
fd.set_GP_basis(
    n_basis=20,
    gaussian_process="matern5/2",        # or "quad_cov_kernel"
    gp_lengthscale=0.15,
    gp_variance=4.0,
)
```

Useful properties:

- `fd.coords`, `fd.data`, `fd.missingnessMask`, `fd.domainDim`
- `fd.empirical_basis_coefficients` — least-squares projection `T = B⁺ Y`
- `fd.stan_friendly_data` — the sufficient-statistic dict consumed by Stan
  (uses Theorem 2 of the paper to handle missingness efficiently)
- `fd.compute_semivariogram(...)` / `fd.plot_semivariogram(...)` —
  exploratory tools for choosing the kernel lengthscale.

`VectorialData(data)` is a subclass for ordinary `(p × N)` vectorial data
treated as functional over the standard basis in `R^p`. It defaults to an
identity basis and offers `set_PCA_basis(...)` (an OALS PCA that supports
missing values).

### `MvtBLF` — the Bayesian model

```python
from MultivariateBLF.MultivariateBLF import MvtBLF

model = MvtBLF(
    n_components=3,                       # K, number of latent factors
    D=2,                                  # number of functional modalities
    heteroscedastic_thetas=True,          # per-mode bool or list of bools
    MGPS=True,                            # multiplicative-gamma loadings shrinkage
    local_shrinkage_LatentFactors=True,   # Student-t local shrinkage on Λ
    absence_of_psi2=[False, False],       # set True for a noise-free modality

    # priors (scalars or length-D lists where applicable)
    a_psi=2.0,  b_psi=2.0,                # noise precisions ψ_d
    a_tau=2.0,  b_tau=0.25,               # basis-coeff precisions τ^(d)_j
    a1=1.3,     a2=2.0,                   # MGPS shape parameters
    nu=3.0,                               # Student-t df for Lambda local shrinkage
    sigma_regr_coeffs=10.0,               # prior scale of β

    # NUTS settings
    n_chains=4, iter_warmup=1000, iter_sampling=1000,
    max_treedepth=12, stan_seed=42,
)
```

Construction is *cheap*: the Stan program is generated on the fly by
`stan_helper._assembleMBLF_stan_code_for_given_D(...)` and compiled by
`cmdstanpy` only once per program signature; the compiled executable is
cached on disk under `MultivariateBLF/stan/`. Subsequent constructions
with the same `D`, heteroscedasticity pattern, MGPS flag, and Lambda
shrinkage flag reuse the cached binary.

The model is fitted by passing the predictor matrix `X` together with the
per-modality `FunctionalData` objects:

```python
# X : (L, N) or (N, L) — covariates per subject (L = 0 if there are none)
model.fit(X=X, modes=[fd1, fd2])
```

#### Inspecting the posterior

| Method | Returns |
|---|---|
| `model.fit_result` | underlying `CmdStanMCMC` |
| `model.idata` | `arviz.InferenceData` (posterior) |
| `model.get_Lambda_d(d)` | loadings Λ^(d), dims `(chain, draw, p, K)` |
| `model.get_BLambda_d(d)` | `B^(d) Λ^(d)` in the domain, dims `(chain, draw, M, K)` |
| `model.get_Sigma_epsilon(d)` | noise covariance for mode `d` (or block-diag over a list of modes) |
| `model.get_Sigma_theta(d)` | basis-coefficient residual covariance |

#### Identifiability and Varimax-RSP

Factor models are identifiable only up to orthogonal rotation and signed
permutation. The package implements the Varimax-RSP post-processor of
Papastamoulis & Ntzoufras (2022) as a cached property:

```python
lambdas_rsp, rotations, perms, ref = model.computeVarimaxRSP
post_BL1 = model.get_BLambda_d_VarimaxRSP(1)  # dims (chain, draw, M, K_rsp)
model.plot_BLambda_d_VarimaxRSP(1, ci=0.80)
```

When you know the ground truth (simulation studies), align it to the same
orientation with

```python
from MultivariateBLF.visualization import align_true_factors_to_varimaxrsp

out = align_true_factors_to_varimaxrsp(
    model=model,
    Lambda_true_list=[Lambda1_true, Lambda2_true],
    B_true_list=[B1_true, B2_true],     # optional, defaults to model bases
)
out["BLambda_true_aligned_per_mode"]    # list of (M_d, K) ready to plot
```

#### Cross-modal prediction (Theorem 4 of the paper)

For a partially observed replicate `(X_*, Y^(−d∗)_*)` predict the held-out
modality `d∗` in closed form. The package exposes the conditional
regression coefficients directly:

```python
regr_src, regr_cov, Cov = model.get_transmodal_regression_coeffs(
    source_modes=[1],          # observed modalities
    target_modes=[2],          # modality to predict
    source_missing_masks=None, # or [bool array per source]: True = missing
)
A = regr_src.mean(("chain", "draw")).values       # (M_target, M_source)
C = regr_cov.mean(("chain", "draw")).values       # (M_target, L)

Y2_hat = A @ Y1_test + C @ X_test                 # posterior-predictive mean
```

`get_transmodal_regression_coeffs` returns:

- `regr_coeffs_from_sources` — the matrix `F^⊤ E^{-1}` that maps observed
  source data to target predictions,
- `regr_coeffs_from_covariates` — the part of the prediction explained by
  covariates after sources are accounted for,
- the conditional covariance.

`predict_dumb_version(...)` is an alternative single-sample API that
loops over draws and supports per-sample missingness in the sources. For
projecting new replicates onto the latent space (Theorem 5 of the paper)
see `get_projection_matrix(...)`.

#### Simulating from the generative model

`model.simulate_dataset(N, basis_matrices=..., L=L, ...)` draws a full
dataset from the prior and returns a dict with the predictor matrix,
per-mode `FunctionalData`, the latent parameters, and (optionally) the
Stan-data payload used to drive prior/posterior predictive workflows.
`model.sample_prior_predictive(...)` wraps this into an
`arviz.InferenceData` with `prior` and `prior_predictive` groups, and
`model.sample_posterior_predictive(...)` does the same on top of a fit.

#### Persistence

```python
model.save("/path/to/run")          # writes run.pkl, run_idata.nc, run_varimax.nc
loaded = MvtBLF.load("/path/to/run")
```

### Baselines: PLS and functional PLS

For function-on-function regression baselines the package ships with

```python
from MultivariateBLF.PartialLeastSquares import (
    PartialLeastSquares,
    FunctionalPartialLeastSquares,
    bootstrap_prediction_uncertainty,
)
```

Both classes are thin sklearn-style wrappers over
`sklearn.cross_decomposition.PLSRegression` operating directly on
`FunctionalData`. `FunctionalPartialLeastSquares` whitens the basis
coefficients by the Gram-matrix square root before fitting and
back-transforms predictions to the domain, following Beyaztas & Shang
(2019).

`bootstrap_prediction_uncertainty(model_class, model_params, X_train,
Y_train, X_test, n_boot=200, alpha=0.05)` resamples subjects with
replacement and returns a `BootstrapResult` with posterior-style mean and
quantile bands, plus the full stack of bootstrap predictions and
domain-projected x/y weights for downstream coverage analysis.

### Stan code generation

`stan_helper._assembleMBLF_stan_code_for_given_D(D, ...)` assembles the
Stan program for any number of modalities. The current cached programs
live in
[`MultivariateBLF/src/MultivariateBLF/stan/`](MultivariateBLF/src/MultivariateBLF/stan/);
each file name encodes `(D, MGPS, local_shrinkage_LatentFactors,
heteroscedastic_thetas, absence_of_psi2)`. Inspect the program of a
constructed model directly:

```python
print(model.stan_code)
```

---

## Reproducing the figures

All experiments are intentionally split into a *data-producing* script
(meant to be run on a server) and a *plotting* script (meant to be run
locally on the resulting CSVs). The CSVs themselves are not checked in.

### Figure 2 — effect of heteroscedastic basis scaling

```bash
cd figure2_confronto_processigaussiani_con_tau_riscalato
python code2.py
```

Produces `figure2_dashboard.png`. The script also exposes a `scaling_rule`
function at the top — edit it to test alternative reweightings of the
KL components.

### Figure 4 — cross-modal prediction MSE

[`prediction ability/run_prediction_ability.py`](prediction%20ability/run_prediction_ability.py)
matches Table 4 of the paper (Fourier truth, deliberate prior
misspecification) and fits four methods — `BMLF_Fourier`, `BMLF_GP`,
`PLS`, `fPLS_Fourier` — across `N ∈ {10, 25, 50, 75, 100}` and
`K ∈ {2, 3, 4}` with 50 seeds each. It writes `N{N}_K{K}_results.csv`
plus per-seed checkpoints under `per_seed/N{N}_K{K}/seed{seed}.csv`.

Restart logic is explicit: scenarios are listed as `(N, K, first_seed)`
tuples, so rows can be deleted, reordered, or have their seed rebased
without affecting the others. Per-seed CSVs are atomic and act as the
source of truth on restart.

```bash
cd "prediction ability"
python run_prediction_ability.py            # heavy: ~3000 BMLF fits in total
python plot.py                              # produces prediction_mse_boxplot_nk_palette.png
```

### Figures 5, 6, 7 — latent factor recovery and coverage

[`coverage_estimation_multiple/coverage_estimation_multiple.py`](coverage_estimation_multiple/coverage_estimation_multiple.py)
fits BMLF (GP basis and B-spline basis), PLS, and FPLS (GP basis and
B-spline basis) on B-spline–generated synthetic data, then computes
reconstruction MSE and empirical 80% pointwise coverage of the Varimax-RSP–
aligned loadings.

```bash
cd coverage_estimation_multiple
python coverage_estimation_multiple.py
python boxplots.py                           # produces boxplots_mse_coverage.png
```

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{ursino_bmlf_2025,
  title   = {A Bayesian latent factors approach for multimodal functional data},
  author  = {Ursino, Bruno and Epifani, Ilenia and Trov{\`o}, Francesco},
  journal = {Journal of the Royal Statistical Society, Series B},
  year    = {2026},
  note    = {Submitted}
}
```


