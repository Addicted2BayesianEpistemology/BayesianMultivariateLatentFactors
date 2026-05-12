from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Make sure nothing tries to pop up on screen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cmdstanpy
import arviz as az


# -----------------------
# Configuration
# -----------------------
FIGURES_DIR = Path("Figures")
OUT_BASE = Path("Explorations")
OUT_BASE.mkdir(parents=True, exist_ok=True)

STAN_FILE = Path("./explore_coverage_results.stan")
STAN_SKEW_FILE = Path("./explore_coverage_results_skewnormal.stan")

# Map "method" -> filename in each Figures/<subdir>/
METHOD_FILES = {
    "blf_gp": "coverage_all_seeds.csv",
    "blf_bspline": "coverage_all_seeds_bspline.csv",
    "pls": "coverage_all_seeds_pls.csv",
    "fpls_gp": "coverage_all_seeds_fpls_gp.csv",
    "fpls_bspline": "coverage_all_seeds_fpls_bspline.csv",
}

# Stan sampling config
CHAINS = 8
ITER_WARMUP = 1000
ITER_SAMPLING = 1000
SEED = 12345

# Coverage clipping (to avoid 0/1 probs)
CLIP_LO, CLIP_HI = 1 / 50, 49 / 50


def mean_and_hdi(idata: az.InferenceData, var: str, hdi_prob: float = 0.95) -> tuple[float, float, float]:
    """Return posterior mean and (low, high) HDI for a scalar variable in idata.posterior."""
    mean_val = float(idata.posterior[var].mean().values)
    hdi_arr = az.hdi(idata, var_names=[var], hdi_prob=hdi_prob)[var].values
    hdi_arr = np.asarray(hdi_arr).reshape(-1)
    if hdi_arr.size != 2:
        raise ValueError(f"Unexpected HDI shape for {var}: {hdi_arr.shape}")
    low, high = map(float, hdi_arr)
    return mean_val, low, high


# -----------------------
# Compile Stan models once
# -----------------------
stan_model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
stan_model_skew = cmdstanpy.CmdStanModel(stan_file=str(STAN_SKEW_FILE))


# -----------------------
# Main loop over Figures/*
# -----------------------
subdirs = sorted([p for p in FIGURES_DIR.iterdir() if p.is_dir()])

for subdir in subdirs:
    out_dir = OUT_BASE / subdir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + stack data
    dfs = []
    missing = []
    for method, fname in METHOD_FILES.items():
        fpath = subdir / fname
        if not fpath.exists():
            missing.append(fname)
            continue
        tmp = pd.read_csv(fpath)
        tmp["method"] = method
        dfs.append(tmp)

    if not dfs:
        (out_dir / "RUN_LOG.txt").write_text(
            f"No input CSVs found in {subdir}. Missing (expected): {list(METHOD_FILES.values())}\n"
        )
        continue

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_dir / "combined_coverage_all_methods.csv", index=False)

    # ---- Boxplots (saved only)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, y="mse", x="method", ax=axs[0], hue="method")
    axs[0].set_yscale("log")
    axs[0].set_title("")

    sns.boxplot(data=df, y="coverage_pct", x="method", ax=axs[1], hue="method")
    axs[1].axhline(80, linestyle="--", color="C2")
    axs[1].set_title("")

    axs[0].text(
        -0.12, 1.03, "(a)",
        transform=axs[0].transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    axs[1].text(
        -0.12, 1.03, "(b)",
        transform=axs[1].transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    fig.subplots_adjust(top=0.92, wspace=0.25)
    fig.savefig(out_dir / "boxplots_mse_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Stan fits + posterior plots (NO idata saving)
    methods_in_df = list(df["method"].dropna().unique())

    # Collect summaries for a single text output per Figures/<subdir>/
    summary_rows = []

    for method in methods_in_df:
        y = (df.loc[df["method"] == method, "coverage_pct"].to_numpy(dtype=float) / 100.0)
        y = np.clip(y, CLIP_LO, CLIP_HI)
        data_dict = {"N": int(len(y)), "y": y.tolist()}

        # ---- Model 1
        fit1 = stan_model.sample(
            data=data_dict,
            chains=CHAINS,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            seed=SEED,
            show_progress=False,
        )
        idata1 = az.from_cmdstanpy(posterior=fit1)

        mu_mean, mu_lo, mu_hi = mean_and_hdi(idata1, "mu", hdi_prob=0.95)

        fig, ax = plt.subplots(figsize=(10, 5))
        az.plot_posterior(idata1, var_names=["mu"], hdi_prob=0.95, ax=ax)
        ax.set_title(f"Posterior of coverage proportion μ — {method}")
        fig.tight_layout()
        fig.savefig(out_dir / f"posterior_mu_{method}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- Model 2 (skew-normal on transformed z)
        fit2 = stan_model_skew.sample(
            data=data_dict,
            chains=CHAINS,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            seed=SEED,
            show_progress=False,
        )
        idata2 = az.from_cmdstanpy(posterior=fit2)

        cdf_mean, cdf_lo, cdf_hi = mean_and_hdi(idata2, "cdf_at_zero", hdi_prob=0.95)
        avg_mean, avg_lo, avg_hi = mean_and_hdi(idata2, "avg_y_rep", hdi_prob=0.95)

        # Optional: posterior plots for the new quantities
        fig, ax = plt.subplots(figsize=(10, 5))
        az.plot_posterior(idata2, var_names=["cdf_at_zero"], hdi_prob=0.95, ax=ax)
        ax.set_title(f"Posterior of cdf_at_zero — {method}")
        fig.tight_layout()
        fig.savefig(out_dir / f"posterior_cdf_at_zero_{method}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        az.plot_posterior(idata2, var_names=["avg_y_rep"], hdi_prob=0.95, ax=ax)
        ax.set_title(f"Posterior of avg_y_rep — {method}")
        fig.tight_layout()
        fig.savefig(out_dir / f"posterior_avg_y_rep_{method}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append(
            {
                "method": method,
                "mu_mean": mu_mean,
                "mu_hdi_low": mu_lo,
                "mu_hdi_high": mu_hi,
                "cdf_at_zero_mean": cdf_mean,
                "cdf_at_zero_hdi_low": cdf_lo,
                "cdf_at_zero_hdi_high": cdf_hi,
                "avg_y_rep_mean": avg_mean,
                "avg_y_rep_hdi_low": avg_lo,
                "avg_y_rep_hdi_high": avg_hi,
            }
        )

    # ---- Summary text output
    lines = []
    lines.append(f"Figures subdirectory: {subdir}\n")
    lines.append(f"Output directory: {out_dir}\n\n")

    lines.append("Posterior summaries (mean and 95% HDI)\n")
    lines.append(
        "method\tmu_mean\tmu_hdi_low\tmu_hdi_high\t"
        "cdf_mean\tcdf_hdi_low\tcdf_hdi_high\t"
        "avg_mean\tavg_hdi_low\tavg_hdi_high\n"
    )

    for row in sorted(summary_rows, key=lambda r: r["method"]):
        lines.append(
            f'{row["method"]}\t'
            f'{row["mu_mean"]:.4f}\t{row["mu_hdi_low"]:.3f}\t{row["mu_hdi_high"]:.3f}\t'
            f'{row["cdf_at_zero_mean"]:.4f}\t{row["cdf_at_zero_hdi_low"]:.3f}\t{row["cdf_at_zero_hdi_high"]:.3f}\t'
            f'{row["avg_y_rep_mean"]:.4f}\t{row["avg_y_rep_hdi_low"]:.3f}\t{row["avg_y_rep_hdi_high"]:.3f}\n'
        )

    (out_dir / "posterior_summary.txt").write_text("".join(lines))

    # Also save as CSV for convenience
    pd.DataFrame(summary_rows).sort_values("method").to_csv(out_dir / "posterior_summary.csv", index=False)
