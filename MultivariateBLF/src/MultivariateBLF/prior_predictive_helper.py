from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


def _as_list_bool(x: Optional[Union[bool, Sequence[bool]]], D: int, default: bool) -> List[bool]:
    if x is None:
        return [default] * D
    if isinstance(x, bool):
        return [x] * D
    x = list(x)
    if len(x) != D:
        raise ValueError(f"Expected length {D}, got {len(x)}")
    if not all(isinstance(v, (bool, np.bool_)) for v in x):
        raise ValueError("Expected bools")
    return [bool(v) for v in x]


def _gamma_rng(rng: np.random.Generator, shape: float, rate: float, size=None) -> Array:
    """Stan gamma(shape, rate). NumPy uses scale=1/rate."""
    return rng.gamma(shape=shape, scale=1.0 / rate, size=size)


def _inv_gamma_rng(rng: np.random.Generator, shape: float, scale: float, size=None) -> Array:
    """Stan inv_gamma(shape, scale). Sample as 1 / Gamma(shape, rate=scale)."""
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / scale, size=size)


def _student_t_loc_scale_rng(rng: np.random.Generator, df: float, loc: Array, scale: Array) -> Array:
    z = rng.standard_t(df=df, size=loc.shape)
    return loc + scale * z


def _compress_to_theorem3_stats(B: Array, Delta: Array, Y: Array, *, ridge: float = 1e-10) -> Tuple[Array, float, int]:
    """
    A practical compression that is commonly used with Theorem-3 style likelihood rewrites:

    For each subject n, using observed rows I_n = {m : Delta[m,n] = 1},
      T[:,n] = argmin_t ||Y[I_n,n] - B[I_n,:] t||^2,
      rss_n = min value.
    Return T, sum_rss = sum_n rss_n, sum_Mdn = sum_n |I_n|.
    """
    M, p = B.shape
    _, N = Y.shape
    T = np.zeros((p, N), dtype=float)
    sum_rss = 0.0
    sum_Mdn = 0

    for n in range(N):
        obs = Delta[:, n].astype(bool)
        m_n = int(obs.sum())
        sum_Mdn += m_n
        if m_n == 0:
            continue
        Bw = B[obs, :]
        yw = Y[obs, n]
        G = Bw.T @ Bw
        if ridge > 0.0:
            G = G + ridge * np.eye(p)
        t = np.linalg.solve(G, Bw.T @ yw)
        T[:, n] = t
        r = yw - Bw @ t
        sum_rss += float(r @ r)

    return T, float(sum_rss), int(sum_Mdn)


@dataclass(frozen=True)
class MBLFSampler:
    """
    Sampler for the prior and a natural prior predictive of the model induced by your Stan code generator.

    Configuration flags match your generator:
      - MGPS: use delta/omega; else use lambda_var
      - local_shrinkage_LatentFactors: Student-t vs Normal for Lambda
      - heteroscedastic_thetas: per-mode heteroscedastic theta/T priors
      - absence_of_psi2: per-mode removal of psi2 (and theta in the Stan code)

    Note on homoscedastic tau:
      Your Stan code uses: resid ~ normal(0, sqrt(tau)) (homo case),
      whereas the hetero branch treats tau as precision.
      Use `match_stan_homoscedastic_tau_scale=True` to replicate the Stan program exactly.
    """
    D: int
    MGPS: bool = True
    local_shrinkage_LatentFactors: bool = True
    heteroscedastic_thetas: Optional[Union[bool, Sequence[bool]]] = None
    absence_of_psi2: Optional[Union[bool, Sequence[bool]]] = None

    def __post_init__(self):
        object.__setattr__(self, "heteroscedastic_thetas",
                           _as_list_bool(self.heteroscedastic_thetas, self.D, default=True))
        object.__setattr__(self, "absence_of_psi2",
                           _as_list_bool(self.absence_of_psi2, self.D, default=False))

    def sample(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        kind: str = "prior",  # "prior" or "prior_predictive"
        # core sizes/data
        N: int,
        K: int,
        L: int,
        X: Array,  # shape (L, N)
        p: Sequence[int],  # length D
        # hyperparameters (Stan parameterization)
        a_psi: Sequence[float],
        b_psi: Sequence[float],
        a_tau: Sequence[float],
        b_tau: Sequence[float],
        # MGPS vs non-MGPS hypers
        a1: Optional[float] = None,
        a2: Optional[float] = None,
        b_lambdavar: Optional[float] = None,
        a_lambdavar: Optional[float] = None,
        # shared
        nu: Optional[float] = None,
        sigma_regr_coeffs: float = 1.0,
        # optional: per-mode prior means for Lambda (each either None or (p[d],K))
        Lambda_prior_means: Optional[Sequence[Optional[Array]]] = None,
        # Stan homoscedastic tau scale toggle
        match_stan_homoscedastic_tau_scale: bool = False,
        # prior predictive inputs
        B: Optional[Sequence[Array]] = None,         # each (M_d, p_d)
        Delta: Optional[Sequence[Array]] = None,     # each (M_d, N) {0,1}
        obs_prob: float = 1.0,                       # used if Delta is None
        ridge: float = 1e-10,
        include_Y: bool = True,
    ) -> Dict[str, Any]:
        rng = rng if rng is not None else np.random.default_rng()

        if kind not in ("prior", "prior_predictive"):
            raise ValueError("kind must be 'prior' or 'prior_predictive'")

        X = np.asarray(X, dtype=float)
        if X.shape != (L, N):
            raise ValueError(f"X must have shape {(L, N)}, got {X.shape}")

        p = list(map(int, p))
        if len(p) != self.D:
            raise ValueError(f"p must have length D={self.D}")

        if Lambda_prior_means is None:
            Lambda_prior_means = [None] * self.D
        Lambda_prior_means = list(Lambda_prior_means)
        if len(Lambda_prior_means) != self.D:
            raise ValueError(f"Lambda_prior_means must have length D={self.D}")

        # --- sample regression + factor scores
        regr_coeffs = rng.normal(0.0, float(sigma_regr_coeffs), size=(K, L))
        eta = (regr_coeffs @ X) + rng.normal(0.0, 1.0, size=(K, N))

        out: Dict[str, Any] = {
            "regr_coeffs": regr_coeffs,
            "eta": eta,
        }

        # --- sample MGPS / lambda_var and get per-column scale for Lambda
        if self.MGPS:
            if a1 is None or a2 is None or b_lambdavar is None:
                raise ValueError("MGPS=True requires a1, a2, b_lambdavar")
            delta = np.empty(K, dtype=float)
            delta[0] = float(_gamma_rng(rng, float(a1), 1.0))
            if K > 1:
                delta[1:] = _gamma_rng(rng, float(a2), 1.0, size=K - 1)
            omega = np.empty(K, dtype=float)
            omega[0] = float(b_lambdavar) * delta[0]
            for k in range(1, K):
                omega[k] = omega[k - 1] * delta[k]
            lambda_col_scale = 1.0 / np.sqrt(omega)  # scale_expr = inv_sqrt(omega[k])
            out.update({"delta": delta, "omega": omega})
        else:
            if a_lambdavar is None or b_lambdavar is None:
                raise ValueError("MGPS=False requires a_lambdavar, b_lambdavar")
            lambda_var = float(_inv_gamma_rng(rng, float(a_lambdavar), float(b_lambdavar)))
            lambda_col_scale = np.full(K, np.sqrt(lambda_var), dtype=float)  # scale_expr = sqrt(lambda_var)
            out.update({"lambda_var": lambda_var})

        # --- psi2 (with placeholders for absent modes)
        psi2 = np.empty(self.D, dtype=float)
        out["psi2"] = psi2

        # --- per-mode sampling
        for d in range(self.D):
            dd = d + 1
            pd = p[d]
            hetero = self.heteroscedastic_thetas[d]
            no_psi2 = self.absence_of_psi2[d]

            # tau prior
            if hetero:
                tau = _gamma_rng(rng, float(a_tau[d]), float(b_tau[d]), size=pd)
            else:
                tau = float(_gamma_rng(rng, float(a_tau[d]), float(b_tau[d])))

            out[f"tau{dd}"] = tau

            # psi2 prior or placeholder
            if no_psi2:
                psi2[d] = 1.0
            else:
                psi2[d] = float(_inv_gamma_rng(rng, float(a_psi[d]), float(b_psi[d])))

            # Lambda prior mean: Stan uses mean = scale_expr * prior_mean (column wise)
            prior_mu = Lambda_prior_means[d]
            if prior_mu is not None:
                prior_mu = np.asarray(prior_mu, dtype=float)
                if prior_mu.shape != (pd, K):
                    raise ValueError(f"Lambda_prior_means[{d}] must have shape {(pd, K)}, got {prior_mu.shape}")
                mu_Lambda = prior_mu * lambda_col_scale[None, :]
            else:
                mu_Lambda = np.zeros((pd, K), dtype=float)

            scale_Lambda = np.broadcast_to(lambda_col_scale[None, :], (pd, K)).copy()

            # Lambda prior draw
            if self.local_shrinkage_LatentFactors:
                if nu is None:
                    raise ValueError("local_shrinkage_LatentFactors=True requires nu")
                Lambda = _student_t_loc_scale_rng(rng, float(nu), mu_Lambda, scale_Lambda)
            else:
                Lambda = rng.normal(mu_Lambda, scale_Lambda)

            out[f"Lambda{dd}"] = Lambda

            mean = Lambda @ eta  # (pd, N)

            # theta/T prior (depending on absence_of_psi2)
            if hetero:
                # hetero branch is unambiguous in Stan: tau is precision
                sd = 1.0 / np.sqrt(tau)[:, None]
                Z = rng.normal(0.0, 1.0, size=mean.shape)
                draw = mean + Z * sd
            else:
                # homoscedastic: either mimic Stan code (sd = sqrt(tau)) or treat tau as precision (sd = 1/sqrt(tau))
                if match_stan_homoscedastic_tau_scale:
                    sd0 = np.sqrt(tau)
                else:
                    sd0 = 1.0 / np.sqrt(tau)
                draw = mean + rng.normal(0.0, sd0, size=mean.shape)

            if no_psi2:
                out[f"T{dd}"] = draw
            else:
                out[f"theta{dd}"] = draw

        # --- prior predictive layer (optional)
        if kind == "prior_predictive":
            if B is None:
                raise ValueError("kind='prior_predictive' requires B (list of length D)")
            if len(B) != self.D:
                raise ValueError(f"B must have length D={self.D}")

            if Delta is None:
                if not (0.0 < obs_prob <= 1.0):
                    raise ValueError("obs_prob must be in (0,1]")
                Delta = [(rng.random((np.asarray(B[d]).shape[0], N)) < obs_prob).astype(float) for d in range(self.D)]
            else:
                if len(Delta) != self.D:
                    raise ValueError(f"Delta must have length D={self.D}")
                Delta = [np.asarray(Dd, dtype=float) for Dd in Delta]

            out["B"] = [np.asarray(Bd, dtype=float) for Bd in B]
            out["Delta"] = Delta

            for d in range(self.D):
                dd = d + 1
                Bd = out["B"][d]
                Dd = Delta[d]
                if Bd.shape[1] != p[d]:
                    raise ValueError(f"B[{d}] has p={Bd.shape[1]} but expected p[d]={p[d]}")

                if self.absence_of_psi2[d]:
                    # This mode has no psi2 and no theta in Stan; treat T as the observable coefficient layer.
                    T = out[f"T{dd}"]
                    Y = Bd @ T  # noiseless
                    if include_Y:
                        out[f"Y{dd}"] = Y
                    out[f"T_data{dd}"] = T.copy()
                    out[f"sum_rss{dd}"] = 0.0
                    out[f"sum_Mdn{dd}"] = int(Dd.sum())
                else:
                    theta = out[f"theta{dd}"]
                    eps = rng.normal(0.0, np.sqrt(out["psi2"][d]), size=(Bd.shape[0], N))
                    Y = Bd @ theta + eps
                    if include_Y:
                        out[f"Y{dd}"] = Y

                    T_hat, sum_rss, sum_Mdn = _compress_to_theorem3_stats(Bd, Dd, Y, ridge=ridge)
                    # what you'd feed into Stan as "data"
                    out[f"T_data{dd}"] = T_hat
                    out[f"sum_rss{dd}"] = float(sum_rss)
                    out[f"sum_Mdn{dd}"] = int(sum_Mdn)

        return out


# sampler = MBLFSampler(
#     D=2,
#     MGPS=True,
#     local_shrinkage_LatentFactors=True,
#     heteroscedastic_thetas=[True, False],
#     absence_of_psi2=[False, False],
# )

# rng = np.random.default_rng(0)
# N, K, L = 50, 3, 5
# X = rng.normal(size=(L, N))
# p = [10, 12]

# # Prior draw
# prior = sampler.sample(
#     rng=rng, kind="prior",
#     N=N, K=K, L=L, X=X, p=p,
#     a_psi=[2, 2], b_psi=[2, 2],
#     a_tau=[2, 2], b_tau=[2, 2],
#     a1=2.0, a2=2.0, b_lambdavar=1.0,
#     nu=5.0, sigma_regr_coeffs=1.0,
# )

# # Prior predictive draw (generate Y and compute T_data/sum_rss/sum_Mdn)
# B = [rng.normal(size=(100, 10)), rng.normal(size=(80, 12))]
# pp = sampler.sample(
#     rng=rng, kind="prior_predictive",
#     N=N, K=K, L=L, X=X, p=p,
#     B=B, obs_prob=0.9,
#     a_psi=[2, 2], b_psi=[2, 2],
#     a_tau=[2, 2], b_tau=[2, 2],
#     a1=2.0, a2=2.0, b_lambdavar=1.0,
#     nu=5.0, sigma_regr_coeffs=1.0,
# )
