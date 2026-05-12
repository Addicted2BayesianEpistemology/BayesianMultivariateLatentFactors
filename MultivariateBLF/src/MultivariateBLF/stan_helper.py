
from importlib import resources  # Python ≥3.9
import io
from typing import List, Optional, Union
import numpy as np

def _load_stan_code(filename: str) -> str:
    """
    Loads the Stan source code bundled within the package.
    """
    import MultivariateBLF.stan as stan_dir
    with resources.open_text(stan_dir, filename) as f:
        return f.read()    


def _assembleMBLF_stan_code_for_given_D(
    D: int,
    heteroscedastic_thetas : Optional[Union[bool, List[bool]]] = None,
    MGPS: bool = True,
    local_shrinkage_LatentFactors: bool = True,
    decenter_Lambda: Optional[List[List[Union[None, np.ndarray]]]] = None, # if not None, it's a list of length D, of lists of length K, each element is either None or a np.ndarray with shape (p_d,) representing the prior mean for Lambda_d[,k]
    absence_of_psi2: Optional[List[bool]] = None,
) -> str:
    return _assembleMBLF_stan_code_for_given_D_with_the_thetas(D, heteroscedastic_thetas, MGPS, local_shrinkage_LatentFactors, decenter_Lambda, absence_of_psi2)


def _assembleMBLF_stan_code_for_given_D_with_the_thetas(
    D: int,
    heteroscedastic_thetas : Optional[Union[bool, List[bool]]] = None,
    MGPS: bool = True,
    local_shrinkage_LatentFactors: bool = True,
    decenter_Lambda: Optional[List[List[Union[None, np.ndarray]]]] = None, # if not None, it's a list of length D, of lists of length K, each element is either None or a np.ndarray with shape (p_d,) representing the prior mean for Lambda_d[,k]
    absence_of_psi2: Optional[List[bool]] = None,
    slack_factor_on_psi2: float = 10.0,
) -> str:
    """
    Generate Stan code for the Theorem 3 likelihood with D functional modes,
    avoiding arrays of matrices by unrolling per-mode blocks (B1..BD, etc.).

    Flags
    -----
    MGPS : if False, drop multiplicative gamma process (delta/omega),
           behave as if omega_k == 1 (global shrinkage disabled).
    local_shrinkage_LatentFactors : if False, use Normal(0, scale) instead of
           Student_t(nu[d], 0, scale) for Lambda (nu -> infinity case).
    """
    if heteroscedastic_thetas is not None and len(heteroscedastic_thetas) != D:
        raise ValueError("length of heteroscedastic_thetas must equal D")
    if heteroscedastic_thetas is None:
        heteroscedastic_thetas = [True] * D  # default: all heteroscedastic
    if isinstance(heteroscedastic_thetas, bool):
        heteroscedastic_thetas = [heteroscedastic_thetas] * D
    if not all(isinstance(h, bool) for h in heteroscedastic_thetas):
        raise ValueError("heteroscedastic_thetas must be a bool or list of bools")
    if absence_of_psi2 is None:
        absence_of_psi2 = [False] * D  # default: all modes have psi2
    if len(absence_of_psi2) != D:
        raise ValueError("length of absence_of_psi2 must equal D")
    for absent in absence_of_psi2:
        if not isinstance(absent, bool):
            raise ValueError("absence_of_psi2 must be a bool or list of bools")

    decenter_Lambda_transformed = []
    if decenter_Lambda is None:
        decenter_Lambda_transformed = [False] * D
    else:
        for d in range(D):
            if decenter_Lambda[d] is None:
                decenter_Lambda_transformed.append(False)
            else:
                what_to_append = False
                for decenter in decenter_Lambda[d]:
                    if decenter is not None:
                        what_to_append = True
                decenter_Lambda_transformed.append(what_to_append)


    lines = []
    emit = lines.append

    emit("  // Auto-generated Stan code (Theorem 3 likelihood, Section 4.1)")
    emit("data {")
    emit("  int<lower=1> N; // Number of observations")
    emit("  int<lower=1> D; // Number of functional modes")
    emit("  int<lower=1> K; // Number of latent factors")
    emit("  int<lower=0> L; // Number of predictors")
    emit("")
    emit("  matrix[L, N] X; // Predictor matrix")
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Data for functional mode {d}")
        emit(f"  int<lower=1> p{d}; // Number of basis functions for mode {d}")
        emit(f"  int<lower=1> M{d}; // Number of locations (union grid) for mode {d}")
        emit(f"  matrix[M{d}, p{d}] B{d}; // Basis matrix")
        emit(f"  matrix[M{d}, N] Delta{d}; // 1=observed, 0=missing at each grid loc/subject")
        emit(f"  matrix[p{d}, N] T{d}; // Compressed coefficients per Theorem 3")
        emit(f"  real<lower=0> sum_rss{d}; // Sum_n ||R_n^{(d)}||^2")
        emit(f"  int<lower=0> sum_Mdn{d}; // Sum_n M_(d,n) (observed counts)")
        if decenter_Lambda_transformed[d - 1]:
            emit(f"  // Decentering prior means for Lambda{d}")
            emit(f"  matrix[p{d}, K] Lambda{d}_prior_means; // prior means for Lambda{d}")
    emit("")
    emit("  // Hyperparameters")
    emit("  // measurement error variance priors")
    emit("  vector<lower=0>[D] a_psi;") # we assume to have all hyperaparameters for psi2 nonetheless, for simplicity
    emit("  vector<lower=0>[D] b_psi;")
    emit("  // basis coefficients precisions priors")
    emit("  vector<lower=0>[D] a_tau;")
    emit("  vector<lower=0>[D] b_tau;")
    if MGPS:
        emit("  // factor loadings priors (MGPS controls if deltas/omegas are active)")
        emit("  real<lower=0> a1; // alpha1 for MGPS (delta_1)")
        emit("  real<lower=0> a2; // alpha2 for MGPS (delta_k>=2)")
        for d in range(1, D + 1):
            emit(f"  real<lower=0> b_lambdavar{d}; // rate parameter for the precision of the MGPS omegas")
    else:
        emit("  // loadings variance prior")
        emit("  real<lower=0> a_lambdavar; // shape parameter for loadings variance (inverse-gamma prior)")
        for d in range(1, D + 1):
            emit(f"  real<lower=0> b_lambdavar{d}; // rate parameter for loadings variance (inverse-gamma prior)")

    if local_shrinkage_LatentFactors:
        emit("  real<lower=0> nu; // df for Student-t prior on Lambda")
    emit("  // regression coefficients prior")
    emit("  real<lower=0> sigma_regr_coeffs;")
    emit("}")
    emit("")
    emit("")
    emit("transformed data {")
    # if any(absence_of_psi2):
    #     lista_da_hardcodare = [i + 1 for i, absent in enumerate(absence_of_psi2) if not absent]
    #     emit(f"  int number_of_missing_psi2_modes = {sum(absence_of_psi2)};")
    #     emit(f"  int number_of_present_psi2_modes = {D - sum(absence_of_psi2)};")
    #     emit("  array[number_of_present_psi2_modes] int modes_with_psi2 = " + "{" + ", ".join(str(i) for i in lista_da_hardcodare) + "};")
    emit("}")
    emit("")
    emit("")
    emit("parameters {")
    emit("  matrix[K, L] regr_coeffs; // regression coefficients")
    emit("  matrix[K, N] eta; // scores of latent factors")
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Parameters for functional mode {d}")
        emit(f"  matrix[p{d}, K] Lambda{d}; // factor loadings for mode {d}")
        if heteroscedastic_thetas[d - 1]:
            emit(f"  vector<lower=0>[p{d}] tau{d}; // basis-coeff precisions for mode {d} - heteroscedastic")
        else:
            emit(f"  real<lower=0> tau{d}; // basis-coeff precisions for mode {d} - homoscedastic")
        if absence_of_psi2[d - 1]:
            emit(f"  // mode {d} has no psi2 parameter (hence also no theta{d})")
        else:
            emit(f"  matrix[p{d}, N] theta{d}; // latent basis coefficients for mode {d}")
    if MGPS:
        emit(f"  vector<lower=0>[K] delta; // MGPS global multipliers")
    else:
        emit(f"  real<lower=0> lambda_var; // loadings variance")
    emit("")
    if any(absence_of_psi2):
        emit(f"  vector<lower=0>[{D - sum(absence_of_psi2)}] unconstrained_helper_psi2; // measurement error variances for modes that have them")
    else:
        emit("  vector<lower=0>[D] unconstrained_psi2; // measurement error variances per mode")
    emit("}")
    emit("")
    emit("")
    emit("transformed parameters {")
    def psi2_correction_formula(varname: str) -> str:
        retstring = f"(b_psi[{d}] + 0.5 * sum_rss{d}) / (a_psi[{d}] + 0.5 * sum_Mdn{d} - 1) * (1 + ({slack_factor_on_psi2} * {varname}) / sqrt(a_psi[{d}] + 0.5 * sum_Mdn{d} - 2))"
        return retstring
    if any(absence_of_psi2):
        emit(f"  vector<lower=0>[{D - sum(absence_of_psi2)}] helper_psi2; // measurement error variances for modes that have them")
        helper_idx = 1
        for d in range(1, D + 1):
            if not absence_of_psi2[d - 1]:
                emit(f"  helper_psi2[{helper_idx}] = {psi2_correction_formula(f"unconstrained_helper_psi2[{helper_idx}]")};")
                helper_idx += 1
    else:
        emit("  vector<lower=0>[D] psi2; // measurement error variances per mode")
        for d in range(1, D + 1):
            emit(f"  psi2[{d}] = {psi2_correction_formula(f'unconstrained_psi2[{d}]')};")
    if any(absence_of_psi2):
        emit("  vector<lower=0>[D] psi2;")
        for d in range(1, D + 1):
            helper_idx = 1
            if absence_of_psi2[d - 1]:
                emit(f"  psi2[{d}] = 1.0; // mode {d} has no psi2 parameter, this is an unused placeholder")
            else:
                emit(f"  psi2[{d}] = helper_psi2[{helper_idx}];")
                helper_idx += 1
    if MGPS:
        emit(f"  vector<lower=0>[K] omega;")
        emit(f"  omega[1] = delta[1];")
        emit(f"  for (k in 2:K) omega[k] = omega[k-1] * delta[k];")
        emit("")
    emit("}")
    emit("")
    emit("")
    emit("model {")
    emit("  // Priors for regression and scores")
    emit("  to_vector(regr_coeffs) ~ normal(0, sigma_regr_coeffs);")
    emit("  to_vector(eta) ~ normal(to_vector(regr_coeffs * X), 1);")
    emit("")
    # MGPS priors on delta if enabled
    if MGPS:
        emit(f"  // MGPS priors")
        emit(f"  delta[1] ~ gamma(a1, 1);")
        emit(f"  for (k in 2:K) delta[k] ~ gamma(a2, 1);")
    else:
        emit(f"  // Loadings variance prior")
        emit(f"  lambda_var ~ inv_gamma(a_lambdavar, 1.0);")
    emit("")
    # Per-mode priors and likelihood
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Mode {d}: priors for tau, Lambda, theta; and Theorem 3 likelihood")
        if heteroscedastic_thetas[d - 1]:
            emit(f"  // heteroscedastic basis coefficients theta")
        else:
            emit(f"  //  homoscedastic  basis coefficients theta")
        # tau prior: heteroscedastic or homoscedastic works both ways
        emit(f"  tau{d} ~ gamma(a_tau[{d}], b_tau[{d}]);")

        # Choose the scale for Lambda prior
        scale_expr = f"inv_sqrt(omega[k]) * sqrt(b_lambdavar{d})" if MGPS else f"sqrt(lambda_var) * sqrt(b_lambdavar{d})"

        # Lambda prior: Student-t (local shrinkage) or Normal (nu -> inf)
        if local_shrinkage_LatentFactors:
            if decenter_Lambda_transformed[d - 1]:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ student_t(nu, {scale_expr} * Lambda{d}_prior_means[, k], {scale_expr});")
            else:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ student_t(nu, 0, {scale_expr});")
        else:
            if decenter_Lambda_transformed[d - 1]:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ normal({scale_expr} * Lambda{d}_prior_means[, k], {scale_expr});")
            else:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ normal(0, {scale_expr});")

        # theta prior given Lambda, eta, tau
        # emit("  {")
        # emit(f"    matrix[p{d}, p{d}] Prec_theta{d} = diag_matrix(tau{d});")
        # emit(f"    for (n in 1:N)")
        # emit(f"      theta{d}[:, n] ~ multi_normal_prec(Lambda{d} * eta[:, n], Prec_theta{d});")
        # emit("  }")
        emit("  {")
        if absence_of_psi2[d - 1]:
            emit(f"    matrix[p{d}, N] resid_theta{d} = T{d} - Lambda{d} * eta; // no theta{d} in this mode so we pass directly to T{d}")
        else:
            emit(f"    matrix[p{d}, N] resid_theta{d} = theta{d} - Lambda{d} * eta;")

        if heteroscedastic_thetas[d - 1]:
            emit(f"    resid_theta{d} = diag_pre_multiply(sqrt(tau{d}), resid_theta{d});")
            emit(f"    to_vector(resid_theta{d}) ~ normal(0, 1);")
            emit(f"    target += 0.5 * sum(log(tau{d})) * N; // the terms that would have appeared in the theta likelihood if we had written with multi_normal_prec")
        else:
            emit(f"    to_vector(resid_theta{d}) ~ normal(0, sqrt(tau{d}));")
            # emit(f"    resid_theta{d} = resid_theta{d} / sqrt(tau{d});")
        emit("  }")

        # Theorem 3: IG kernel for psi2 and masked quadratic term
        if absence_of_psi2[d - 1]:
            emit(f"  // mode {d} has no psi2 parameter, nor the residual of T-theta, skipping its likelihood contribution")
        else:
            emit(f"  target += inv_gamma_lpdf(psi2[{d}] | a_psi[{d}] + 0.5 * sum_Mdn{d},")
            emit(f"                                       b_psi[{d}] + 0.5 * sum_rss{d});")
            emit("  {")
            emit(f"    matrix[M{d}, N] E{d} = B{d} * (T{d} - theta{d});")
            # emit(f"    target += -0.5 / psi2[{d}] * dot_product(to_vector(Delta{d}), to_vector(E{d} .* E{d}));")
            emit(f"    target += -0.5 / psi2[{d}] * dot_product(to_vector(Delta{d}), square(to_vector(E{d})));")
            emit("  }")
    emit("}")
    emit("")
    emit("")
    emit("generated quantities {")
    # emit("  vector[D] ll_mode;")
    # for d in range(1, D + 1):
    #     emit("  {")
    #     emit(f"    matrix[M{d}, N] E{d} = B{d} * (T{d} - theta{d});")
    #     emit(f"    real quad{d} = dot_product(to_vector(Delta{d}), to_vector(E{d} .* E{d}));")
    #     emit(f"    real shp{d} = a_psi[{d}] + 0.5 * sum_Mdn{d};")
    #     emit(f"    real rte{d} = b_psi[{d}] + 0.5 * sum_rss{d};")
    #     emit(f"    ll_mode[{d}] = inv_gamma_lpdf(psi2[{d}] | shp{d}, rte{d}) - 0.5 / psi2[{d}] * quad{d};")
    #     emit("  }")
    emit("}")

    return "\n".join(lines)


### N.B. THIS FUNCTION IS WRONG! DON'T USE, IT'S LEFT FOR FUTURE DEVELOPMENT PURPOSES ONLY ###
# the error is that I forgot about Wn which integrates the missing data mechanism...
def _assembleMBLF_stan_code_for_given_D_without_the_thetas(
    D: int,
    heteroscedastic_thetas : Optional[Union[bool, List[bool]]] = None,
    MGPS: bool = True,
    local_shrinkage_LatentFactors: bool = True,
    decenter_Lambda: Optional[List[List[Union[None, np.ndarray]]]] = None, # if not None, it's a list of length D, of lists of length K, each element is either None or a np.ndarray with shape (p_d,) representing the prior mean for Lambda_d[,k]
    absence_of_psi2: Optional[List[bool]] = None,
) -> str:
    """
    Generate Stan code for the Theorem 3 likelihood with D functional modes,
    avoiding arrays of matrices by unrolling per-mode blocks (B1..BD, etc.).

    Flags
    -----
    MGPS : if False, drop multiplicative gamma process (delta/omega),
           behave as if omega_k == 1 (global shrinkage disabled).
    local_shrinkage_LatentFactors : if False, use Normal(0, scale) instead of
           Student_t(nu[d], 0, scale) for Lambda (nu -> infinity case).
    """
    if heteroscedastic_thetas is not None and len(heteroscedastic_thetas) != D:
        raise ValueError("length of heteroscedastic_thetas must equal D")
    if heteroscedastic_thetas is None:
        heteroscedastic_thetas = [True] * D  # default: all heteroscedastic
    if isinstance(heteroscedastic_thetas, bool):
        heteroscedastic_thetas = [heteroscedastic_thetas] * D
    if not all(isinstance(h, bool) for h in heteroscedastic_thetas):
        raise ValueError("heteroscedastic_thetas must be a bool or list of bools")
    if absence_of_psi2 is None:
        absence_of_psi2 = [False] * D  # default: all modes have psi2
    if len(absence_of_psi2) != D:
        raise ValueError("length of absence_of_psi2 must equal D")
    for absent in absence_of_psi2:
        if not isinstance(absent, bool):
            raise ValueError("absence_of_psi2 must be a bool or list of bools")

    decenter_Lambda_transformed = []
    if decenter_Lambda is None:
        decenter_Lambda_transformed = [False] * D
    else:
        for d in range(D):
            if decenter_Lambda[d] is None:
                decenter_Lambda_transformed.append(False)
            else:
                what_to_append = False
                for decenter in decenter_Lambda[d]:
                    if decenter is not None:
                        what_to_append = True
                decenter_Lambda_transformed.append(what_to_append)

    lines = []
    emit = lines.append

    emit("  // Auto-generated Stan code (Theorem 3 likelihood, Section 4.1)")
    emit("data {")
    emit("  int<lower=1> N; // Number of observations")
    emit("  int<lower=1> D; // Number of functional modes")
    emit("  int<lower=1> K; // Number of latent factors")
    emit("  int<lower=0> L; // Number of predictors")
    emit("")
    emit("  matrix[L, N] X; // Predictor matrix")
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Data for functional mode {d}")
        emit(f"  int<lower=1> p{d}; // Number of basis functions for mode {d}")
        emit(f"  int<lower=1> M{d}; // Number of locations (union grid) for mode {d}")
        emit(f"  matrix[M{d}, p{d}] B{d}; // Basis matrix")
        emit(f"  matrix[M{d}, N] Delta{d}; // 1=observed, 0=missing at each grid loc/subject")
        emit(f"  matrix[p{d}, N] T{d}; // Compressed coefficients per Theorem 3")
        emit(f"  real<lower=0> sum_rss{d}; // Sum_n ||R_n^{(d)}||^2")
        emit(f"  int<lower=0> sum_Mdn{d}; // Sum_n M_(d,n) (observed counts)")
        if decenter_Lambda_transformed[d - 1]:
            emit(f"  // Decentering prior means for Lambda{d}")
            emit(f"  matrix[p{d}, K] Lambda{d}_prior_means; // prior means for Lambda{d}")
    emit("")
    emit("  // Hyperparameters")
    emit("  // measurement error variance priors")
    emit("  vector<lower=0>[D] a_psi;") # we assume to have all hyperaparameters for psi2 nonetheless, for simplicity
    emit("  vector<lower=0>[D] b_psi;")
    emit("  // basis coefficients precisions priors")
    emit("  vector<lower=0>[D] a_tau;")
    emit("  vector<lower=0>[D] b_tau;")
    if MGPS:
        emit("  // factor loadings priors (MGPS controls if deltas/omegas are active)")
        emit("  real<lower=0> a1; // alpha1 for MGPS (delta_1)")
        emit("  real<lower=0> a2; // alpha2 for MGPS (delta_k>=2)")
        emit("  real<lower=0> b_lambdavar; // rate parameter for the precision of the MGPS omegas")
    else:
        emit("  // loadings variance prior")
        emit("  real<lower=0> a_lambdavar; // shape parameter for loadings variance (inverse-gamma prior)")
        emit("  real<lower=0> b_lambdavar; // rate parameter for loadings variance (inverse-gamma prior)")

    if local_shrinkage_LatentFactors:
        emit("  real<lower=0> nu; // df for Student-t prior on Lambda")
    emit("  // regression coefficients prior")
    emit("  real<lower=0> sigma_regr_coeffs;")
    emit("}")
    emit("")
    emit("")
    emit("transformed data {")
    # if any(absence_of_psi2):
    #     lista_da_hardcodare = [i + 1 for i, absent in enumerate(absence_of_psi2) if not absent]
    #     emit(f"  int number_of_missing_psi2_modes = {sum(absence_of_psi2)};")
    #     emit(f"  int number_of_present_psi2_modes = {D - sum(absence_of_psi2)};")
    #     emit("  array[number_of_present_psi2_modes] int modes_with_psi2 = " + "{" + ", ".join(str(i) for i in lista_da_hardcodare) + "};")
    for d in range(1, D + 1):
        emit(f"  matrix[p{d}, p{d}] G{d} = B{d}' * B{d};")
        emit(f"  matrix[p{d}, p{d}] I{d} = diag_matrix(rep_vector(1.0, p{d}));")
    emit("}")
    emit("")
    emit("")
    emit("parameters {")
    emit("  matrix[K, L] regr_coeffs; // regression coefficients")
    emit("  matrix[K, N] eta; // scores of latent factors")
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Parameters for functional mode {d}")
        emit(f"  matrix[p{d}, K] Lambda{d}; // factor loadings for mode {d}")
        if heteroscedastic_thetas[d - 1]:
            emit(f"  vector<lower=0>[p{d}] tau{d}; // basis-coeff precisions for mode {d} - heteroscedastic")
        else:
            emit(f"  real<lower=0> tau{d}; // basis-coeff precisions for mode {d} - homoscedastic")
        if absence_of_psi2[d - 1]:
            emit(f"  // mode {d} has no psi2 parameter (hence also no theta{d})")
        else:
            emit(f"  matrix[p{d}, N] theta{d}; // latent basis coefficients for mode {d}")
    if MGPS:
        emit(f"  vector<lower=0>[K] delta; // MGPS global multipliers")
    else:
        emit(f"  real<lower=0> lambda_var; // loadings variance")
    emit("")
    if any(absence_of_psi2):
        emit(f"  vector<lower=0>[{D - sum(absence_of_psi2)}] helper_psi2; // measurement error variances for modes that have them")
    else:
        emit("  vector<lower=0>[D] psi2; // measurement error variances per mode")
    emit("}")
    emit("")
    emit("")
    emit("transformed parameters {")
    if any(absence_of_psi2):
        emit("  vector<lower=0>[D] psi2;")
        for d in range(1, D + 1):
            helper_idx = 1
            if absence_of_psi2[d - 1]:
                emit(f"  psi2[{d}] = 1.0; // mode {d} has no psi2 parameter, this is an unused placeholder")
            else:
                emit(f"  psi2[{d}] = helper_psi2[{helper_idx}];")
                helper_idx += 1
    if MGPS:
        emit(f"  vector<lower=0>[K] omega;")
        emit(f"  omega[1] = b_lambdavar * delta[1];")
        emit(f"  for (k in 2:K) omega[k] = omega[k-1] * delta[k];")
        emit("")
    for d in range(1, D + 1):
        emit(f"  matrix[p{d}, p{d}] M{d};")
        if heteroscedastic_thetas[d - 1]:
            emit(f"  M{d} = I{d} + 1.0 / psi2[{d}] * diag_post_multiply(diag_pre_multiply(sqrt(tau{d}), G{d}), sqrt(tau{d}));")
        else:
            emit(f"  M{d} = I{d} + tau{d} / psi2[{d}] * G{d};")
        emit(f"  matrix[p{d}, p{d}] Lchol{d} = cholesky_decompose(M{d});")
    emit("}")
    emit("")
    emit("")
    emit("model {")
    emit("  // Priors for regression and scores")
    emit("  to_vector(regr_coeffs) ~ normal(0, sigma_regr_coeffs);")
    emit("  to_vector(eta) ~ normal(to_vector(regr_coeffs * X), 1);")
    emit("")
    # MGPS priors on delta if enabled
    if MGPS:
        emit(f"  // MGPS priors")
        emit(f"  delta[1] ~ gamma(a1, 1);")
        emit(f"  for (k in 2:K) delta[k] ~ gamma(a2, 1);")
    else:
        emit(f"  // Loadings variance prior")
        emit(f"  lambda_var ~ inv_gamma(a_lambdavar, b_lambdavar);")
    emit("")
    # Per-mode priors and likelihood
    for d in range(1, D + 1):
        emit("")
        emit(f"  // Mode {d}: priors for tau, Lambda, theta; and Theorem 3 likelihood")
        if heteroscedastic_thetas[d - 1]:
            emit(f"  // heteroscedastic basis coefficients theta")
        else:
            emit(f"  //  homoscedastic  basis coefficients theta")
        # tau prior: heteroscedastic or homoscedastic works both ways
        emit(f"  tau{d} ~ gamma(a_tau[{d}], b_tau[{d}]);")

        # Choose the scale for Lambda prior
        scale_expr = f"inv_sqrt(omega[k])" if MGPS else f"sqrt(lambda_var)"

        # Lambda prior: Student-t (local shrinkage) or Normal (nu -> inf)
        if local_shrinkage_LatentFactors:
            if decenter_Lambda_transformed[d - 1]:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ student_t(nu, {scale_expr} * Lambda{d}_prior_means[, k], {scale_expr});")
            else:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ student_t(nu, 0, {scale_expr});")
        else:
            if decenter_Lambda_transformed[d - 1]:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ normal({scale_expr} * Lambda{d}_prior_means[, k], {scale_expr});")
            else:
                emit(f"  for (k in 1:K) Lambda{d}[, k] ~ normal(0, {scale_expr});")

        # theta prior given Lambda, eta, tau
        # emit("  {")
        # emit(f"    matrix[p{d}, p{d}] Prec_theta{d} = diag_matrix(tau{d});")
        # emit(f"    for (n in 1:N)")
        # emit(f"      theta{d}[:, n] ~ multi_normal_prec(Lambda{d} * eta[:, n], Prec_theta{d});")
        # emit("  }")
        emit("  {")
        if absence_of_psi2[d - 1]:
            emit(f"    matrix[p{d}, N] resid_theta{d} = T{d} - Lambda{d} * eta; // no theta{d} in this mode so we pass directly to T{d}")
        else:
            # emit(f"    matrix[p{d}, N] resid_theta{d} = theta{d} - Lambda{d} * eta;")
            pass

        if heteroscedastic_thetas[d - 1]:
            emit(f"    resid_theta{d} = diag_pre_multiply(sqrt(tau{d}), resid_theta{d});")
            emit(f"    to_vector(resid_theta{d}) ~ normal(0, 1);")
            emit(f"    target += 0.5 * sum(log(tau{d})) * N; // the terms that would have appeared in the theta likelihood if we had written with multi_normal_prec")
        else:
            emit(f"    to_vector(resid_theta{d}) ~ normal(0, sqrt(tau{d}));")
            # emit(f"    resid_theta{d} = resid_theta{d} / sqrt(tau{d});")
        emit("  }")

        # Theorem 3: IG kernel for psi2 and masked quadratic term
        if absence_of_psi2[d - 1]:
            emit(f"  // mode {d} has no psi2 parameter, nor the residual of T-theta, skipping its likelihood contribution")
        else:
            emit(f"  target += sum(-log(diagonal(Lchol{d}))); // log determinant term from integrating out theta{d}")
            emit(f"  target += inv_gamma_lpdf(psi2[{d}] | a_psi[{d}] + 0.5 * sum_Mdn{d},")
            emit(f"                                       b_psi[{d}] + 0.5 * sum_rss{d});")
            # emit("  {")
            # emit(f"    matrix[M{d}, N] E{d} = B{d} * (T{d} - theta{d});")
            # emit(f"    target += -0.5 / psi2[{d}] * dot_product(to_vector(Delta{d}), to_vector(E{d} .* E{d}));")
            # emit(f"    target += -0.5 / psi2[{d}] * dot_product(to_vector(Delta{d}), square(to_vector(E{d})));")
            # emit("  }")
    emit("}")
    emit("")
    emit("")
    emit("generated quantities {")
    # emit("  vector[D] ll_mode;")
    # for d in range(1, D + 1):
    #     emit("  {")
    #     emit(f"    matrix[M{d}, N] E{d} = B{d} * (T{d} - theta{d});")
    #     emit(f"    real quad{d} = dot_product(to_vector(Delta{d}), to_vector(E{d} .* E{d}));")
    #     emit(f"    real shp{d} = a_psi[{d}] + 0.5 * sum_Mdn{d};")
    #     emit(f"    real rte{d} = b_psi[{d}] + 0.5 * sum_rss{d};")
    #     emit(f"    ll_mode[{d}] = inv_gamma_lpdf(psi2[{d}] | shp{d}, rte{d}) - 0.5 / psi2[{d}] * quad{d};")
    #     emit("  }")
    emit("}")

    return "\n".join(lines)
### N.B. THIS FUNCTION IS WRONG! DON'T USE, IT'S LEFT FOR FUTURE DEVELOPMENT PURPOSES ONLY ###





import hashlib, json, os, sys, platform, pathlib, contextlib, time
from typing import Optional
from cmdstanpy import CmdStanModel, cmdstan_path, cmdstan_version
try:
    from platformdirs import user_cache_dir  # lightweight dep; recommend adding to install_requires
except Exception:
    # Fallback: simple per-user cache path
    def user_cache_dir(appname: str, appauthor: Optional[str] = None):
        root = os.path.expanduser("~")
        # Try OS-specific defaults
        if sys.platform.startswith("win"):
            base = os.getenv("LOCALAPPDATA") or os.path.join(root, "AppData", "Local")
        elif sys.platform == "darwin":
            base = os.path.join(root, "Library", "Caches")
        else:
            base = os.getenv("XDG_CACHE_HOME") or os.path.join(root, ".cache")
        return os.path.join(base, appname)

# Optional lock dependency
try:
    from filelock import FileLock
except Exception:
    FileLock = None

# ---- Configure your package name for cache dir
_PKG_CACHE_ROOT = user_cache_dir("MultivariateBLF")
pathlib.Path(_PKG_CACHE_ROOT).mkdir(parents=True, exist_ok=True)

def _stan_build_cache_key(stan_code: str,
                          stanc_options: Optional[dict] = None,
                          cpp_options: Optional[dict] = None) -> str:
    """
    Build a deterministic cache key that changes whenever anything that affects the
    resulting binary could change.
    """
    payload = {
        "stan_sha256": hashlib.sha256(stan_code.encode("utf-8")).hexdigest(),
        "cmdstan_version": str(cmdstan_version()),        # e.g. (2, 35)
        "cmdstan_path": str(cmdstan_path()),              # path can matter if toolchain layout differs
        "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "stanc_options": stanc_options or {},
        "cpp_options": cpp_options or {},
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()  # short, collision-resistant key


def _get_cached_model_dir(cache_key: str) -> str:
    d = os.path.join(_PKG_CACHE_ROOT, "stan_models", cache_key)
    os.makedirs(d, exist_ok=True)
    return d


def _get_or_compile_cmdstan_model(
    stan_code: str,
    stanfilename: str,
    stanc_options: Optional[dict] = None,
    cpp_options: Optional[dict] = None,
    quiet: bool = True,
) -> CmdStanModel:
    """
    Returns a CmdStanModel whose executable is cached on disk.
    If not already compiled for this exact (code+env) key, compiles once and stores it.
    """
    cache_key = _stan_build_cache_key(stan_code, stanc_options, cpp_options)
    cache_dir = _get_cached_model_dir(cache_key)
    stan_path = os.path.join(cache_dir, f"{stanfilename}.stan")
    exe_path = os.path.join(cache_dir, f"{stanfilename}")  # CmdStan adds .exe on Windows automatically

    # If the exe exists, just reuse it
    exe_exists = os.path.exists(exe_path) or os.path.exists(exe_path + (".exe" if sys.platform.startswith("win") else ""))
    if exe_exists:
        # Build a lightweight wrapper that points to the cached exe
        return CmdStanModel(stan_file=stan_path, exe_file=exe_path)

    # Prepare optional lock for multi-process safety
    lock_path = os.path.join(cache_dir, ".build.lock")
    if FileLock is not None:
        lock = FileLock(lock_path)
        ctx = lock
    else:
        # Fallback "null" context if filelock isn't available
        @contextlib.contextmanager
        def _noop_lock():
            yield
        ctx = _noop_lock()

    with ctx:
        # Double-check after acquiring lock (another proc may have compiled it)
        exe_exists = os.path.exists(exe_path) or os.path.exists(
            exe_path + (".exe" if sys.platform.startswith("win") else "")
        )
        if exe_exists:
            return CmdStanModel(stan_file=stan_path, exe_file=exe_path)

        # Write the stan source into the cache dir (so exe_file and stan_file are colocated)
        with open(stan_path, "w", encoding="utf-8") as f:
            f.write(stan_code)

        # Compile into the cache dir. CmdStanModel uses the stan_file location to place exe nearby.
        model = CmdStanModel(
            stan_file=stan_path,
            stanc_options=stanc_options or {},
            cpp_options=cpp_options or {},
        )
        # Compilation side-effects put the exe alongside the stan file
        # Validate it exists
        exe_exists = os.path.exists(exe_path) or os.path.exists(
            exe_path + (".exe" if sys.platform.startswith("win") else "")
        )
        if not exe_exists:
            # Rare edge case: some environments stash exe under a different name; wait a tick and recheck
            time.sleep(0.1)
            exe_exists = os.path.exists(exe_path) or os.path.exists(
                exe_path + (".exe" if sys.platform.startswith("win") else "")
            )
        if not exe_exists:
            raise RuntimeError("CmdStan compilation reported success but executable not found.")

    # Now return a model object that reuses cached exe (and won't try to recompile)
    return CmdStanModel(stan_file=stan_path, exe_file=exe_path)




if __name__ == "__main__":
    # store _assembleMBLF_stan_code_for_given_D() output in a file for inspection at ./stan/MBLF_D{D}.stan
    D = 2
    MGPS = True # enable/disable MGPS, we should correct that a few hyperparameters are unused when MGPS is off
    local_shrinkage_LatentFactors = True # enable/disable local shrinkage (Student-t vs Normal prior on Lambda) to still double-check
    heteroscedastic_thetas = [True, False]  # per-mode heteroscedasticity flags
    stan_code = _assembleMBLF_stan_code_for_given_D(
        D=D,
        MGPS=MGPS,
        local_shrinkage_LatentFactors=local_shrinkage_LatentFactors,
        heteroscedastic_thetas=heteroscedastic_thetas
    )
    out_dir = os.path.join(os.path.dirname(__file__), "stan")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"MBLF_D{D}_MGPS{MGPS}_local_shrinkage{local_shrinkage_LatentFactors}_heteroscedastic{heteroscedastic_thetas}.stan")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stan_code)