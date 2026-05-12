  // Auto-generated Stan code (Theorem 3 likelihood, Section 4.1)
data {
  int<lower=1> N; // Number of observations
  int<lower=1> D; // Number of functional modes
  int<lower=1> K; // Number of latent factors
  int<lower=1> L; // Number of predictors

  matrix[L, N] X; // Predictor matrix

  // Data for functional mode 1
  int<lower=1> p1; // Number of basis functions for mode 1
  int<lower=1> M1; // Number of locations (union grid) for mode 1
  matrix[M1, p1] B1; // Basis matrix
  matrix[M1, N] Delta1; // 1=observed, 0=missing at each grid loc/subject
  matrix[p1, N] T1; // Compressed coefficients per Theorem 3
  real<lower=0> sum_rss1; // Sum_n ||R_n^1||^2
  int<lower=0> sum_Mdn1; // Sum_n M_(d,n) (observed counts)

  // Data for functional mode 2
  int<lower=1> p2; // Number of basis functions for mode 2
  int<lower=1> M2; // Number of locations (union grid) for mode 2
  matrix[M2, p2] B2; // Basis matrix
  matrix[M2, N] Delta2; // 1=observed, 0=missing at each grid loc/subject
  matrix[p2, N] T2; // Compressed coefficients per Theorem 3
  real<lower=0> sum_rss2; // Sum_n ||R_n^2||^2
  int<lower=0> sum_Mdn2; // Sum_n M_(d,n) (observed counts)

  // Hyperparameters
  // measurement error variance priors
  vector<lower=0>[D] a_psi;
  vector<lower=0>[D] b_psi;
  // basis coefficients precisions priors
  vector<lower=0>[D] a_tau;
  vector<lower=0>[D] b_tau;
  // factor loadings priors (MGPS controls if deltas/omegas are active)
  real<lower=0> a1; // alpha1 for MGPS (delta_1)
  real<lower=0> a2; // alpha2 for MGPS (delta_k>=2)
  real<lower=0> b_lambdavar; // rate parameter for the precision of the MGPS omegas
  real<lower=0> nu; // df for Student-t prior on Lambda
  // regression coefficients prior
  real<lower=0> sigma_regr_coeffs;
}


parameters {
  matrix[K, L] regr_coeffs; // regression coefficients
  matrix[K, N] eta; // scores of latent factors

  // Parameters for functional mode 1
  matrix[p1, K] Lambda1; // factor loadings for mode 1
  vector<lower=0>[p1] tau1; // basis-coeff precisions for mode 1 - heteroscedastic
  matrix[p1, N] theta1; // latent basis coefficients for mode 1

  // Parameters for functional mode 2
  matrix[p2, K] Lambda2; // factor loadings for mode 2
  real<lower=0> tau2; // basis-coeff precisions for mode 2 - homoscedastic
  matrix[p2, N] theta2; // latent basis coefficients for mode 2
  vector<lower=0>[K] delta; // MGPS global multipliers

  vector<lower=0>[D] psi2; // measurement error variances per mode
}


transformed parameters {
  vector<lower=0>[K] omega;
  omega[1] = b_lambdavar * [1];
  for (k in 2:K) omega[k] = omega[k-1] * delta[k];

}


model {
  // Priors for regression and scores
  to_vector(regr_coeffs) ~ normal(0, sigma_regr_coeffs);
  to_vector(eta) ~ normal(to_vector(regr_coeffs * X), 1);

  // MGPS priors
  delta[1] ~ gamma(a1, 1);
  for (k in 2:K) delta[k] ~ gamma(a2, 1);


  // Mode 1: priors for tau, Lambda, theta; and Theorem 3 likelihood
  // heteroscedastic basis coefficients theta
  tau1 ~ gamma(a_tau[1], b_tau[1]);
  for (k in 1:K) Lambda1[, k] ~ student_t(nu, 0, inv_sqrt(omega[k]));
  {
    matrix[p1, N] resid_theta1 = theta1 - Lambda1 * eta;
    resid_theta1 = diag_pre_multiply(inv_sqrt(tau1), resid_theta1);
    to_vector(resid_theta1) ~ normal(0, 1);
    target += 0.5 * sum(log(tau1)) * N; // the terms that would have appeared in the theta likelihood if we had written with multi_normal_prec
  }
  target += inv_gamma_lpdf(psi2[1] | a_psi[1] + 0.5 * sum_Mdn1,
                                       b_psi[1] + 0.5 * sum_rss1);
  {
    matrix[M1, N] E1 = B1 * (T1 - theta1);
    target += -0.5 / psi2[1] * dot_product(to_vector(Delta1), square(to_vector(E1)));
  }

  // Mode 2: priors for tau, Lambda, theta; and Theorem 3 likelihood
  //  homoscedastic  basis coefficients theta
  tau2 ~ gamma(a_tau[2], b_tau[2]);
  for (k in 1:K) Lambda2[, k] ~ student_t(nu, 0, inv_sqrt(omega[k]));
  {
    matrix[p2, N] resid_theta2 = theta2 - Lambda2 * eta;
    to_vector(resid_theta2) ~ normal(0, sqrt(tau2));
  }
  target += inv_gamma_lpdf(psi2[2] | a_psi[2] + 0.5 * sum_Mdn2,
                                       b_psi[2] + 0.5 * sum_rss2);
  {
    matrix[M2, N] E2 = B2 * (T2 - theta2);
    target += -0.5 / psi2[2] * dot_product(to_vector(Delta2), square(to_vector(E2)));
  }
}


generated quantities {
}