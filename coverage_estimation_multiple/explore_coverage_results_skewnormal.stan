data {
    int N;
    array[N] real<lower=0, upper=1> y;
}

transformed data {
    array[N] real z;
    for (n in 1:N) {
        z[n] = log(y[n] / 4) - log(1 - y[n]);  // logit transformation
    }

    int B = 1000;
}

parameters {
  real mu;                 // location
  real<lower=0> sigma;     // scale
  real alpha;              // skewness
}

model {
  // weakly informative priors
  mu ~ normal(0, 10);
  sigma ~ lognormal(0, 5);
  alpha ~ normal(0, 10);

  // likelihood
  z ~ skew_normal(mu, sigma, alpha);
}

generated quantities {
    array[B] real z_rep;
    array[B] real y_rep;
    real cdf_at_zero;
    for (b in 1:B) {
        z_rep[b] = skew_normal_rng(mu, sigma, alpha);
        y_rep[b] = 1 / (1 + exp(-z_rep[b]) / 4);
    }
    cdf_at_zero = skew_normal_cdf(0.0 | mu, sigma, alpha); // this is the probability of y < 0.8
    real avg_y_rep = mean(y_rep);
}