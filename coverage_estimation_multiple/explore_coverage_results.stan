data {
    int N;
    array[N] real<lower=0, upper=1> y;
}


parameters {
    real<lower=0, upper=1> mu;
    real<lower=0> K;
    array[N] real<lower=2> g;
}

model {
    for (n in 1:N) {
        y[n] ~ beta(mu * K * g[n], (1 - mu) * K * g[n]);
        g[n] ~ gamma(2, 1.33);
    }

    mu ~ beta(1.0, 1.0);
    K ~ gamma(3, 0.2);

}