
import numpy as np
import scipy.spatial.distance as distance


def quad_cov_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    dists = distance.cdist(X1 / lengthscale, X2 / lengthscale, 'sqeuclidean')
    return variance * np.exp(-0.5 * dists)


def matern52_cov_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    dists = distance.cdist(X1 / lengthscale, X2 / lengthscale, 'euclidean')
    sqrt5_dists = np.sqrt(5) * dists
    return variance * (1 + sqrt5_dists + (5.0 / 3.0) * dists**2) * np.exp(-sqrt5_dists)


def formula_eval(formula, X1, X2):
    """
    In the future it will deal with formula-based kernel evaluation, of the type:
    "quad_cov_kernel(1.0, 1.0) + nugget_kernel(0.1)"
    """
    raise NotImplementedError("This function is a placeholder for formula-based kernel evaluation.")


