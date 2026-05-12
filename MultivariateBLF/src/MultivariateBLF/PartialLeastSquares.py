from typing import List, Union, Optional
import numpy as np
from scipy.linalg import sqrtm, inv
    
from sklearn.cross_decomposition import PLSRegression
from .MultivariateBLF import FunctionalData

from dataclasses import dataclass
from tqdm.auto import tqdm
import copy


class PartialLeastSquares:
    """
    Standard Partial Least Squares (PLS) regression for discretized functional data.

    This class treats the functional data as multivariate vectors, ignoring the 
    functional smoothness or basis expansions. It performs PLS directly on the 
    observed values at the grid points.

    Parameters
    ----------
    n_components : int, optional
        Number of PLS components to use. Default is 2.
    scale : bool, optional
        Whether to scale the data (zero mean, unit variance) before fitting. 
        Default is True.
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.model = PLSRegression(n_components=self.n_components, scale=self.scale)
        self._Y_coords = None # To reconstruct the output FunctionalData
        self._fitted = False

    def _process_input(self, fd_list: List[FunctionalData]) -> np.ndarray:
        """
        Helper to convert a list of FunctionalData objects into a single 
        scikit-learn compatible matrix (N_samples x M_features).
        """
        blocks = []
        for fd in fd_list:
            # FunctionalData.data is (M, N).
            # sklearn expects (N, M).
            data = fd.data.T
            
            # PLS in sklearn does not handle NaNs. 
            # We perform a naive imputation (0.0) if NaNs exist, 
            # matching the logic seen in FunctionalData.empirical_basis_coefficients.
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=0.0)
                
            blocks.append(data)
        
        # Stack features horizontally: Shape (N, sum(M_i))
        return np.hstack(blocks)

    def fit(self, X: Union[FunctionalData, List[FunctionalData]], Y: FunctionalData):
        """
        Fit the PLS model to the functional data vectors.

        Parameters
        ----------
        X : FunctionalData or List[FunctionalData]
            The predictor functional data.
        Y : FunctionalData
            The response functional data.
        """
        # Store Y domain for prediction reconstruction
        self._Y_coords = Y.coords

        # 1. Process X
        if isinstance(X, FunctionalData):
            X_list = [X]
        else:
            X_list = X
        
        X_mat = self._process_input(X_list)

        # 2. Process Y
        # Y must be a single FunctionalData object for standard PLS in this context
        Y_mat = self._process_input([Y])

        # Check sample alignment
        if X_mat.shape[0] != Y_mat.shape[0]:
            raise ValueError(f"X has {X_mat.shape[0]} samples, but Y has {Y_mat.shape[0]}.")

        # 3. Fit Model
        self.model.fit(X_mat, Y_mat)
        self._fitted = True

    def predict(self, X: Union[FunctionalData, List[FunctionalData]]) -> FunctionalData:
        """
        Predict using the fitted PLS model.

        Parameters
        ----------
        X : FunctionalData or List[FunctionalData]
            The predictor functional data. Must have the same grid size (M) 
            as the training data.

        Returns
        -------
        FunctionalData
            The predicted response functional data.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if isinstance(X, FunctionalData):
            X_list = [X]
        else:
            X_list = X

        # 1. Prepare Input
        X_mat = self._process_input(X_list)

        # 2. Predict
        # Output is (N_samples, M_y_features)
        Y_pred_mat = self.model.predict(X_mat)

        # 3. Reconstruct FunctionalData
        # We need to transpose back to (M, N) for FunctionalData structure
        # We use the coords stored during fit.
        return FunctionalData(
            coords=self._Y_coords,
            data=Y_pred_mat.T
        )

class FunctionalPartialLeastSquares:
    """
    Functional Partial Least Squares (FPLS) regression using basis expansions.

    This implementation follows the methodology where the functional regression 
    problem is reduced to a multivariate PLS problem operating on the 
    basis coefficients, corrected by the inner product matrices (Gram matrices)
    of the basis functions.

    Reference: 
    Beyaztas, U., & Shang, H. L. (2019). On function-on-function regression: 
    Partial least squares approach.

    Parameters
    ----------
    n_components : int, optional
        Number of PLS components to use. Default is 2.
    scale : bool, optional
        Whether to scale the transformed coefficients inside the PLS algorithm. 
        Default is True.
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.model = PLSRegression(n_components=self.n_components, scale=self.scale)
        
        self._Wy_inv = None  # Inverse of Phi^(1/2)
        self._Wx_list = []   # List of Psi^(1/2) matrices
        self._Y_basis = None # Basis matrix of Y
        self._Y_coords = None # Domain of Y
        self._fitted = False

    def _compute_gram_sqrt(self, basis_matrix: np.ndarray) -> np.ndarray:
        """
        Computes W = (B^T B)^(1/2).
        """
        # Gram matrix approximation via discrete dot product
        G = basis_matrix.T @ basis_matrix
        
        # Matrix square root
        W = sqrtm(G)
        
        # Ensure real output
        if np.iscomplexobj(W):
            W = W.real
        return W

    def fit(self, X: Union[FunctionalData, List[FunctionalData]], Y: FunctionalData):
        """
        Fit the FPLS model.

        Parameters
        ----------
        X : FunctionalData or List[FunctionalData]
            The functional predictors (source modes).
        Y : FunctionalData
            The functional response (target mode).
        """
        if not hasattr(Y, '_basis_matrix') or Y._basis_matrix is None:
             raise ValueError("Y must have a basis set (use set_raw_basis, set_GP_basis, etc).")

        self._Y_coords = Y.coords
        
        # 1. Transform Response Y
        # C is (p_y x N), we need (N x p_y)
        C = Y.empirical_basis_coefficients.T
        B_y = Y._basis_matrix
        self._Y_basis = B_y 

        # W_y = (B_y^T B_y)^(1/2)
        W_y = self._compute_gram_sqrt(B_y)
        
        # Store W_y^(-1) for back-transformation
        self._Wy_inv = inv(W_y)

        # Z_y = C * W_y
        Z_y = C @ W_y

        # 2. Transform Predictors X
        if isinstance(X, FunctionalData):
            X_list = [X]
        else:
            X_list = X
        
        self._Wx_list = []
        Z_x_blocks = []

        for i, fd in enumerate(X_list):
            if not hasattr(fd, '_basis_matrix') or fd._basis_matrix is None:
                raise ValueError(f"Predictor mode {i} must have a basis set.")
            
            # D is (p_x x N) -> (N x p_x)
            D = fd.empirical_basis_coefficients.T
            B_x = fd._basis_matrix
            
            # W_x = (B_x^T B_x)^(1/2)
            W_x = self._compute_gram_sqrt(B_x)
            self._Wx_list.append(W_x)
            
            # Z_x = D * W_x
            Z_x_k = D @ W_x
            Z_x_blocks.append(Z_x_k)

        # Stack latent predictor blocks: (N, p_x_total)
        Z_x = np.hstack(Z_x_blocks)

        if Z_x.shape[0] != Z_y.shape[0]:
            raise ValueError(f"Sample size mismatch: X has {Z_x.shape[0]}, Y has {Z_y.shape[0]}.")

        # 3. Fit PLS
        self.model.fit(Z_x, Z_y)
        self._fitted = True

    def predict(self, X: Union[FunctionalData, List[FunctionalData]]) -> FunctionalData:
        """
        Predict functional response.

        Parameters
        ----------
        X : FunctionalData or List[FunctionalData]
            The predictor functional data.

        Returns
        -------
        FunctionalData
            Predicted response.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        if isinstance(X, FunctionalData):
            X_list = [X]
        else:
            X_list = X

        if len(X_list) != len(self._Wx_list):
            raise ValueError(f"Expected {len(self._Wx_list)} predictor modes, got {len(X_list)}.")

        # 1. Transform Inputs
        Z_x_blocks = []
        for i, fd in enumerate(X_list):
            D_new = fd.empirical_basis_coefficients.T
            W_x = self._Wx_list[i]
            
            if D_new.shape[1] != W_x.shape[0]:
                 raise ValueError(f"Predictor {i} basis dimension mismatch.")

            Z_x_k = D_new @ W_x
            Z_x_blocks.append(Z_x_k)

        Z_x_new = np.hstack(Z_x_blocks)

        # 2. PLS Prediction
        Z_y_pred = self.model.predict(Z_x_new)

        # 3. Back-transform to Coefficients
        # C_hat = Z_y_pred * W_y^(-1)
        C_hat = Z_y_pred @ self._Wy_inv

        # 4. Reconstruct Functions
        # Y_hat = B_y * C_hat^T
        # (M_y x p_y) @ (p_y x N) -> (M_y x N)
        Y_pred_matrix = self._Y_basis @ C_hat.T

        return FunctionalData(
            coords=self._Y_coords,
            data=Y_pred_matrix
        )
    


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    mean_pred: FunctionalData
    lower_ci: FunctionalData
    upper_ci: FunctionalData
    all_preds_stack: np.ndarray  # Shape (B, M, N_test)
    mean_x_weights: np.ndarray 
    all_x_weights_stack: np.ndarray
    mean_y_weights: np.ndarray
    all_y_weights_stack: np.ndarray


def slice_functional_data(fd: FunctionalData, indices: np.ndarray) -> FunctionalData:
    """
    Helper to slice a FunctionalData object by sample indices (columns).
    Crucially, this preserves the basis matrix if it has been computed,
    avoiding the need to re-compute expensive kernels/expansions.
    """
    # Create new instance with sliced data
    # fd.data is (M, N), so we slice the second dimension
    sliced_data = fd.data[:, indices]
    
    new_fd = FunctionalData(coords=fd.coords, data=sliced_data)
    
    # Transfer the basis matrix if it exists (Critical for FPLS)
    if hasattr(fd, '_basis_matrix') and fd._basis_matrix is not None:
        new_fd.set_raw_basis(fd._basis_matrix)
        
    return new_fd

def bootstrap_prediction_uncertainty(
    model_class: type,
    model_params: dict,
    X_train: Union[FunctionalData, List[FunctionalData]],
    Y_train: FunctionalData,
    X_test: Union[FunctionalData, List[FunctionalData]],
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> BootstrapResult:
    """
    Performs nonparametric bootstrap (resampling subjects with replacement)
    to estimate prediction uncertainty.

    Parameters
    ----------
    model_class : class
        The class to instantiate (PartialLeastSquares or FunctionalPartialLeastSquares).
    model_params : dict
        Dictionary of parameters to pass to model_class __init__ (e.g., {'n_components': 3}).
    X_train : FunctionalData or List[FunctionalData]
        Training predictors.
    Y_train : FunctionalData
        Training response.
    X_test : FunctionalData or List[FunctionalData]
        Test predictors to evaluate uncertainty on.
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level for Confidence Intervals (e.g., 0.05 for 95% CI).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    BootstrapResult object containing FunctionalData objects for mean, CI bounds, etc.
    """
    rng = np.random.default_rng(random_state)
    
    from MultivariateBLF.MultivariateBLF import FunctionalData
    from MultivariateBLF.PartialLeastSquares import slice_functional_data
    
    if isinstance(X_train, FunctionalData): X_train = [X_train]; X_test = [X_test]
    
    N_samples = Y_train.data.shape[1]
    M_points_Y = Y_train.data.shape[0]
    N_test = X_test[0].data.shape[1]
    M_points_X_total = sum([fd.data.shape[0] for fd in X_train])
    K_components = model_params.get('n_components', 2)

    boot_preds = np.zeros((n_boot, M_points_Y, N_test))
    boot_x_weights = np.zeros((n_boot, M_points_X_total, K_components))
    boot_y_weights = np.zeros((n_boot, M_points_Y, K_components))

    iterator = range(n_boot)
    if verbose:
        iterator = tqdm(iterator, desc=f"Bootstrap {model_class.__name__}")

    for b in iterator:
        # Resample
        resample_idx = rng.choice(N_samples, size=N_samples, replace=True)
        X_train_boot = [slice_functional_data(fd, resample_idx) for fd in X_train]
        Y_train_boot = slice_functional_data(Y_train, resample_idx)
        
        model = model_class(**model_params)
        try:
            model.fit(X_train_boot, Y_train_boot)
            
            # PI Estimation
            Y_hat_train = model.predict(X_train_boot)
            residuals = Y_train_boot.data - Y_hat_train.data
            sigma_noise = np.nanstd(residuals) 
            
            pred_fd = model.predict(X_test)
            noise_matrix = rng.normal(loc=0.0, scale=sigma_noise, size=pred_fd.data.shape)
            boot_preds[b, :, :] = pred_fd.data + noise_matrix
            
            # Weights Extraction
            raw_x_weights = model.model.x_weights_ 
            raw_y_weights = model.model.y_weights_

            if "Functional" in model_class.__name__:
                # --- Project X Weights ---
                current_row = 0
                w_coef_start = 0
                for i, fd in enumerate(X_train):
                    basis = fd._basis_matrix 
                    gram_sqrt = model._Wx_list[i]
                    p_dim = gram_sqrt.shape[0]
                    m_dim = basis.shape[0]
                    
                    w_part_whitened = raw_x_weights[w_coef_start : w_coef_start + p_dim, :]
                    w_part_coef = gram_sqrt @ w_part_whitened
                    w_part_domain = basis @ w_part_coef
                    
                    boot_x_weights[b, current_row : current_row + m_dim, :] = w_part_domain
                    
                    current_row += m_dim
                    w_coef_start += p_dim

                # --- Project Y Weights ---
                B_y = model._Y_basis
                W_y = model._compute_gram_sqrt(B_y)
                
                w_y_coef = W_y @ raw_y_weights
                w_y_domain = B_y @ w_y_coef
                boot_y_weights[b, :, :] = w_y_domain

            else:
                boot_x_weights[b, :, :] = model.model.x_loadings_
                boot_y_weights[b, :, :] = model.model.y_loadings_
                
        except Exception as e:
            boot_preds[b, :, :] = np.nan
            boot_x_weights[b, :, :] = np.nan
            boot_y_weights[b, :, :] = np.nan

    # Statistics
    mean_pred = np.nanmean(boot_preds, axis=0)
    lower_pi = np.nanpercentile(boot_preds, 100*(alpha/2), axis=0)
    upper_pi = np.nanpercentile(boot_preds, 100*(1-alpha/2), axis=0)
    
    mean_x_weights = np.nanmean(boot_x_weights, axis=0)
    mean_y_weights = np.nanmean(boot_y_weights, axis=0)

    return BootstrapResult(
        mean_pred=FunctionalData(Y_train.coords, mean_pred),
        lower_ci=FunctionalData(Y_train.coords, lower_pi),
        upper_ci=FunctionalData(Y_train.coords, upper_pi),
        mean_x_weights=mean_x_weights,
        all_x_weights_stack=boot_x_weights,
        mean_y_weights=mean_y_weights,
        all_y_weights_stack=boot_y_weights,
        all_preds_stack=boot_preds
    )

