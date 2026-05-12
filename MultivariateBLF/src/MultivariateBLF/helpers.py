import numpy as np


# copied from sklearn
def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
    """Return rotated components."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix), rotation_matrix




class OALS_PCA:
    """
    Orthogonalized-Alternating Least Squares (O-ALS) for PCA with missing values.
    """

    def __init__(
        self,
        n_components,
        max_iter=500,
        tol_percent=1e-11,
        n_init=1,
        init_max_iter=35,
        center=False,
        random_state=None,
        verbose=False,
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol_percent = float(tol_percent)
        self.n_init = int(n_init)
        self.init_max_iter = int(init_max_iter)
        self.center = bool(center)
        self.random_state = random_state
        self.verbose = verbose

        # Learned quantities
        self.components_ = None              # P (C x N)
        self.scores_ = None                 # T (R x N)
        self.mean_ = None
        self.error_history_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    # ---------- Linear algebra helpers ----------

    @staticmethod
    def _gram_schmidt_scores(T):
        """Orthogonalize columns of T (scores) without normalizing."""
        T_ortho = T.copy()
        n_rows, n_cols = T_ortho.shape

        for k in range(n_cols):
            v = T_ortho[:, k]
            for j in range(k):
                u = T_ortho[:, j]
                denom = np.dot(u, u)
                if denom > 1e-15:
                    v = v - (np.dot(v, u) / denom) * u
            T_ortho[:, k] = v
        return T_ortho

    @staticmethod
    def _gram_schmidt_loadings(P):
        """
        Orthogonalize columns of P (loadings) and normalize each
        to unit norm, so P^T P ≈ I.
        """
        P_ortho = P.copy()
        n_rows, n_cols = P_ortho.shape

        for k in range(n_cols):
            v = P_ortho[:, k]
            for j in range(k):
                u = P_ortho[:, j]
                denom = np.dot(u, u)
                if denom > 1e-15:
                    v = v - (np.dot(v, u) / denom) * u
            norm = np.linalg.norm(v)
            if norm > 1e-15:
                v = v / norm
            else:
                v = np.zeros_like(v)
                v[k % n_rows] = 1.0
            P_ortho[:, k] = v

        return P_ortho

    @staticmethod
    def _reconstruction_error(Dm, T, P, mask):
        """Squared Frobenius error on available entries."""
        R = np.dot(T, P.T)
        diff = Dm - R
        diff = diff[mask]
        return np.dot(diff, diff)

    def _als_loop(self, Dm, mask, P_init, max_iter, tol_percent=None):
        """
        Core O-ALS iterations for a single initialization.
        """
        R, C = Dm.shape
        N = self.n_components

        P = self._gram_schmidt_loadings(P_init)
        T = np.zeros((R, N), dtype=float)

        errors = []
        prev_err = None

        for it in range(max_iter):
            # ----- Step 2A: update scores T row-wise -----
            for i in range(R):
                valid = mask[i, :]
                if not np.any(valid):
                    continue
                d_i = Dm[i, valid]
                P_sub = P[valid, :]
                t_i, _, _, _ = np.linalg.lstsq(P_sub, d_i, rcond=None)
                T[i, :] = t_i

            T = self._gram_schmidt_scores(T)

            # ----- Step 2B: update loadings P column-wise -----
            for j in range(C):
                valid = mask[:, j]
                if not np.any(valid):
                    continue
                d_j = Dm[valid, j]
                T_sub = T[valid, :]
                p_j, _, _, _ = np.linalg.lstsq(T_sub, d_j, rcond=None)
                P[j, :] = p_j

            P = self._gram_schmidt_loadings(P)

            # ----- Convergence check using reconstruction error -----
            err = self._reconstruction_error(Dm, T, P, mask)
            errors.append(err)

            if tol_percent is not None and prev_err is not None:
                rel_change_percent = abs(err - prev_err) / (prev_err + 1e-20) * 100.0
                if rel_change_percent < tol_percent:
                    if self.verbose:
                        print(
                            f"Converged after {it+1} iterations "
                            f"(Δerr = {rel_change_percent:.2e} %)."
                        )
                    break

            prev_err = err

        return T, P, errors

    # ---------- Public API ----------

    def fit(self, X):
        """
        Fit the O-ALS PCA model to a data matrix with missing values (NaNs).
        """
        X = np.asarray(X, dtype=float)
        R, C = X.shape
        N = self.n_components

        mask = np.isfinite(X)

        # Center (optional)
        if self.center:
            self.mean_ = np.zeros(C, dtype=float)
            for j in range(C):
                col = X[:, j]
                valid = np.isfinite(col)
                if np.any(valid):
                    self.mean_[j] = np.mean(col[valid])
                else:
                    self.mean_[j] = 0.0
            Dm = X - self.mean_
        else:
            self.mean_ = np.zeros(C, dtype=float)
            Dm = X.copy()

        rng = np.random.RandomState(self.random_state)

        best_P = None
        best_T = None
        best_err = np.inf

        # ---- multiple random initializations (if n_init > 1) ----
        for init_idx in range(self.n_init):
            P_init = rng.randn(C, N)

            T_tmp, P_tmp, errors_tmp = self._als_loop(
                Dm, mask, P_init, max_iter=self.init_max_iter, tol_percent=None
            )
            err_last = errors_tmp[-1]

            if self.verbose:
                print(
                    f"[init {init_idx+1}/{self.n_init}] "
                    f"error after {self.init_max_iter} iters = {err_last:.6e}"
                )

            if err_last < best_err:
                best_err = err_last
                best_P = P_tmp.copy()
                best_T = T_tmp.copy()

            if self.n_init == 1:
                break

        # ---- full run from best initialization ----
        T_final, P_final, errors_full = self._als_loop(
            Dm,
            mask,
            best_P,
            max_iter=self.max_iter,
            tol_percent=self.tol_percent,
        )

        # ---------- explained variance from T, sort components ----------
        n_samples = T_final.shape[0]
        denom = max(n_samples - 1, 1)

        # eigenvalue-like quantity for each component
        explained_variance = np.sum(T_final**2, axis=0) / denom
        total_model_variance = explained_variance.sum() + 1e-20
        explained_ratio = explained_variance / total_model_variance

        # sort by decreasing explained variance
        order = np.argsort(explained_variance)[::-1]
        T_final = T_final[:, order]
        P_final = P_final[:, order]
        explained_variance = explained_variance[order]
        explained_ratio = explained_ratio[order]

        # store
        self.components_ = P_final
        self.scores_ = T_final
        self.error_history_ = errors_full
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_ratio

        return self

    def transform(self, X):
        """
        Project new data into the learned PCA space using the
        current (sorted) components_.
        """
        if self.components_ is None:
            raise RuntimeError("The model must be fitted before calling transform().")

        X = np.asarray(X, dtype=float)
        R, C = X.shape
        N = self.n_components

        if C != self.components_.shape[0]:
            raise ValueError("X has a different number of features than the fitted model.")

        if self.center:
            Xc = X - self.mean_
        else:
            Xc = X.copy()

        mask = ~np.isnan(Xc)
        T_new = np.zeros((R, N), dtype=float)
        P = self.components_

        for i in range(R):
            valid = mask[i, :]
            if not np.any(valid):
                continue
            d_i = Xc[i, valid]
            P_sub = P[valid, :]
            t_i, _, _, _ = np.linalg.lstsq(P_sub, d_i, rcond=None)
            T_new[i, :] = t_i

        T_new = self._gram_schmidt_scores(T_new)
        return T_new

    def inverse_transform(self, T):
        """
        Reconstruct data from scores using X_hat = T P^T (+ mean if centered).
        """
        if self.components_ is None:
            raise RuntimeError("The model must be fitted before calling inverse_transform().")

        T = np.asarray(T, dtype=float)
        X_rec = np.dot(T, self.components_.T)
        if self.center:
            X_rec = X_rec + self.mean_
        return X_rec

