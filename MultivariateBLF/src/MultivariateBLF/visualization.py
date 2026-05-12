import numpy as np
import xarray as xr
from typing import List, Optional, Dict, Any
from scipy.optimize import linear_sum_assignment

from .MultivariateBLF import MvtBLF
from .helpers import _ortho_rotation


def align_true_factors_to_varimaxrsp(
    model: MvtBLF,
    Lambda_true_list: List[np.ndarray],
    B_true_list: Optional[List[np.ndarray]] = None,
    method: str = "varimax",
) -> Dict[str, Any]:
    """
    Align *true* factor loadings (from a simulated dataset) to the
    Varimax-RSP orientation used by a fitted `MvtBLF` model.

    Steps
    -----
    1. Stack true Lambda_d over modes (p_total x K).
    2. Apply orthogonal rotation (by default Varimax) to the stacked matrix.
    3. Take the posterior Varimax-RSP loadings from `model.computeVarimaxRSP`
       and compute their mean over (chain, draw) as the reference orientation.
    4. Find the best signed permutation mapping from true (varimaxed)
       factors to the reference factors using the Hungarian algorithm.
    5. Apply this signed permutation to the true (varimaxed) Lambda and
       optionally to BΛ, returning aligned quantities per mode.

    Parameters
    ----------
    model : MvtBLF
        Fitted model with `computeVarimaxRSP` available.
    Lambda_true_list : list of np.ndarray
        True loadings per mode, each of shape (p_d, K). Order of modes must
        match the order used in `model.LambdaStacked` (i.e., mode 1 rows,
        then mode 2, etc.).
    B_true_list : list of np.ndarray, optional
        True basis matrices per mode, each of shape (M_d, p_d). If None,
        uses the basis matrices stored inside `model._modes[d]._basis_matrix`.
    method : {"varimax", "quartimax"}, default "varimax"
        Orthogonal rotation method passed to `_ortho_rotation`.

    Returns
    -------
    out : dict
        {
          "Lambda_true_varimax_stacked": (p_total, K) np.ndarray,
          "Lambda_true_aligned_stacked": (p_total, K) np.ndarray,
          "Lambda_true_aligned_per_mode": list of (p_d, K) arrays,
          "BLambda_true_aligned_per_mode": list of (M_d, K) arrays or None,
          "perm": np.ndarray of shape (K,),  # source index for each canonical factor
          "signs": np.ndarray of shape (K,), # ±1 for each canonical factor
        }

    Notes
    -----
    - This is meant for simulation studies: you know the true Lambda used
      to generate the data (e.g. from prior predictive draws).
    - After calling this, you can overlay the aligned true BΛ curves on
      top of `model.plot_BLambda_d_VarimaxRSP(...)`.
    """
    D = model.D
    K = model.n_components

    if len(Lambda_true_list) != D:
        raise ValueError(f"expected {D} true Lambda matrices, got {len(Lambda_true_list)}")

    # Normalize Lambda_true_list and collect p_d
    Lambda_true_list = [np.asarray(Ld, dtype=float) for Ld in Lambda_true_list]
    p_list = []
    for d, Ld in enumerate(Lambda_true_list):
        if Ld.ndim != 2 or Ld.shape[1] != K:
            raise ValueError(
                f"Lambda_true_list[{d}] must have shape (p_d, K={K}), "
                f"got {Ld.shape}"
            )
        p_list.append(Ld.shape[0])
    p_total = int(sum(p_list))

    # Basis matrices: if not given, take from fitted model
    if B_true_list is None:
        if model._modes is None:
            raise ValueError(
                "B_true_list is None and model._modes is not set. "
                "Pass B_true_list explicitly or fit the model first."
            )
        B_true_list = [np.asarray(m._basis_matrix, dtype=float) for m in model._modes]
    else:
        if len(B_true_list) != D:
            raise ValueError(f"expected {D} basis matrices, got {len(B_true_list)}")
        B_true_list = [np.asarray(B, dtype=float) for B in B_true_list]

    # 1) Stack true Lambda over modes: (p_total, K)
    Lambda_true_stacked = np.concatenate(Lambda_true_list, axis=0)  # (p_total, K)

    # 2) Varimax (or other orthogonal) rotation of the true stacked Lambda
    if method is not None:
        Lambda_true_varimax_stacked, R_true = _ortho_rotation(
            Lambda_true_stacked, method=method
        )
    else:
        Lambda_true_varimax_stacked = Lambda_true_stacked.copy()
        R_true = np.eye(K)

    # 3) Posterior Varimax-RSP reference from fitted model
    lambdas_rsp_xr, _, _, _ = model.computeVarimaxRSP
    # lambdas_rsp_xr dims: ("chain", "draw", "p", "K")
    Lambda_post_ref = lambdas_rsp_xr.mean(dim=("chain", "draw")).values  # (p_total, K)

    if Lambda_post_ref.shape != (p_total, K):
        raise RuntimeError(
            f"Shape mismatch between true stacked Lambda (p_total={p_total}, K={K}) "
            f"and posterior reference {Lambda_post_ref.shape}"
        )

    # 4) Signed permutation alignment: true_varimax -> posterior reference
    A = Lambda_post_ref              # reference (p_total x K)
    B = Lambda_true_varimax_stacked  # candidate (p_total x K)

    A_norm_sq = np.sum(A ** 2, axis=0)  # (K,)
    B_norm_sq = np.sum(B ** 2, axis=0)  # (K,)

    cost = np.empty((K, K), dtype=float)
    sign_matrix = np.empty((K, K), dtype=float)

    for i in range(K):
        a = A[:, i]
        for j in range(K):
            b = B[:, j]
            dot = float(np.dot(a, b))
            s = 1.0 if dot >= 0.0 else -1.0
            sign_matrix[i, j] = s
            # squared-error with optimal sign = ||a||^2 + ||b||^2 - 2|a^T b|
            cost[i, j] = A_norm_sq[i] + B_norm_sq[j] - 2.0 * abs(dot)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Build aligned true Lambda (stacked)
    Lambda_true_aligned_stacked = np.zeros_like(B)
    signs = np.zeros(K, dtype=float)
    perm = np.zeros(K, dtype=int)

    for idx in range(len(row_ind)):
        i = int(row_ind[idx])   # canonical factor index (target)
        j = int(col_ind[idx])   # source factor index (true varimax)
        s = float(sign_matrix[i, j])
        Lambda_true_aligned_stacked[:, i] = s * B[:, j]
        signs[i] = s
        perm[i] = j

    # 5) Split aligned Lambda back into per-mode pieces and compute BΛ
    Lambda_true_aligned_per_mode: List[np.ndarray] = []
    BLambda_true_aligned_per_mode: List[np.ndarray] = []

    start = 0
    for d in range(D):
        p_d = p_list[d]
        end = start + p_d

        Lambda_d_aligned = Lambda_true_aligned_stacked[start:end, :]  # (p_d, K)
        Lambda_true_aligned_per_mode.append(Lambda_d_aligned)

        B_d = B_true_list[d]  # (M_d, p_d)
        BLambda_d_aligned = B_d @ Lambda_d_aligned                    # (M_d, K)
        BLambda_true_aligned_per_mode.append(BLambda_d_aligned)

        start = end

    out: Dict[str, Any] = {
        "Lambda_true_varimax_stacked": Lambda_true_varimax_stacked,
        "Lambda_true_aligned_stacked": Lambda_true_aligned_stacked,
        "Lambda_true_aligned_per_mode": Lambda_true_aligned_per_mode,
        "BLambda_true_aligned_per_mode": BLambda_true_aligned_per_mode,
        "perm": perm,   # for each canonical factor i, original varimax index perm[i]
        "signs": signs, # sign applied to each canonical factor column
        "R_true": R_true,
    }
    return out
