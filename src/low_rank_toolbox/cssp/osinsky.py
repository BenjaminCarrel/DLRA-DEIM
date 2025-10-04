import numpy as np
from numpy import ndarray
import scipy.linalg as la

def OCSS(U: ndarray, compute_M: bool = False) -> list:
    """
    Osinsky's OCSS algorithm for row selection via QR decomposition.

    Reference:
        "Close to optimal column approximations with a single SVD." by A.I. Osinsky, 2023.

    Parameters
    ----------
    U : numpy.ndarray
        Orthonormal real matrix of shape (n, r) defining a row space approximation
    compute_M : bool
        If True, return also the matrix U @ inv(U[S, :])

    Returns
    -------
    J : list
        Selected column indices
    M : numpy.ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    n, r = U.shape
    A = U.T.copy()
    P = np.arange(n) 
    l_scores = np.zeros(n)
    eps = np.finfo(np.float64).eps # Machine epsilon for float64

    for k in range(r):
        # --- Pivot Selection ---
        current_sub_A = A[k:r, k:n] 
        # Use **2 for real matrices instead of np.abs()**2
        norms_sq = np.sum(current_sub_A**2, axis=0) 
        current_l = l_scores[k:n]
        
        # Criterion: norms_sq / (1 + l_j), handle small norms/denominators
        denominators = 1.0 + current_l
        pivot_vals = np.zeros_like(denominators)
        # Check against epsilon^2 for squared norms? Or just eps? Use eps for norm check.
        valid_mask = norms_sq > eps 
        pivot_vals[valid_mask] = norms_sq[valid_mask] / denominators[valid_mask]
            
        best_idx_in_slice = np.argmax(pivot_vals)
        j_pivot = k + best_idx_in_slice

        # --- Swap ---
        if k != j_pivot:
            A[:, [k, j_pivot]] = A[:, [j_pivot, k]]
            P[k], P[j_pivot] = P[j_pivot], P[k]
            l_scores[k], l_scores[j_pivot] = l_scores[j_pivot], l_scores[k]
            
        # --- Householder Reflection ---
        x = A[k:r, k].copy()
        norm_x = np.linalg.norm(x)
        d_update = A[k, k:n].copy() # Initialize update scores from current row

        if norm_x > eps: 
            alpha = x[0]
            v = x 
            
            # Real Householder update: v[0] = x[0] + sign(x[0]) * ||x||
            # np.copysign handles sign(0) returning 1 (or -1 depending on impl)
            # If alpha is exactly 0, copysign(norm_x, 0.0) returns norm_x.
            v[0] = alpha + np.copysign(norm_x, alpha if alpha != 0 else 1.0)
            
            norm_v = np.linalg.norm(v)

            if norm_v > eps:
                v /= norm_v
                # Apply reflection: A_sub = A_sub - 2 * v * (v.T @ A_sub)
                sub_matrix_to_update = A[k:r, k:n]
                # Use v.T @ ... for dot product with real vector v
                vT_Asub = v.T.dot(sub_matrix_to_update)
                A[k:r, k:n] -= 2 * np.outer(v, vT_Asub)
                d_update = A[k, k:n].copy() # Get updated row slice

        # --- Update Scores ---
        # Use **2 for real scores update
        l_scores[k:n] += d_update**2 
    # --- End Loop ---

    if compute_M:
        M = la.solve(U[P[:r], :].T.conj(), U.T.conj()).T.conj()
        return P[:r], M
    return P[:r]
