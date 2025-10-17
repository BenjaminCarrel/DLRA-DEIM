import numpy as np
import scipy.linalg as la

def _householder_vector(x):
    norm_x = np.linalg.norm(x) # Full norm
    if norm_x < 1e-15:
        return np.zeros_like(x), 0.0 # Zero vector input

    # Complex Householder vector calculation
    # For complex numbers, use phase normalization instead of sign
    if x[0] == 0:
        alpha = norm_x
    else:
        alpha = (x[0] / np.abs(x[0])) * norm_x
    
    v = x.copy()
    v[0] += alpha

    # Normalize v using complex dot product
    v_norm_sq = np.vdot(v, v).real  # np.vdot handles complex conjugation
    if v_norm_sq < 1e-15:
        return np.zeros_like(x), 0.0 # Should not happen if norm_x > 0

    beta = 2.0 / v_norm_sq
    return v, beta

def _apply_householder_right(A, v, beta):
    v = v.reshape(-1, 1) # Ensure v is n x 1
    # Apply the Householder transformation
    Av = A.dot(v)
    A_updated = A - beta * (Av @ v.T.conj())
    return A_updated


def ARP(U: np.ndarray, compute_M: bool = False, seed: int=None):
    """
    Implements the Adaptive Randomized Pivoting (ARP) algorithm for Column Subset Selection Problem (CSSP).

    Reference: ADAPTIVE RANDOMIZED PIVOTING FOR COLUMN SUBSET SELECTION, DEIM, AND LOW-RANK APPROXIMATION by Cortinovis and Kressner. 

    Note: The algorithm is similar to Osinsky's algorithm for the CSSP. The randomization step allows for better error bounds (in expectation).
    (See Algorithm 2.1)
    
    Parameters
    ----------
    U: ndarray
        An (n x r) matrix with orthonormal columns.
    compute_M: bool
        If True, return also the matrix U @ inv(U[S, :])
    seed: int, optional
        Random seed for reproducibility.

    Returns
    -------
    J: list
        Selection of m column indices.
    M: ndarray (optional)
        Matrix U @ inv(U[S, :])
    """
    # Seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Adaptive Randomized Pivoting (ARP) algorithm
    n, r = U.shape
    J = []
    Uk = U.copy()

    for k in range(r):

        # Calculate squared norms of rows for the remaining columns
        row_norms_sq = np.sum(np.abs(Uk[:, k:r])**2, axis=1)
        
        # Set probabilities to zero for already selected indices
        for idx in J:
            row_norms_sq[idx] = 0.0
            
        total_norm_sq = np.sum(row_norms_sq)
        
        # Avoid division by zero
        if total_norm_sq < 1e-15:
            # If all remaining norms are negligible, pick remaining indices arbitrarily
            remaining_indices = [i for i in range(n) if i not in J]
            jk = remaining_indices[0] if remaining_indices else 0
        else:
            probs = row_norms_sq / total_norm_sq
            # Sample an index jk according to the probabilities
            jk = np.random.choice(n, p=probs)
            
        J.append(jk)

        # Householder reflection to zero out the jk-th row from column k onwards
        x = Uk[jk, k:r].copy()  # Extract the row vector to be zeroed out
        v, beta = _householder_vector(x)
        if beta != 0: # Apply only if reflection is non-trivial
            Uk[:, k:r] = _apply_householder_right(Uk[:, k:r], v, beta)

    if compute_M:
        M = la.lstsq(U[J, :].T.conj(), U.T.conj())[0].T.conj()
        return J, M
    else:
        return J