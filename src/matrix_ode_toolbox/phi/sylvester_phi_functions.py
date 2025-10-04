"""
Phi functions for a Sylvester operator S(X) = AX + XB^T.
"""

# %% Imports
import numpy as np
import math
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spala
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix, QuasiSVD, SVD
from matrix_ode_toolbox import MatrixOde, SylvesterLikeOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from krylov_toolbox import ExtendedKrylovSpace, RationalKrylovSpace

Matrix = ndarray | LowRankMatrix | spmatrix


#%% General phi functions
def sylvester_phi_k(A: Matrix, B: Matrix, h: float, k: int, X0: Matrix) -> Matrix:
    """
    Computes the action of phi_k(hS) on X0, where S(X) = AX + XB.

    If the input is low-rank, then the output is low-rank.
    
    Computing Z(t) = t^k phi_k(tS) (X0) is equivalent to solving the ODE
        dZ/dt = A X + X B + t^(k-1)/(k-1)! X0
        Z(0) = 0
    
        The case k=0 is special and is handled separately:
            Z(t) = phi_0(tS) X0 = exp(tS) X0 = exp(tA) X0 exp(tB)
        which is equivalent to solving the ODE
            dZ/dt = A Z + Z B
            Z(0) = X0

    Parameters
    ----------
    A : Matrix
        The matrix (operator) that acts on X0 such that S(X) = AX + XB.
    B : Matrix
        The matrix (operator) that acts on X0 such that S(X) = AX + XB.
    h : float
        The step size.
    k : int
        The order of the phi function.
    X0 : Matrix
        The matrix X0.

    Returns
    -------
    Z : Matrix
        The matrix Z(t) = phi_k(tS) (X0).
    """
    
    if k == 0:
        # Use scipy.sparse.linalg.expm_multiply
        if isinstance(X0, LowRankMatrix):
            Z = X0.expm_multiply(A, h, side='left').expm_multiply(B, h, side='right')
        else:
            Z = spala.expm_multiply(A, X0, start=0, stop=h, endpoint=True, num=2)[-1]
            Z = spala.expm_multiply(B.T, Z.T, start=0, stop=h, endpoint=True, num=2)[-1].T

    else:
        # Define the Sylvester-like ODE
        def ode(t, Z):
            return t**(k-1)/math.factorial(k-1) * X0
        sylvester_like_ode = SylvesterLikeOde(A, B, ode)
        # Solve the Sylvester-like ODE
        Z = solve_matrix_ivp(sylvester_like_ode, (0, h), np.zeros(X0.shape), solver='automatic', dense_output=True)
        if isinstance(X0, LowRankMatrix):
            Z = SVD.from_dense(Z)

    return Z    


# %% Combined phi functions
def sylvester_combined_phi_k(A: Matrix, B: Matrix, h: float, k: int, X: list, solver: str = 'automatic', **extra_args) -> Matrix:
    """
    Computes phi_0(hS) X[0] + h phi_1(hS) X[1] + ... + h^k phi_k(hS) X[k], where S(X) = AX + XB.

    If the initial value X[0] is low-rank, then the output is low-rank.

    NOTE: Be aware that the coefficients are h^k, not h. 
    To compensate, you have to manually divide X[k] by h^k BEFORE calling this function.

    NOTE: If you want to use a Krylov method, reduce A, B and X[j] before calling this function.

    Examples
    --------
    If k=0, then computes
        phi_0(hS) X[0] = exp(hS) X[0] = exp(hA) X[0] exp(hB)
    If k=1, then computes
        phi_0(hS) X[0] + h phi_1(hS) X[1]
    If k=2, then computes
        phi_0(hS) X[0] + h phi_1(hS) X[1] + h^2 phi_2(hS) X[2]

    Parameters
    ----------
    A : Matrix
        The matrix (operator) such that S(X) = AX + XB.
    B : Matrix
        The matrix (operator) such that S(X) = AX + XB.
    h : float
        The step size.
    k : int
        The order of the phi function.
    X : list
        The matrices X[0], X[1], ..., X[k].

    Returns
    -------
    Z : Matrix
        The matrix Z(h) = phi_0(hS) X[0] + h phi_1(hS) X[1] + ... + h^k phi_k(hS) X[k].
    """

    # Check length of Xj
    if len(X) != k+1:
        raise ValueError("The length of Xj must be k+1.")
    
    # Check that all Xj have the same type
    if not all(isinstance(Xj, type(X[0])) for Xj in X):
        raise ValueError("All Xj must have the same type. Different types are not supported yet.")

    if k == 0:
        # Use scipy.sparse.linalg.expm_multiply
        if isinstance(X[0], LowRankMatrix):
            Z = X[0].expm_multiply(A, h, side='left').expm_multiply(B, h, side='right')
        else:
            Z = spala.expm_multiply(A, X[0], start=0, stop=h, endpoint=True, num=2)[-1]
            Z = spala.expm_multiply(B.T, Z.T, start=0, stop=h, endpoint=True, num=2)[-1].T


    if k == 1:
        # Define the equivalent ODE
        def G(t, Z):
            return X[1]
        ode = SylvesterLikeOde(A, B, G)
        # Solve the ODE
        Z = solve_matrix_ivp(ode, (0, h), X[0], solver=solver, **extra_args)
        if isinstance(X[0], LowRankMatrix):
            Z = SVD.from_dense(Z)

    else:
        # Define the non-homogeneous part of the ODE
        def G(t, Z):
            return np.sum([t**(j-1)/math.factorial(j-1) * X[j] for j in np.arange(1, k+1)], axis=0)
        # Define the equivalent ODE
        ode = SylvesterLikeOde(A, B, G)
        # Solve the ODE
        Z = solve_matrix_ivp(ode, (0, h), X[0], solver=solver, **extra_args)
        if isinstance(X[0], LowRankMatrix):
            Z = SVD.from_dense(Z)

    return Z




        
        
