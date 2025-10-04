"""
Phi functions for a matrix operator A.
"""

#%% Imports
import math
import numpy as np
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spala
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix, SVD
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp


Matrix = ndarray | spmatrix | LowRankMatrix
#%% General phi functions
def matrix_phi_k(A: Matrix, h: float, X0: Matrix, k: int) -> Matrix:
    """
    Computes the action of phi_k(hA) on X0.

    If the input is low-rank, then the output is low-rank.

    Computing Z(t) = t^k phi_k(tA) (X0) is equivalent to solving the ODE
        dZ/dt = A Z + t^(k-1)/(k-1)! X0
        Z(0) = 0
    
    The case k=0 is special and is handled separately:
        Z(t) = phi_0(tA) X0 = exp(tA) X0
    which is equivalent to solving the ODE
        dZ/dt = A Z
        Z(0) = X0

    Parameters
    ----------
    A : Matrix
        The matrix (operator) that acts on X0.
    h : float
        The step size.
    X0 : Matrix
        The matrix X0.
    k : int
        The order of the phi function.

    Returns
    -------
    Z : Matrix
        The matrix Z(t) = phi_k(tA) (X0).
    """

    if k == 0:
        # Use scipy.sparse.linalg.expm_multiply
        if isinstance(X0, LowRankMatrix):
            Z = X0.expm_multiply(A, h, side='left')
        else:
            Z = spala.expm_multiply(A, X0, start=0, stop=h, endpoint=True, num=2)[-1]

    else:
        # Solve the matrix ODE
        def ode(t, Z):
            return t**(k-1)/math.factorial(k-1) * X0 + A.dot(Z)
        matrix_ode = MatrixOde()
        matrix_ode.ode_F = ode
        Z = solve_matrix_ivp(matrix_ode, (0, h), np.zeros(X0.shape), solver='scipy', dense_output=True)
        if isinstance(X0, LowRankMatrix):
            Z = SVD.from_dense(Z)

    return Z
