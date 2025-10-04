"""
Author: Benjamin Carrel, University of Geneva, 2022

Schrodinger differential equation structure. Subclass of MatrixODE.
"""

# %% IMPORTATIONS
from scipy.sparse import spmatrix
import numpy as np
import scipy.linalg as la
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix, QuasiSVD, SVD
from .matrix_ode import MatrixOde
from typing import Union


# %% CLASS NON LINEAR SCHRÖDINGER ODE
class NonLinearSchrodingerOde(MatrixOde):
    """
    Subclass of MatrixOde. Specific to the non-linear Schrodinger equation.

    Non-linear Schrodinger equation :
        .. math::
            \\dot{X}(t) = + i/2  (A X(t) + X(t) A) + i \\alpha |X(t)|^2 X(t).

    NOTE: The ODE is complex-valued.

    A is a sparse matrix
    alpha is a complex number
    non-linear term is taken element-wise

    References:
        [1] E. Kieri and B. Vandereycken, “Projection Methods for Dynamical Low-Rank Approximation of High-Dimensional Problems,” Computational Methods in Applied Mathematics, Jan. 2019
    """

    # ATTRIBUTES
    name = 'Non-linear Schrodinger'
    _use_low_rank_hadamard = True
    A = MatrixOde.create_parameter_alias(0)
    B = A # for compatibility with SylvesterLikeOde

    def __init__(self, A: spmatrix, alpha: complex):
        # Check inputs
        assert isinstance(A, spmatrix), "A must be a sparse matrix"
        self.alpha = complex(alpha)

        # Call parent constructor
        super().__init__(A, alpha)
        
    def preprocess_ode(self):
        "Pre-processing specific to the ODE"
        super().preprocess_ode()
        ode_type = self.ode_type
        

    def low_rank_square_abs(self, X: QuasiSVD) -> QuasiSVD:
        "Component-wise square absolute value of X"
        # Transform to SVD
        if not isinstance(X, SVD):
            Y = SVD.truncated_svd(X)
        else:
            Y = X
        # Square absolute value of SVD
        s = Y.sing_vals
        U1, U2 = np.real(Y.U), np.imag(Y.U)
        V1, V2 = np.real(Y.V), np.imag(Y.V)
        A = SVD(U1, s, V1) + SVD(U2, s, V2) # real part
        B = SVD(U2, s, V1) - SVD(U1, s, V2) # imaginary part
        # Hadamard products
        X2 = A.hadamard(A) + B.hadamard(B)
        # print('Rank of X2:', X2.rank)
        return X2


    # VECTOR FIELDS
    def ode_F(self, t: float, X: Union[ndarray, spmatrix, LowRankMatrix], rows: list = None, cols: list = None) -> Union[ndarray, spmatrix, LowRankMatrix]:
        "Vector field of the ODE"
        LX = self.linear_field(t, X, rows, cols)
        GX = self.non_linear_field(t, X, rows, cols)
        dX = LX + GX
        return dX
    
    def linear_field(self, t: float, X: Union[ndarray, spmatrix, LowRankMatrix], rows: list = None, cols: list = None) -> Union[ndarray, spmatrix, LowRankMatrix]:
        "Linear field of the ODE"
        if rows is not None and cols is not None:
            dX = 1j/2 * (self.A[rows, :].dot(X[:, cols]) + self.A[:, cols].T.dot(X[rows, :].T).T)
        elif rows is not None:
            if isinstance(X, LowRankMatrix):
                dX = 1j/2 * (X.dot(self.A[rows, :], side='left', dense_output=True) + self.A.T.dot(X[rows, :].T).T)
            else:
                dX = 1j/2 * (self.A[rows, :].dot(X) + self.A.T.dot(X[rows, :].T).T)
        elif cols is not None:
            if isinstance(X, LowRankMatrix):
                dX = 1j/2 * (self.A.dot(X[:, cols]) + X.dot(self.A[:, cols], side='right', dense_output=True))
            else:
                dX = 1j/2 * (self.A.dot(X[:, cols]) + self.A[:, cols].T.dot(X.T).T)
        else:
            if isinstance(X, LowRankMatrix):
                dX = 1j/2 * (X.dot(self.A, side='left') + X.dot(self.B, side='right'))
            else:
                dX = 1j/2 * (self.A.dot(X) + self.B.T.dot(X.T).T)
        return dX
        

    def non_linear_field(self, t: float, X: Union[ndarray, spmatrix, LowRankMatrix], rows: list = None, cols: list = None) -> Union[ndarray, spmatrix, LowRankMatrix]:
        "Non linear field of the ODE"
        if rows is not None and cols is not None:
            dX = 1j * self.alpha * np.abs(X[rows, :][:, cols])**2 * X[rows, :][:, cols]
        elif rows is not None:
            if isinstance(X, LowRankMatrix):
                dX = 1j * self.alpha * np.abs(X[rows, :])**2 * X[rows, :]
            else:
                dX = 1j * self.alpha * np.abs(X[rows, :])**2 * X[rows, :]
        elif cols is not None:
            if isinstance(X, LowRankMatrix):
                dX = 1j * self.alpha * np.abs(X[:, cols])**2 * X[:, cols]
            else:
                dX = 1j * self.alpha * np.abs(X[:, cols])**2 * X[:, cols]
        else:
            if isinstance(X, LowRankMatrix):
                if self._use_low_rank_hadamard:
                    X2 = self.low_rank_square_abs(X)
                    dX = 1j * self.alpha * X2.hadamard(X)
                else:
                    X_full = X.todense()
                    X2 = np.abs(X_full)**2
                    dX = 1j * self.alpha * X2 * X_full
            else:
                X2 = np.abs(X)**2
                dX = 1j * self.alpha * X2 * X
        return dX
