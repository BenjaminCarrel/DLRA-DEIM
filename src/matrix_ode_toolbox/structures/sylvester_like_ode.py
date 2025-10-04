"""
Author: Benjamin Carrel, University of Geneva, 2023

Sylvester-like ODE structure. Subclass of MatrixOde.
"""

# %% IMPORTATIONS
from __future__ import annotations
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Callable
import scipy.sparse.linalg as spala
import warnings

Matrix = ndarray | spmatrix | LowRankMatrix

# %% CLASS SYLVESTER-LIKE
class SylvesterLikeOde(MatrixOde):
    """
    Class for Sylvester-like equations. Subclass of MatrixOde.

    Sylvester-like differential equation : 
    X'(t) = A X(t) + X(t) B + G(t, X(t)).
    Initial value given by X(t_0) = X0.

    Typically, A and B are sparse matrices, and G is a non-linear function.

    The linear field is assumed to be stiff, and the non-linear field is assumed to be non-stiff. To change this, edit the stiff_field and non_stiff_field methods.
    """

    #%% ATTRIBUTES
    name = 'Sylvester-like'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    G = MatrixOde.create_parameter_alias(2)

    def __init__(self, A: Matrix, B: Matrix, G: Callable, **kwargs):
        """Sylvester-like differential equation: X'(t) = A X(t) + X(t) B + G(X(t))."""
        # Check inputs
        assert isinstance(A, Matrix), "A must be a sparse matrix"
        assert isinstance(B, Matrix), "B must be a sparse matrix"
        assert callable(G), "G must be a function"

        # INITIALIZATION
        super().__init__(A, B, G, **kwargs)

    @property
    def shape(self) -> tuple:
        return (self.A.shape[0], self.B.shape[1])

    def ode_F(self, t: float, X: Matrix, rows: list = None, cols: list = None) -> Matrix:
        """Return the right-hand side of the ODE."""
        if rows is not None and cols is not None:
            try:
                G_rc = self.G(t, X, rows=rows, cols=cols)
            except:
                print("Warning: pointwise evaluation of G failed. Falling back to full evaluation.")
                G_rc = self.G(t, X)[rows, cols]
            return self.A[rows, :].dot(X[:, cols]) + self.B[:, cols].T.dot(X[rows, :].T).T + G_rc
        elif rows is not None:
            try:
                G_r = self.G(t, X, rows=rows)
            except:
                print("Warning: pointwise evaluation of G failed. Falling back to full evaluation.")
                G_r = self.G(t, X)[rows, :]
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A[rows, :], side='opposite', dense_output=True) + self.B.T.dot(X[rows, :].T).T + G_r
            else:
                return self.A[rows, :].dot(X) + self.B.T.dot(X[rows, :].T).T + G_r
        elif cols is not None:
            try:
                G_c = self.G(t, X, cols=cols)
            except:
                print("Warning: pointwise evaluation of G failed. Falling back to full evaluation.")
                G_c = self.G(t, X)[:, cols]
            if isinstance(X, LowRankMatrix):
                return self.A.dot(X[:, cols]) + X.dot(self.B[:, cols], side='right', dense_output=True) + G_c
            else:
                return self.A.dot(X[:, cols]) + self.B[:, cols].T.dot(X.T).T + G_c
        else:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.A, side='opposite') + X.dot(self.B) + self.G(t, X)
            else:
                return self.G(t, X) + self.A.dot(X) + self.B.T.dot(X.T).T
    
    def preprocess_ode(self):
        "Preprocess the ODE -> compute the factors of the selected ODE"
        super().preprocess_ode()
        A, B, G = self.A, self.B, self.G
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == "F":
            self.Ar, self.Br, self.Gr = A, B, G
            return self
        else:
            raise ValueError(f"ode_type {ode_type} not recognized for Sylvester-like ODE. Valid types are: 'F'.")

    def linear_field(self, t: float, X: Matrix, rows: list = None, cols: list = None) -> Matrix:
        if rows is not None and cols is not None:
            return self.Ar[rows, :].dot(X[:, cols]) + self.Br[:, cols].T.dot(X[rows, :].T).T
        elif rows is not None:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.Ar[rows, :], side='opposite', dense_output=True) + self.Br.T.dot(X[rows, :].T).T
            else:
                return self.Ar[rows, :].dot(X) + self.Br.T.dot(X[rows, :].T).T
        elif cols is not None:
            if isinstance(X, LowRankMatrix):
                return self.Ar.dot(X[:, cols]) + X.dot(self.Br[:, cols], side='right', dense_output=True)
            else:
                return self.Ar.dot(X[:, cols]) + self.Br[:, cols].T.dot(X.T).T
        else:
            if isinstance(X, LowRankMatrix):
                return X.dot(self.Ar, side='left') + X.dot(self.Br, side='right')
            else:
                return self.Ar.dot(X) + self.Br.T.dot(X.T).T

    def non_linear_field(self, t: float, X: Matrix, **extra_args) -> Matrix:
        return self.Gr(t, X, **extra_args)

    def stiff_field(self, t: float, X: Matrix, **extra_args) -> Matrix:
        return self.linear_field(t, X, **extra_args)

    def non_stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        return self.non_linear_field(t, Y, **extra_args)
    
    def solve_reduced_stiff_field(self, t: float, X0: Matrix) -> Matrix:
        "Closed form solution of the stiff field X' = Ar X + X Br and X(0) = X0"
        X0eBr = spala.expm_multiply(self.Br.T, X0.T, start=0, stop=t, num=2, endpoint=True)[-1].T
        return spala.expm_multiply(self.Ar, X0eBr, start=0, stop=t, num=2, endpoint=True)[-1]
    
    def solve_full_stiff_field(self, t: float, X0: Matrix) -> Matrix:
        "Closed form solution of the stiff field X' = A X + X B and X(0) = X0"
        if isinstance(X0, LowRankMatrix):
            return X0.expm_multiply(self.A, t, side='left').expm_multiply(self.B, t, side='right')
        else:
            X0eB = spala.expm_multiply(self.B.T, X0.T, start=0, stop=t, num=2, endpoint=True).T[-1]
            return spala.expm_multiply(self.A, X0eB, start=0, stop=t, num=2, endpoint=True)[-1]
    

    