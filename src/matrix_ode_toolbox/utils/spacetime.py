"""
Author: Benjamin Carrel, University of Geneva, 2022
path: matrix_ode_toolbox/utils/spacetime.py
File for spacetime discretization
"""

#%% Imports
import numpy as np
import scipy.sparse as sparse

#%% Centered finite difference matrix O(dx^2)

## First order derivative
def centered_1d_dx2(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^2))
    """
    D = sparse.diags([-1, 1], [-1, 1], shape=(n, n), format='csc') / (2 * dx)
    if periodic:
        D[0, -1] = -1 / (2 * dx)
        D[-1, 0] = 1 / (2 * dx)
    return D

## Second order derivative (Laplacian)
def laplacian_1d_dx2(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete Laplacian matrix in 1D (error O(dx^2))
    """
    DD = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='lil') / (dx ** 2)
    if periodic:
        DD[0, -1] = 1 / (dx ** 2)
        DD[-1, 0] = 1 / (dx ** 2)
    DD.tocsc()
    return DD

#%% Centered finite difference matrix O(dx^4)

## First order derivative
def centered_1d_dx4(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^4))
    """
    D = sparse.diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(n, n), format='csc') / (12 * dx)
    if periodic:
        D[0, -2] = 1 / (12 * dx)
        D[0, -1] = -8 / (12 * dx)
        D[1, -1] = 1 / (12 * dx)
        D[-1, 0] = 8 / (12 * dx)
        D[-1, 1] = -1 / (12 * dx)
        D[-2, 0] = -1 / (12 * dx)
    return D

## Second order derivative (Laplacian)
def laplacian_1d_dx4(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete Laplacian matrix in 1D (error O(dx^4))
    """
    DD = sparse.diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(n, n), format='csc') / (12 * dx ** 2)
    if periodic:
        DD[0, -2] = -1 / (12 * dx ** 2)
        DD[0, -1] = 16 / (12 * dx ** 2)
        DD[1, -1] = -1 / (12 * dx ** 2)
        DD[-1, 0] = 16 / (12 * dx ** 2)
        DD[-1, 1] = -1 / (12 * dx ** 2)
        DD[-2, 0] = -1 / (12 * dx ** 2)
    return DD



    