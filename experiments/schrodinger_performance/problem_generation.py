#%% Imports
from shared_parameters import *
from matrix_ode_toolbox.integrate import solve_matrix_ivp

from scipy import sparse as sps
from matrix_ode_toolbox import NonLinearSchrodingerOde


# %% Non linear Schrodinger equation
def make_nonlinear_nonstiff_schrodinger_ode(size):
    """
    Make a non-linear non-stiff Schrodinger problem.

    Reference
      "Projection Methods for Dynamical Low-Rank Approximation of High-Dimensional Problems" by Kieri & Vandereycken, 2019.

    Parameters
    ----------
    size : int
        Size of the discretization.

    Returns
    -------
    ode : NonLinearSchrodingerOde
        The ODE structure
    X0 : np.ndarray
        The initial value.
    """
    ## OPERATOR: coupling of lattice sites
    A = sps.diags([1, 1], [-1, 1], shape=(size, size), format="csc", dtype=np.complex128)

    ## alpha: parameter of the nonlinear term
    alpha = 0.1

    ## Define the ODE
    ode = NonLinearSchrodingerOde(A, alpha)

    ## INITIAL VALUE: X0 as described in the paper
    mu1 = round(0.6 * size)
    mu2 = round(0.5 * size)
    nu1 = round(0.5 * size)
    nu2 = round(0.4 * size)
    sigma = 0.1 * size
    X0 = np.zeros((size, size), dtype=np.complex128)
    for j in range(size):
        for k in range(size):
            X0[j, k] = np.exp(-((j - mu1) ** 2 + (k - nu1) ** 2) / (sigma ** 2)) + np.exp(-((j - mu2) ** 2 + (k - nu2) ** 2) / (sigma ** 2))

    return ode, X0
