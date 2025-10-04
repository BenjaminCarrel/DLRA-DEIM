#%% Imports
from shared_parameters import *
from matrix_ode_toolbox import SylvesterLikeOde
from matrix_ode_toolbox.utils import laplacian_1d_dx2
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from scipy.sparse.linalg import spsolve


# %% Allen-Cahn equation
def make_allen_cahn(size: int):
    """
    Allen-Cahn equation
        X' = AX + XA + X - X^3
        X(0) = X0
    where A is the 1D Laplacian (times epsilon) as stencil 1/dx^2 [1 -2 1] in csc format, periodic BC

    Reference: Rodgers and Venturi, 2022, Implicit step-truncation integration of nonlinear PDEs on low-rank tensor manifolds.
    """
    ## PARAMETERS
    epsilon = 0.01
    dx = 2 * np.pi / (size+1)

    ## DISCRETIZATION
    xs = np.linspace(dx, 2*np.pi-dx, size) # periodic boundary conditions!
    ys = np.linspace(dx, 2*np.pi-dx, size)

    ## OPERATOR: Laplacian
    A = epsilon * laplacian_1d_dx2(size, dx=dx, periodic=True)
    # A = epsilon * laplacian_1d_dx4(size, dx=xs[1] - xs[0], periodic=True)

    ## SOURCE: G(t, X) = X - X^3 (hadamard product)
    def G(t, X, rows: list = None, cols: list = None):
        if rows is not None and cols is not None:
            return X[rows,:][:, cols] - X[rows, :][:, cols]**3
        elif rows is not None:
            return X[rows, :] - X[rows, :]**3
        elif cols is not None:
            return X[:, cols] - X[:, cols]**3
        else:
            if isinstance(X, LowRankMatrix):
                return X - X.hadamard(X.hadamard(X))
            else:
                return X - X**3

    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE
    u = lambda x, y: (np.exp(-np.tan(x)**2) + np.exp(-np.tan(x)**2)) * np.sin(x) * np.sin(y) / (1 + np.exp(np.abs(1/np.sin(-x/2))) + np.exp(np.abs(1/np.sin(-y/2))))
    f = lambda x, y: u(x,y) # - u(x, 2*y) + u(3*x + np.pi, 3*y + np.pi) - 2*u(4*x, 4*y) + 2 * u(5*x, 5*y)
    # NOTE: you can play with it to get different initial rank
    X0 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            X0[i,j] = u(xs[i], ys[-j])
    # X0 = SVD.truncated_svd(X0)

    return ode, X0

#%% Setup the ODE
make_ode = make_allen_cahn
ode, X0 = make_ode(size)
ode._use_low_rank_hadamard = True
invA = spsolve(ode.A.tocsc(), np.eye(ode.A.shape[0])).dot

#%% Skip initial value instabilities for DLRA
X0 = solve_matrix_ivp(ode, (0, 0.01), X0, dense_output=True)

#%% Solve the ODE with the DLRA methods
Y0 = SVD.truncated_svd(X0, rank)
# %%
