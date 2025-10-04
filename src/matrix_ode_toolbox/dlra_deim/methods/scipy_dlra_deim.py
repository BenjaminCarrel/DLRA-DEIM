"""
Author: Benjamin Carrel, University of Geneva, 2024

Scipy wrapper to solve the DLRA-DEIM
This method is inefficient but can be used as a reference solution of the DLRA-DEIM.
"""

#%% Imports
from numpy import ndarray
from matrix_ode_toolbox.dlra_deim import DlraDeimSolver
from scipy.integrate import solve_ivp, OdeSolver
from matrix_ode_toolbox import MatrixOde
from low_rank_toolbox import LowRankMatrix, SVD, DEIM, QDEIM, gpode, gpodr
from numpy import ndarray

class ScipyDlraDeim(DlraDeimSolver):
    """
    Class ScipyDlraDeim.
    The method is a wrapper around scipy.integrate.solve_ivp.
    It solves the problem obliquely projected onto the matrix manifold, which is the DLRA-DEIM.
    The method is inefficient and might be unstable, but can be used as a reference solution of the DLRA-DEIM for small problems.
    NOTE: Apply this method only on small problems, for testing purposes.
    """

    #%% Class attributes
    # Name of the method
    name = 'Scipy DLRA-DEIM'

    # Perform DEIM at each evaluation
    # True is the theoretical way to do it, but it is not efficient.
    # False is more efficient as it only performs DEIM at the beginning of the integration.
    _perform_deim_at_each_evaluation = False # Default is False
    
    def __init__(self, 
                 matrix_ode: MatrixOde, 
                 nb_substeps: int = 1, 
                 deim_method: str = 'qdeim',
                 deim_kwargs: dict = {},
                 scipy_kwargs: dict = {'solver': 'RK45', 'rtol': 1e-12, 'atol': 1e-12},
                 **extra_args) -> None:
        deim_args = {'kind': deim_method, 'kwargs': deim_kwargs}
        super().__init__(matrix_ode, nb_substeps, deim_method, deim_kwargs, **extra_args)
        # Save the kwargs
        self.scipy_kwargs = scipy_kwargs
        self.extra_data['solver'] = []

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'DLRA-DEIM solved by scipy (for testing purposes only on small problems) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- DEIM method: {self.deim_method} \n"
        info += f"---- DEIM kwargs: {self.deim_kwargs} \n"
        info += f"-- Scipy's solver: {self.scipy_kwargs["solver"]} \n"
        info += "---- Scipy's extra args: \n"
        for key, value in {k: v for k, v in self.scipy_kwargs.items() if k != 'solver'}.items():
            info += f"------ {key}: {value} \n"
        return info
    
    
    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix) -> SVD:
        "Solves Y' = P_DEIM(Y)[F(Y)] using scipy.integrate.solve_ivp."
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, LowRankMatrix), "Y0 must be a LowRankMatrix."

        # Initialisation
        rank = Y0.rank
        shape = Y0.shape
        y0 = Y0.todense().flatten()

        # Vectorized function
        def vec_f(t: float, y: ndarray) -> ndarray:
            Y = SVD.truncated_svd(y.reshape(shape), rank)
            if self._perform_deim_at_each_evaluation:
                # Perform DEIM at each evaluation
                U_indexes, M_u = self.DEIM_shortcut(Y.U)
                V_indexes, M_v = self.DEIM_shortcut(Y.V)
            else:
                # Use the precomputed DEIM indexes
                U_indexes, M_u = self.indexes_U, self.M_U
                V_indexes, M_v = self.indexes_V, self.M_V
            dY = self.matrix_ode.DEIM_tangent_space_ode_F(t, Y, U_indexes=U_indexes, V_indexes=V_indexes, M_u=M_u, M_v=M_v)
            return dY.todense().flatten()

        # Solve the ODE
        sol = solve_ivp(vec_f, t_subspan, y0, dense_output=True, **self.scipy_kwargs)
        y1 = sol.y[:, -1]
        self.extra_data['solver'].append(sol)
        Y1 = SVD.truncated_svd(y1.reshape(shape), rank)
        return Y1
