"""
Author: Benjamin Carrel, University of Geneva, 2024

Projected Runge-Kutta methods for the DLRA-DEIM.
"""

# %% Imports
from low_rank_toolbox import QuasiSVD, SVD
import numpy as np
from matrix_ode_toolbox.dlra_deim import DlraDeimSolver
from matrix_ode_toolbox import MatrixOde


#%% Runge-Kutta tables
# ORDER 1
a1 = float(0)
b1 = np.ones(1)
# ORDER 2
a2 = np.zeros((2, 2))
a2[1, 0] = 1
b2 = np.zeros(2)
b2[0] = 1/2
b2[1] = 1/2
# ORDER 3
a3 = np.zeros((3, 3))
a3[1, 0] = 1/3
a3[2, 0] = 0
a3[2, 1] = 2/3
b3 = np.zeros(3)
b3[0] = 1/4
b3[2] = 3/4
# ORDER 4
a4 = np.zeros((4, 4))
a4[1, 0] = 1/2
a4[2, 1] = 1/2
a4[3, 2] = 1
b4 = np.zeros(4)
b4[0] = 1/6
b4[1] = 1/3
b4[2] = 1/3
b4[3] = 1/6
# Rule 6(5)9b
a8 = np.zeros((8, 8))
a8[1, 0] = 1/8
a8[2, 0] = 1/18
a8[3, 0] = 1/16
a8[4, 0] = 1/4
a8[5, 0] = 134/625
a8[6, 0] = -98/1875
a8[7, 0] = 9/50
a8[2, 1] = 1/9
a8[3, 2] = 3/16
a8[4, 2] = -3/4
a8[5, 2] = -333/625
a8[6, 2] = 12/625
a8[7, 2] = 21/25
a8[4, 3] = 1
a8[5, 3] = 476/625
a8[6, 3] = 10736/13125
a8[7, 3] = -2924/1925
a8[5, 4] = 98/625
a8[6, 4] = -1936/1875
a8[7, 4] = 74/25
a8[6, 5] = 22/21
a8[7, 5] = -15/7
a8[7, 6] = 15/22
b8 = np.zeros(8)
b8[0] = 11/144
b8[3] = 256/693
b8[5] = 125/504
b8[6] = 125/528
b8[7] = 5/72


#%% Class Projected Runge Kutta
class ProjectedRungeKuttaDeim(DlraDeimSolver):
    """
    Class for the projected Runge-Kutta methods accelerate with DEIM methods.
    See Carrel and Vandereycken 2024.
    """

    #%% Class attributes
    # Name of the method
    name = 'Projected Runge-Kutta DEIM'

    # Automatic truncation at each stage:
    # True is more efficient but might lead to rank deficiency issues. 
    # False is more expensive but preserves the rank. -> default
    _allow_automatic_truncation = False # default is False

    # Perform DEIM at each stage:
    # True is less efficient but more accurate. Theory is based on this -> default
    # False is more efficient but less accurate.
    _perform_deim_at_each_stage = True # default is True

    def __init__(self,
                matrix_ode: MatrixOde,
                nb_substeps: int = 1,
                deim_method: str = 'sqdeim',
                deim_kwargs: dict = {},
                order: int = 2,
                **extra_kwargs) -> None:
        """
        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve.
        nb_substeps : int, optional
            The number of substeps to take. The default is 1.
        deim_method : str, optional
            The DEIM method to use. The default is 'sqdeim'.
        deim_kwargs : dict, optional
            The DEIM method arguments. The default is {}.
        order : int, optional
            The order of the Runge-Kutta method. The default is 2.
        **extra_kwargs : dict, optional
            Extra arguments for the DEIM method.
        """
        super().__init__(matrix_ode, nb_substeps, deim_method, deim_kwargs, **extra_kwargs)
        self.order = order

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Projected Runge-Kutta accelerated by the DEIM (PRK-DEIM) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- DEIM method: {self.deim_method} \n"
        info += f"---- DEIM kwargs: {self.deim_kwargs} \n"
        info += f'-- {self.order} stage(s)'
        return info

    @property
    def RK_rule(self) -> tuple:
        """Shortcut for calling the table"""
        s = self.order
        if s == 1:
            a = a1
            b = b1
        elif s == 2:
            a = a2
            b = b2
        if s == 3:
            a = a3
            b = b3
        if s == 4:
            a = a4
            b = b4
        if s == 8:
            a = a8
            b = b8
        c = np.zeros(s)
        for i in np.arange(1, s):
            c[i] = sum(a[i, j] for j in np.arange(1, i))
        return a, b, c

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> SVD:
        """
        One-step method of projected Runge-Kutta of the given order.

        Parameters
        ----------
        t_subspan : tuple
            The time interval (t0, tf).
        Y0 : QuasiSVD
            The initial value. 
        """
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, QuasiSVD), "Y0 must be a QuasiSVD (or SVD)."
        
        # Variable
        rank = Y0.rank
        a, b, c = self.RK_rule
        s = self.order
        h = t_subspan[1] - t_subspan[0]
        eta = np.empty(s, dtype=SVD)
        kappa = np.empty(s, dtype=SVD)

        # PRK METHOD
        eta[0] = Y0
        extra_args = {'U_indexes': self.indexes_U, 'M_u': self.M_U, 'V_indexes': self.indexes_V, 'M_v': self.M_V, 'truncate': self._allow_automatic_truncation}
        kappa[0] = self.matrix_ode.DEIM_tangent_space_ode_F(t_subspan[0], eta[0], **extra_args)

        # PRK LOOP
        for j in np.arange(1, s):
            if self._allow_automatic_truncation:
                big_eta = Y0 + h * np.sum([a[j, i] * kappa[i] for i in np.arange(0, j)])
            else:
                big_eta = SVD.multi_add([Y0] + [h * a[j, i] * kappa[i] for i in np.arange(0, j)])
            eta[j] = big_eta.truncate(rank)
            tj = t_subspan[0] + c[j] * h
            if self._perform_deim_at_each_stage:
                S_u, M_u = self.DEIM_shortcut(eta[j].U)
                S_v, M_v = self.DEIM_shortcut(eta[j].V)
            else:
                S_u, M_u = self.indexes_U, self.M_U
                S_v, M_v = self.indexes_V, self.M_V
            extra_args = {'U_indexes': S_u, 'M_u': M_u, 'V_indexes': S_v, 'M_v': M_v, 'truncate': self._allow_automatic_truncation}
            kappa[j] = self.matrix_ode.DEIM_tangent_space_ode_F(tj, eta[j], **extra_args)

        # PRK OUTPUT
        if self._allow_automatic_truncation:
            Y1 = Y0 + h * np.sum([b[i] * kappa[i] for i in np.arange(0, s)])
        else:
            Y1 = SVD.multi_add([Y0] + [h * b[i] * kappa[i] for i in np.arange(0, s)], truncate=False)
        Y1 = Y1.truncate(rank)

        return Y1
    
