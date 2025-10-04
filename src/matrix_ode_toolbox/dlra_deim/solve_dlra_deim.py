"""
Author: Benjamin Carrel, University of Geneva, 2024

Utility functions for the DLRA-DEIM framework.
"""

# %% Imports
import numpy as np
import time
from numpy import ndarray
from tqdm import tqdm
from typing import Tuple
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolution
from low_rank_toolbox import LowRankMatrix
from .methods import *
from .dlra_deim_solver import DlraDeimSolver


Matrix = ndarray | LowRankMatrix

available_dlra_deim_methods = {'scipy_dlra_deim': ScipyDlraDeim,
                               'projected_runge_kutta_deim': ProjectedRungeKuttaDeim,
                               'PRK-DEIM': ProjectedRungeKuttaDeim,
                               'projected_exponential_runge_kutta_deim': ProjectedExponentialRungeKuttaDeim,
                                 'PERK-DEIM': ProjectedExponentialRungeKuttaDeim}

def solve_dlra_deim(matrix_ode: MatrixOde,
                    t_span: Tuple[float, float],
                    initial_value: LowRankMatrix,
                    dlra_deim_solver: str | DlraDeimSolver,
                    dlra_deim_kwargs: dict = {'nb_substeps': 1, 'deim_method': 'sqdeim', 'deim_kwargs': {}},
                    t_eval: list = None,
                    dense_output: bool = False,
                    monitor: bool = False) -> MatrixOdeSolution | LowRankMatrix:
    """
    Solver the DLRA-DEIM with the chosen solver.
    NOTE: The rank for the DLRA-DEIM is automatically set to the rank of the initial value. It is consistent with the definition of DLRA-DEIM. If the rank changes during the integration, a message warns the user.
    
    Parameters
    ----------
    matrix_ode : MatrixOde
        The matrix ODE.
    t_span : Tuple[float, float]
        The time span.
    initial_value : LowRankMatrix
        The initial value.
    dlra_deim_solver : str | DlraDeimSolver
        The DLRA-DEIM solver.
    dlra_deim_kwargs : dict, optional
        Additional DLRA-DEIM arguments, by default {'nb_substeps': 1, 'deim_method': 'sqdeim', 'deim_kwargs': {}}.
    t_eval : list, optional
        The time grid points, by default None.
    dense_output : bool, optional
        If True, the solution is stored at each time grid point, by default False.
    monitor : bool, optional
        If True, display the progress bar, by default False.

    Returns
    -------
    MatrixOdeSolution | LowRankMatrix
        The solution of the DLRA-DEIM.
    """
    # Copy the ODE
    matrix_ode = matrix_ode.copy()

    # Select the method
    if isinstance(dlra_deim_solver, str):
        solver = available_dlra_deim_methods[dlra_deim_solver](matrix_ode, **dlra_deim_kwargs)
    else:
        solver = dlra_deim_solver(matrix_ode, **dlra_deim_kwargs)
    
    # Check the initial value
    if not isinstance(initial_value, LowRankMatrix):
        raise ValueError('The initial value must be a LowRankMatrix.')
    if initial_value.rank == 0:
        raise ValueError('The rank of the initial value must be greater than 0.')
    
    # Check the time span
    if not isinstance(t_span, tuple):
        raise ValueError('t_span must be a tuple.')
    if len(t_span) != 2:
        raise ValueError('t_span must be a tuple of length 2.')
    if t_span[0] >= t_span[1]:
        raise ValueError(f't_span must be a tuple (t0, t1) with t0 < t1, not {t_span}.')
    
    # Single output case   
    if t_eval is None:
        Y1 = solver.solve(t_span, initial_value)
        if dense_output:
            return Y1.todense()
        else:
            return Y1
    
    # Other cases   
    ## Process t_eval
    t_eval = np.array(t_eval)
    if t_eval[0] != t_span[0]:
        t_eval = np.concatenate([[t_span[0]], t_eval])
    if t_eval[-1] != t_span[1]:
        t_eval = np.concatenate([t_eval, [t_span[1]]])

    ## Preallocate
    n = len(t_eval)
    Ys = np.empty(n, dtype=type(initial_value))

    ## Monitor
    if monitor:
        print('----------------------------------------')
        print(f'{solver.info}')
        loop = tqdm(np.arange(n-1), desc=f'Solving DLRA-DEIM')
    else:
        loop = np.arange(n-1)

    ## Integrate
    Ys[0] = initial_value
    computation_time = np.zeros(n-1)
    for i in loop:
        c0 = time.time()
        Ys[i+1] = solver.solve((t_eval[i], t_eval[i+1]), Ys[i])
        computation_time[i] = time.time() - c0

    ## Return
    if dense_output:
        for i in np.arange(n):
            Ys[i] = Ys[i].todense()

    return MatrixOdeSolution(matrix_ode, t_eval, Ys, computation_time) #, **solver.extra_data)