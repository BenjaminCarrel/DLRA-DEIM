"""
Author: Benjamin Carrel, University of Geneva, 2022

Utility functions for solving the DLRA.
"""

#%% Imports
import numpy as np
import time
from numpy import ndarray
from tqdm import tqdm
from typing import Tuple
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolution
from matrix_ode_toolbox.dlra import methods
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox.dlra import DlraSolver



Matrix = ndarray | LowRankMatrix

available_dlra_methods = {'scipy_dlra': methods.ScipyDlra,
                          'projected_runge_kutta': methods.ProjectedRungeKutta,
                          'PRK': methods.ProjectedRungeKutta, # shortcut
                          'projected_exponential_runge_kutta': methods.ProjectedExponentialRungeKutta,
                          'PERK': methods.ProjectedExponentialRungeKutta, # shortcut
                          }



#%% DLRA with a fixed rank
def solve_dlra(matrix_ode: MatrixOde, 
               t_span: Tuple[float, float], 
               initial_value: LowRankMatrix,
               dlra_solver: str | DlraSolver = 'scipy_dlra',
               dlra_kwargs: dict = {'nb_substeps': 1},
               t_eval: list = None,
               dense_output: bool = False,
               monitor: bool = False,
               **extra_args) -> MatrixOdeSolution | LowRankMatrix:
    """
    Solve the DLRA with the chosen solver.
    NOTE: The rank for the DLRA is automatically set to the rank of the initial value. It is consistent with the definition of DLRA. If the rank changes during the integration, a message warns the user.

    Parameters
    ----------
    matrix_ode: MatrixOde
        The matrix ODE to solve.
    t_span : tuple
        The time interval (t0, t1) where the solution is computed.
    initial_value : LowRankMatrix
        The low-rank initial value
    dlra_solver : str | DlraSolver, optional
        The method to use, by default 'scipy_dlra'
    dlra_kwargs : dict
        Additional arguments specific to the solver (see the documentation of the solver).
    t_eval : list, optional
        The times where the solution is computed, by default None. If None, only the final value is returned.
    dense_output : bool, optional
        Whether to return a dense output, by default False. Only for testing purposes.
    monitor : bool, optional
        Whether to monitor the progress, by default False.

    Returns
    -------
    solution: MatrixOdeSolution | LowRankMatrix
        The solution of the DLRA. If t_eval is None, only the final value is returned. If dense_output is True, the solutions are converted to dense matrices (only for testing purposes).
    """
    # Copy the ODE
    matrix_ode = matrix_ode.copy()

    # Select the method
    if isinstance(dlra_solver, str):
        solver = available_dlra_methods[dlra_solver](matrix_ode, **dlra_kwargs)
    else:
        solver = dlra_solver(matrix_ode, **dlra_kwargs)

    # Check the initial value
    if not isinstance(initial_value, LowRankMatrix):
        raise ValueError(f'Initial value must be a LowRankMatrix, not {type(initial_value)}.')
    if initial_value.rank is None:
        raise ValueError(f'Initial value must have a rank, not None.')
    if initial_value.rank == 0:
        raise ValueError(f'Initial value must have a rank > 0, not 0.')

    # Check the time span
    if not isinstance(t_span, tuple):
        raise ValueError(f't_span must be a tuple, not {type(t_span)}.')
    if len(t_span) != 2:
        raise ValueError(f't_span must be a tuple of length 2, not {len(t_span)}.')
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
        loop = tqdm(np.arange(n-1), desc=f'Solving DLRA')
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

    try:
        timer = {'Total time': computation_time, 'Solver specific times': solver.timer}
    except:
        timer = computation_time
    return MatrixOdeSolution(matrix_ode, t_eval, Ys, timer, **solver.extra_data)




















