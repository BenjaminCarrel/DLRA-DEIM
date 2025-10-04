"""
Author: Benjamin Carrel, University of Geneva, 2022

General structure for matrix ODEs
"""


# Imports
from __future__ import annotations
from copy import deepcopy
from numpy import ndarray
import numpy as np
import numpy.linalg as la
import warnings
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix, SVD, QuasiSVD, DEIM

Matrix = ndarray | LowRankMatrix

#%% Class
class MatrixOde:
    r"""General matrix ODE structure class. Contains essential methods for matrix ODEs.

    A matrix ODE is of the form :math:`\dot{X}(t) = F(t, X(t))` where :math:`X(t)` is a matrix.

    How to create a specific ODE structure:
    1. Create a new class that inherits from MatrixOdeStructure.
    2. Overload the necessary methods. See the documentation of the methods for more details. See SylvesterOdeStructure for an example.
    """

    #%% ATTRIBUTES
    name = "General"

    #%% FUNDAMENTALS
    def __init__(self, *parameters, **kwargs):
        "Initialize the problem."
        self._parameters = parameters
        self.select_ode(**kwargs)

    def __repr__(self) -> str:
        return (f'{self.name} ODE structure with {len(self._parameters)} parameters.')

    def __call__(self, *args, **kwds):
        return self.ode(*args, **kwds)

    def copy(self):
        "Copy the problem"
        return deepcopy(self) # use deepcopy otherwise some elements might not be copied

    @staticmethod
    def create_parameter_alias(index: int) -> property:
        def getter(self) -> ndarray:
            return self._parameters[index]

        def setter(self, value: ndarray):
            self._parameters[index] = value

        return property(getter, setter)
    
    # Valid ODEs
    valid_odes = {'F': 'ode_F'}

    @property
    def ode(self):
        return getattr(self, self.valid_odes[self.ode_type])
    
    @ode.setter
    def ode(self, value):
        setattr(self, self.valid_odes[self.ode_type], value)

    def vec_ode(self, t: float, x: np.ndarray, shape: tuple) -> np.ndarray:
        "Current ode vectorized"
        def fun_vec_ode(t, x):
            X = np.reshape(x, shape)
            dX = self.ode(t, X)
            return dX.flatten()
        return fun_vec_ode(t, x)    

    def select_ode(self, ode_type: str = 'F', mats_uv: tuple = (), **extra_args):
        """
        Select the current ODE that will be integrated using any of the integrate methods.
        Parameters
        ----------
        ode_type: str
            Can be F, K, S, L, minus_S.
        mats_UV: tuple
            Depending on type, you need to supply the orthonormal matrices U, V in mats_UV. Example: (U,)
        extra_args: dict
            Extra arguments to be passed to the ode. 
            Examples: 
                extra_args['indexes'] = [indexes_U, indexes_V] for DEIM.
                extra_args['inv_mats_deim'] = [inv_PtU_deim, inv_PtV_deim] for DEIM.
        """
        # SET CURRENT ODE
        self.ode_type = ode_type
        self.mats_uv: tuple = mats_uv
        self.extra_args = extra_args
        self.preprocess_ode()
        

    def preprocess_ode(self):
        "Preprocess the ODE. Overload this method for specific structures."
        pass

    #%% VECTOR FIELDS
    ## General vector field
    def ode_F(self, t: float, X: Matrix, **extra_args) -> Matrix:
        "Function of the ODE. Overload this method."
        "extra_args are optional and can be used to pass extra arguments to the ODE, like interpolation points for DEIM."
        return NotImplementedError('Cannot compute the ODE. Overload the method "ode_F".')

    ## Tangent space projections    
    def DEIM_tangent_space_ode_F(self, t: float, X: SVD, U_indexes: list, V_indexes: list, M_u: ndarray, M_v: ndarray, truncate: bool = False) -> SVD:
        "Project the ODE obliquely onto the tangent space of rank r matrices using DEIM."
        # Check input
        if not isinstance(X, LowRankMatrix):
            raise TypeError("X must be a LowRankMatrix.")
        # Compute the ODE
        FX_u = self.ode_F(t, X, rows=U_indexes)
        FX_v = self.ode_F(t, X, cols=V_indexes)
        FX_uv = self.ode_F(t, X, rows=U_indexes, cols=V_indexes)
        PFX = X.project_onto_DEIM_tangent_space(FX_u, FX_v, FX_uv, M_u, M_v, truncate)
        return PFX
    
    def tangent_space_ode_F(self, t: float, X: SVD, truncate: bool = False) -> SVD:
        "Project the ODE onto the tangent space of rank r matrices. The rank is given by the input matrix. Overloading this method may be more efficient."
        # Check input
        if not isinstance(X, SVD):
            raise TypeError("X must be a SVD.")

        # Compute the ODE
        FX = self.ode_F(t, X)
        PFX = X.project_onto_tangent_space(FX, truncate)
        return PFX

    ## Other vector fields 
    def linear_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Linear field of the ODE. Specific to a problem. Overload this method."
        return NotImplementedError('Cannot compute the linear field. Overload the method "linear_field".')

    def non_linear_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Non-linear field of the ODE. Specific to a problem. Overload this method."
        return NotImplementedError('Cannot compute the non-linear field. Overload the method "non_linear_field".')

    def stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Stiff field of the ODE. Specific to a problem. Overload this method. By default, it is the linear field."
        return self.linear_field(t, Y, **extra_args)

    def non_stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        "Non-stiff field of the ODE. Specific to a problem. Overload this method. By default, it is the non-linear field."
        return self.non_linear_field(t, Y, **extra_args)
