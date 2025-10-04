'''
Author: Benjamin Carrel, University of Geneva, 2024

General class for implementing DLRA-DEIM solvers.
'''

#%% Imports
import numpy as np
from low_rank_toolbox import LowRankMatrix, DEIM, QDEIM, gpode, gpodr, ARP, OCSS, sQDEIM, oversampling_sQDEIM
from matrix_ode_toolbox import MatrixOde

#%% Common class for the DLRA-DEIM solvers
class DlraDeimSolver:
    '''
    Class for the DLRA-DEIM solvers
    
    How to implement a new DLRA-DEIM method:
    1. Create a new class that inherits from DlraDeimSolver
    2. Create a specific init method that takes the necessary parameters
    3. Implement the method stepper
    Well done! You can now use your method in the solve_dlra_deim function.
    '''

    #%% Class attributes
    # Name of the class
    name = 'Generic DLRA-DEIM'

    # List of available DEIM methods
    _deim_methods = {'DEIM': DEIM,
                        'deim': DEIM,
                        'QDEIM': QDEIM,
                        'qdeim': QDEIM,
                        'sQDEIM': sQDEIM,
                        'sqdeim': sQDEIM,
                        'ARP': ARP,
                        'arp': ARP,
                        'OCSS': OCSS,
                        'ocss': OCSS,
                        'gpode': gpode,
                        'gpodr': gpodr,
                        'oversampling_sqdeim': oversampling_sQDEIM}
    
    # Perform DEIM at each substep: 
    # True is less efficient but more accurate, it is also the theoretical way to do it -> default
    # False is more efficient but less accurate
    _perform_deim_at_each_substep = True # default is True
    

    #%% Static methods
    def __init__(self, 
                 matrix_ode: MatrixOde, 
                 nb_substeps: int = 1,
                 deim_method: str = 'sqdeim',
                 deim_kwargs: dict = {},
                **extra_kwargs) -> None:
        """
        Class for the DLRA-DEIM solvers.
        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE.
        nb_substeps : int, optional
            The number of substeps, by default 1.
        deim_method : str, optional
            The kind of DEIM to use, by default 'sqdeim'.
        deim_kwargs : dict, optional
            The DEIM arguments, by default {}.
        extra_kwargs : dict, optional
            Extra arguments, by default {}.

        Notes
        -----
        - The DEIM kind can be 'deim', 'qdeim', 'sqdeim', 'arp', 'ocss', 'gpode', 'gpodr' or 'oversampling_sqdeim'.
        - The DEIM arguments are passed to the DEIM function.
        # - The discontinuous detector can be 'projected_field_1', 'projected_field_2' or 'indexes'.
        # - The discontinuous detector arguments are passed to the discontinuous detection function.
        # - The smoothing method can be 'bisection' or 'double_deim'.
        # - The smoothing arguments are passed to the smoothing function.
        - The extra arguments are not used in this class, but can be used in the child classes.
        """
        # Check the inputs
        if not isinstance(matrix_ode, MatrixOde):
            raise ValueError("matrix_ode must be a MatrixOde.")
        if not isinstance(nb_substeps, int):
            raise ValueError("nb_substeps must be an integer.")
        if not isinstance(deim_method, str):
            raise ValueError("deim_method must be a string.")
        if not isinstance(deim_kwargs, dict):
            raise ValueError("deim_kwargs must be a dictionary.")
        if not deim_method in self._deim_methods:
            raise ValueError(f"deim_method must be one of {self._deim_methods.keys()}.")
        
        # Save the arguments
        self.matrix_ode = matrix_ode.copy()
        self.nb_substeps = nb_substeps
        self.extra_kwargs = extra_kwargs

        # Process the DEIM arguments
        self.deim_method = deim_method
        self.deim_kwargs = deim_kwargs

        # Store extra args
        self.extra_args = extra_kwargs

        # Make extra data storage
        self.extra_data = {}
        
    
    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Generic DLRA-DEIM \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- DEIM method: {self.deim_method} \n"
        info += f"---- DEIM kwargs: {self.deim_kwargs} \n"
        return info
    
    def __repr__(self) -> str:
        return self.info
    
    def DEIM_shortcut(self, U: np.ndarray) -> tuple:
        "Shortcut for DEIM"
        # Check if the DEIM method is valid
        if self.deim_method not in self._deim_methods:
            raise ValueError(f"DEIM method {self.deim_method} not recognized.")
        return self._deim_methods[self.deim_method](U, compute_M=True, **self.deim_kwargs)
    
    def solve(self, t_span: tuple, Y0: LowRankMatrix):
        "Solve the DLRA-DEIM by calling the adaptive_stepper method. Necessary for substepping."
        # Initialization
        t0, tf = t_span
        ts = np.linspace(t0, tf, self.nb_substeps + 1, endpoint=True)
        self.stepsize = ts[1] - ts[0]
        # self.current_inner_loop = 0
        Y = Y0

        if not self._perform_deim_at_each_substep:
            # Perform DEIM
            self.indexes_U, self.M_U = self.DEIM_shortcut(Y.U)
            self.indexes_V, self.M_V = self.DEIM_shortcut(Y.V)

        # Loop over the substeps
        for i in np.arange(self.nb_substeps):
            # Initialization
            previous_rank = Y.rank

            # Perform DEIM
            if self._perform_deim_at_each_substep:
                self.indexes_U, self.M_U = self.DEIM_shortcut(Y.U)
                self.indexes_V, self.M_V = self.DEIM_shortcut(Y.V)
            
            # Perform the adaptive substep
            # X = self.selected_stepper(ts[i:i+2], X)
            Y = self.stepper(ts[i:i+2], Y)

            # Check if the rank has changed
            if Y.rank != previous_rank:
                print(f'Rank has changed from {previous_rank} to {Y.rank} at t = {ts[i+1]}')

        return Y
    
    
    #%% Methods to be overloaded
    def stepper(self, t_span: tuple, X0: LowRankMatrix) -> LowRankMatrix:
        """Perform one step of the DLRA-DEIM method.
        It must be overloaded in the child class.
        For efficient computation, use the self.indexes_U, self.M_U and self.indexes_V, self.M_V attributes.
        
        Parameters
        ----------
        t_span : tuple
            The time span of the substep.
        X0 : LowRankMatrix
            The initial condition.
        
        Returns
        -------
        X : LowRankMatrix
            The solution at the end of the substep.
        """
        raise NotImplementedError("This is a generic class. The method stepper must be overloaded in the child class.")


    
