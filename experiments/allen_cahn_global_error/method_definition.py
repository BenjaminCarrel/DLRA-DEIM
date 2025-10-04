#%% Import packages
from shared_parameters import *
from matrix_ode_toolbox.dlra_deim import ProjectedExponentialRungeKuttaDeim

#%% Two colors
colors = ['#1f77b4', '#ff7f0e']

#%% DLRA methods

# Initialize data
dlra_solvers = []
dlra_kwargs = []
dlra_names = []
dlra_linestyles = []
dlra_colors = []

# PERK1 parameters
dlra_solvers += ['PERK']
dlra_kwargs += [{'order': 1,
                'strict_order_conditions': True, 
                'use_closed_form': True, 
                'scipy_method': 'RK45',
                'krylov_kwargs': krylov_kwargs
                }]
dlra_names += [f'PERK1']
dlra_linestyles += ['-.']
dlra_colors += [colors[0]]

# PERK2 parameters
dlra_solvers += ['PERK']
dlra_kwargs += [{'order': 2,
                'strict_order_conditions': True, 
                'use_closed_form': True, 
                'scipy_method': 'RK45',
                'krylov_kwargs': krylov_kwargs
                }]
dlra_names += [f'PERK2']
dlra_linestyles += ['-.']
dlra_colors += [colors[1]]

#%% DLRA-DEIM methods

# Initialize data
dlra_deim_solvers = []
dlra_deim_kwargs = []
dlra_deim_names = []
dlra_deim_linestyles = []
dlra_deim_colors = []

# PERK1 - sQDEIM
dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
dlra_deim_kwargs += [{'order': 1,
                      'strict_order_conditions': True, 
                      'use_closed_form': True, 
                      'scipy_method': 'RK45',
                      'deim_method': 'sqdeim',
                      'deim_kwargs': {},
                      'krylov_kwargs': krylov_kwargs
                    }]
dlra_deim_names += [f'PERK1-sQDEIM']
dlra_deim_linestyles += ['-']
dlra_deim_colors += [colors[0]]

# PERK2 - sQDEIM
dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
dlra_deim_kwargs += [{'order': 2,
                      'strict_order_conditions': True, 
                      'use_closed_form': True, 
                      'scipy_method': 'RK45',
                      'deim_method': 'sqdeim',
                      'deim_kwargs': {},
                      'krylov_kwargs': krylov_kwargs
                    }]
dlra_deim_names += [f'PERK2-sQDEIM']
dlra_deim_linestyles += ['-']
dlra_deim_colors += [colors[1]]


# %%
