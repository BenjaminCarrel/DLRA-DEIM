#%% Import packages
from shared_parameters import *
from matrix_ode_toolbox.dlra_deim import ProjectedRungeKuttaDeim

# Three colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

#%% DLRA methods

# Initialize data
dlra_solvers = []
dlra_kwargs = []
dlra_names = []
dlra_linestyles = []
dlra_colors = []

# PRK1 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 1}]
dlra_names += [f'PRK1']
dlra_linestyles += ['-']
dlra_colors += [colors[0]]

# PRK2 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 2}]
dlra_names += [f'PRK2']
dlra_linestyles += ['-']
dlra_colors += [colors[1]]

# PRK3 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 3}]
dlra_names += [f'PRK3']
dlra_linestyles += ['-']
dlra_colors += [colors[2]]

#%% DLRA-DEIM methods

# Initialize data
dlra_deim_solvers = []
dlra_deim_kwargs = []
dlra_deim_names = []
dlra_deim_linestyles = []
dlra_deim_colors = []

# PRK1 - sQDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
            'order': 1,
            'deim_method': 'sqdeim',
            'deim_kwargs': {}
            }]
dlra_deim_names += [f'PRK1-sQDEIM']
dlra_deim_linestyles += ['s']
dlra_deim_colors += [colors[0]]

# PRK2 - sQDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
            'order': 2, 
            'deim_method': 'sqdeim',
            'deim_kwargs': {}
            }]
dlra_deim_names += [f'PRK2-sQDEIM']
dlra_deim_linestyles += ['s']
dlra_deim_colors += [colors[1]]

# PRK3 - sQDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
      'order': 3, 
      'deim_method': 'sqdeim',
      'deim_kwargs': {}
      }]
dlra_deim_names += [f'PRK3-sQDEIM']
dlra_deim_linestyles += ['s']
dlra_deim_colors += [colors[2]]
