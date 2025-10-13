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
dlra_legends = []
dlra_linestyles = []
dlra_colors = []

# PRK1 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 1, 'nb_substeps': nb_substeps}]
dlra_names += [f'PRK1']
dlra_legends += [f'PRK1']
dlra_linestyles += ['-.']
dlra_colors += [colors[0]]

# PRK2 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 2, 'nb_substeps': nb_substeps}]
dlra_names += [f'PRK2']
dlra_legends += [f'PRK2']
dlra_linestyles += ['-.']
dlra_colors += [colors[1]]

# PRK3 parameters
dlra_solvers += ['PRK']
dlra_kwargs += [{'order': 3, 'nb_substeps': nb_substeps}]
dlra_names += [f'PRK3']
dlra_legends += [f'PRK3']
dlra_linestyles += ['-.']
dlra_colors += [colors[2]]

#%% DLRA-DEIM methods

# Initialize data
dlra_deim_solvers = []
dlra_deim_kwargs = []
dlra_deim_names = []
dlra_deim_legends = []
dlra_deim_linestyles = []
dlra_deim_colors = []

# PRK1 - sQDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
            'order': 1,
            'deim_method': 'sqdeim',
            'deim_kwargs': {}
            }]
dlra_deim_names += [rf'PRK1-SRRQR']
dlra_deim_legends += [rf'PRK1-SRRQR ($\eta=2$)']
dlra_deim_linestyles += ['-']
dlra_deim_colors += [colors[0]]

# PRK2 - sRRQR
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
            'order': 2, 
            'deim_method': 'sqdeim',
            'deim_kwargs': {}
            }]
dlra_deim_names += [rf'PRK2-SRRQR']
dlra_deim_legends += [rf'PRK2-SRRQR ($\eta=2$)']
dlra_deim_linestyles += ['-']
dlra_deim_colors += [colors[1]]

# PRK3 - sRRQR
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
      'order': 3, 
      'deim_method': 'sqdeim',
      'deim_kwargs': {}
      }]
dlra_deim_names += [rf'PRK3-SRRQR']
dlra_deim_legends += [rf'PRK3-SRRQR ($\eta=2$)']
dlra_deim_linestyles += ['-']
dlra_deim_colors += [colors[2]]

# # PRK1 - ARP
# dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
#             'order': 1,
#             'deim_method': 'arp',
#             'deim_kwargs': {}
#             }]
# dlra_deim_names += [f'PRK1-ARP']
# dlra_deim_linestyles += ['-']
# dlra_deim_colors += [colors[0]]

# # PRK2 - ARP
# dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
#             'order': 2, 
#             'deim_method': 'arp',
#             'deim_kwargs': {}
#             }]
# dlra_deim_names += [f'PRK2-ARP']
# dlra_deim_linestyles += ['-']
# dlra_deim_colors += [colors[1]]

# # PRK3 - APR
# dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
#       'order': 3, 
#       'deim_method': 'arp',
#       'deim_kwargs': {}
#       }]
# dlra_deim_names += [f'PRK3-ARP']
# dlra_deim_linestyles += ['-']
# dlra_deim_colors += [colors[2]]
