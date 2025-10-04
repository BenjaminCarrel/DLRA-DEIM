#%% Import packages
from shared_parameters import *
from matrix_ode_toolbox.dlra_deim import ProjectedRungeKuttaDeim, ProjectedExponentialRungeKuttaDeim

#%% DLRA methods

# Initialize data
dlra_solvers = []
dlra_kwargs = []
dlra_names = []
dlra_linestyles = []

# PRK parameters
dlra_solvers += ['PRK']
# dlra_solvers += ['PERK']
# dlra_solvers += ['scipy_dlra']
dlra_kwargs += [{'nb_substeps': nb_substeps, 
                  'order': order,
                  'krylov_kwargs': krylov_kwargs,
                  'strict_order_conditions': True,
                  'use_closed_form': False,
                  'scipy_method': 'LSODA'
                  }]
dlra_names += [f'DLRA']
dlra_linestyles += ['-.']

#%% DLRA-DEIM methods
# Initialize data
dlra_deim_solvers = []
dlra_deim_kwargs = []
dlra_deim_names = []
dlra_deim_legends = []
dlra_deim_linestyles = []

# Standard DEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
                                'order': order,
                                'deim_method': 'deim',
                                'deim_kwargs': {},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'LSODA'
                  }]
dlra_deim_names += [f'DLRA-DEIM']
dlra_deim_legends += [f'DLRA-DEIM']
dlra_deim_linestyles += ['-']

# QDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
                                'order': order, 
                                'deim_method': 'qdeim',
                                'deim_kwargs': {},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'LSODA'
                  }]
dlra_deim_names += [f'DLRA-QDEIM']
dlra_deim_legends += [f'DLRA-QDEIM']
dlra_deim_linestyles += ['-']

# sQDEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
                                'order': order, 
                                'deim_method': 'sqdeim',
                                'deim_kwargs': {},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'LSODA'
        }]
dlra_deim_names += [f'DLRA-sQDEIM']
dlra_deim_legends += [rf'DLRA-sQDEIM ($\eta = 2$)']
dlra_deim_linestyles += ['s']

# OCSS
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps,
                                'order': order, 
                                'deim_method': 'ocss',
                                'deim_kwargs': {},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'LSODA'
        }]
dlra_deim_names += [f'DLRA-OCSS']
dlra_deim_legends += [f'DLRA-OCSS']
dlra_deim_linestyles += ['v']

# ARP
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += [ProjectedExponentialRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
np.random.seed(0)  # For reproducibility
dlra_deim_kwargs += [{'nb_substeps': nb_substeps,
                                'order': order, 
                                'deim_method': 'arp',
                                'deim_kwargs': {},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'LSODA'
        }]
dlra_deim_names += [f'DLRA-ARP']
dlra_deim_legends += [f'DLRA-ARP']
dlra_deim_linestyles += ['-']

# GPODE DEIM
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps, 
                                'order': order,
                                'deim_method': 'gpode',
                                'deim_kwargs': {'oversampling_size': rank},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'RK45'
        }]
dlra_deim_names += [f'DLRA-GPODE']
dlra_deim_legends += [rf'DLRA-GPODE ($\ell = {rank}$)']
dlra_deim_linestyles += ['-']

# sQDEIM+
dlra_deim_solvers += [ProjectedRungeKuttaDeim]
# dlra_deim_solvers += ['scipy_dlra_deim']
dlra_deim_kwargs += [{'nb_substeps': nb_substeps,
                                'order': order,
                                'deim_method': 'oversampling_sqdeim',
                                'deim_kwargs': {'oversampling_size': rank},
                                'krylov_kwargs': krylov_kwargs,
                                'strict_order_conditions': True,
                                'use_closed_form': False,
                                'scipy_method': 'RK45'
        }]
dlra_deim_names += [f'DLRA-sQDEIM+']
dlra_deim_legends += [rf'DLRA-sQDEIM+ ($\ell = {rank}$)']
dlra_deim_linestyles += ['-']