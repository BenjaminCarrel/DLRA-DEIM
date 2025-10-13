import numpy as np
from low_rank_toolbox import SVD
import os

# ODE parameters
size = 64
t_span = (0, 10)
nb_steps = 100
t_eval = np.linspace(t_span[0], t_span[1], nb_steps+1)
dt = t_eval[1] - t_eval[0]

# DLRA solvers parameters
rank = 6
order = 2
nb_substeps = 100
krylov_kwargs = {'size': 2, 'kind': 'extended'}

## Path to save data
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/size_{size}/rank_{rank}/nb_substeps_{nb_substeps}/")
if not os.path.exists(data_path):
    os.makedirs(data_path)