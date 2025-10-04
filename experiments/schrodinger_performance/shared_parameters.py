import numpy as np
from low_rank_toolbox import SVD
import os

# ODE parameters
sizes = [1024]
t_span = (0, 1)
nb_steps = 10
t_eval = np.linspace(t_span[0], t_span[1], nb_steps+1)

# DLRA solvers parameters
ranks = [[3, 6, 9]]
nb_substeps = 100

## Path to save data
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/size_{sizes}/rank_{ranks}/time_{t_span[1]}_steps_{nb_steps}/substeps_{nb_substeps}")
if not os.path.exists(data_path):
    os.makedirs(data_path)
