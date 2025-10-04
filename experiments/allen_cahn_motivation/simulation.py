#%% Imports
from shared_parameters import *
from problem_generation import ode, X0, Y0, invA
from method_definition import dlra_solvers, dlra_kwargs, dlra_names, dlra_deim_solvers, dlra_deim_kwargs, dlra_deim_names, ProjectedExponentialRungeKuttaDeim
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from matrix_ode_toolbox.dlra_deim import solve_dlra_deim
from low_rank_toolbox import SVD

import os
import dill as pickle
if not os.path.exists(data_path):
    os.makedirs(data_path)


#%% Reference solution
ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
Xs_ref = ref_sol.todense()

# Export the reference solution using pickle
with open(f'{data_path}/ref_{nb_steps}.pkl', 'wb') as f:
    pickle.dump({'Xs_ref': Xs_ref, 't_eval': t_eval}, f, recurse=True)



for i in range(len(dlra_solvers)):
    if dlra_solvers[i] == 'PERK':
        # Add invA to the kwargs
        dlra_kwargs[i]['krylov_kwargs']['invA'] = invA
        dlra_kwargs[i]['krylov_kwargs']['invB'] = invA

    # Solve the DLRA
    dlra_sol = solve_dlra(ode, t_span, Y0, dlra_solvers[i], dlra_kwargs[i], t_eval, monitor=True)
    
    # Save the solution using pickle
    with open(f'{data_path}/{dlra_names[i]}_{nb_steps}_{nb_substeps}.pkl', 'wb') as f:
        pickle.dump(dlra_sol, f, recurse=True)

#%% Solve the ODE with the DLRA-DEIM methods
dlra_deim_sols = []

for i in range(len(dlra_deim_solvers)):
    if dlra_deim_solvers[i] == ProjectedExponentialRungeKuttaDeim:
        # Add invA to the kwargs
        dlra_deim_kwargs[i]['krylov_kwargs']['invA'] = invA
        dlra_deim_kwargs[i]['krylov_kwargs']['invB'] = invA

    # Solve the DLRA-DEIM
    dlra_deim_sol = solve_dlra_deim(ode, t_span, Y0, dlra_deim_solvers[i], t_eval=t_eval, monitor=True, dlra_deim_kwargs=dlra_deim_kwargs[i])

    # Save the solution using pickle
    with open(f'{data_path}/{dlra_deim_names[i]}_{nb_steps}_{nb_substeps}.pkl', 'wb') as f:
        pickle.dump(dlra_deim_sol, f, recurse=True)
    dlra_deim_sols.append(dlra_deim_sol)


# %%
