#%% Imports
from shared_parameters import *
from problem_generation import ode, X0, invA
from method_definition import dlra_solvers, dlra_kwargs, dlra_names, dlra_deim_solvers, dlra_deim_kwargs, dlra_deim_names
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from matrix_ode_toolbox.dlra_deim import solve_dlra_deim
from low_rank_toolbox import SVD
from scipy.sparse.linalg import spsolve

import dill as pickle


#%% Compute reference solution
ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
Xs_ref = ref_sol.todense()

# Export the reference solution using pickle
with open(f'{data_path}/ref_{nb_steps}.pkl', 'wb') as f:
    pickle.dump({'Xs_ref': Xs_ref, 't_eval': t_eval}, f, recurse=True)

#%% Solve the ODE with the DLRA methods
Y0 = SVD.truncated_svd(X0, rank)
# Loop over the number of steps
for i, nb in enumerate(nb_substeps):
    print('*************************************************')
    print(f'Solving DLRA with {nb} substeps. ({i+1}/{len(nb_substeps)})')

    # Loop over the DLRA methods
    for j, method in enumerate(dlra_solvers):
        # Update number of substeps
        dlra_kwargs[j]['nb_substeps'] = int(nb)
        # Add the inverse of A to krylov_kwargs
        dlra_kwargs[j]['krylov_kwargs']['invA'] = invA
        dlra_kwargs[j]['krylov_kwargs']['invB'] = invA
        # Compute the solution with the current method
        Ys = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=t_eval, monitor=True, dlra_kwargs=dlra_kwargs[j], dense_output=True)

        # Export the solution using pickle
        with open(f'{data_path}/{dlra_names[j]}_{nb_steps}_{int(nb)}.pkl', 'wb') as f:
            pickle.dump(Ys, f, recurse=True)
        
#%% Solve the ODE with the DLRA methods
for i, nb in enumerate(nb_substeps):
    print('*************************************************')
    print(f'Solving DLRA-DEIM with {nb} substeps. ({i+1}/{len(nb_substeps)})')
    # Loop over the DLRA-DEIM methods
    for j, method in enumerate(dlra_deim_solvers):
        # Update number of substeps
        dlra_deim_kwargs[j]['nb_substeps'] = int(nb)
        # Add the inverse of A to krylov_kwargs
        dlra_deim_kwargs[j]['krylov_kwargs']['invA'] = invA
        dlra_deim_kwargs[j]['krylov_kwargs']['invB'] = invA
        # Compute the solution with the current method
        Ys = solve_dlra_deim(ode, t_span, Y0, dlra_deim_solver=method, t_eval=t_eval, monitor=True, dlra_deim_kwargs=dlra_deim_kwargs[j], dense_output=True)

        # Export the solution using pickle
        with open(f'{data_path}/{dlra_deim_names[j]}_{nb_steps}_{int(nb)}.pkl', 'wb') as f:
            pickle.dump(Ys, f, recurse=True)



# %%
