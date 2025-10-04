#%% Imports
from shared_parameters import *
from problem_generation import make_allen_cahn
from method_definition import dlra_solvers, dlra_kwargs, dlra_names, dlra_deim_solvers, dlra_deim_kwargs, dlra_deim_names
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from matrix_ode_toolbox.dlra_deim import solve_dlra_deim
from low_rank_toolbox import SVD
from scipy.sparse.linalg import spsolve

import dill as pickle


#%% Solve the ODE with the DLRA methods
for i, size in enumerate(sizes):
    print('*************************************************')
    print(f'Problem size: {size}. ({i+1}/{len(sizes)})')
    # Update the size of the problem
    ode, X0 = make_allen_cahn(size)
    ode._use_low_rank_hadamard = True
    invA = spsolve(ode.A.tocsc(), np.eye(ode.A.shape[0])).dot
    # Update the initial condition
    X0 = solve_matrix_ivp(ode, (0, 0.1), X0, dense_output=True)

    # Compute the reference solution
    ref_sol = solve_matrix_ivp(ode, t_span, X0, t_eval=t_eval, monitor=True)
    Xs_ref = ref_sol.todense()

    # Export the reference solution using pickle
    with open(f'{data_path}/ref_{nb_steps}_{size}.pkl', 'wb') as f:
        pickle.dump({'Xs_ref': Xs_ref, 't_eval': t_eval}, f, recurse=True)

    for rank in ranks[i]:
        print('*************************************************')
        print(f'Rank: {rank}. ({i+1}/{len(ranks)})')
        # Low-rank approximation of the initial condition
        Y0 = SVD.truncated_svd(X0, rank)
        
        # Loop over the DLRA methods
        for j, method in enumerate(dlra_solvers):
            # Add the inverse of A to krylov_kwargs
            dlra_kwargs[j]['krylov_kwargs']['invA'] = invA
            dlra_kwargs[j]['krylov_kwargs']['invB'] = invA
            # Compute the solution with the current method
            Ys = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=t_eval, monitor=True, dlra_kwargs=dlra_kwargs[j], dense_output=True)

            # Export the solution using pickle
            with open(f'{data_path}/{dlra_names[j]}_{nb_steps}_{nb_substeps}_{size}_{rank}.pkl', 'wb') as f:
                pickle.dump(Ys, f, recurse=True)

        # Loop over the DLRA-DEIM methods
        for j, method in enumerate(dlra_deim_solvers):
            # Add the inverse of A to krylov_kwargs
            dlra_deim_kwargs[j]['krylov_kwargs']['invA'] = invA
            dlra_deim_kwargs[j]['krylov_kwargs']['invB'] = invA
            # Compute the solution with the current method
            Ys = solve_dlra_deim(ode, t_span, Y0, dlra_deim_solver=method, t_eval=t_eval, monitor=True, dlra_deim_kwargs=dlra_deim_kwargs[j], dense_output=True)

            # Export the solution using pickle
            with open(f'{data_path}/{dlra_deim_names[j]}_{nb_steps}_{nb_substeps}_{size}_{rank}.pkl', 'wb') as f:
                pickle.dump(Ys, f, recurse=True)




# %%
