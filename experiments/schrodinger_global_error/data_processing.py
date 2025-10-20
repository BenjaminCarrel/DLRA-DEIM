#%% Imports
import numpy as np
from shared_parameters import *
from method_definition import dlra_deim_names, dlra_names
from low_rank_toolbox import SVD
import pickle

#%% Import reference solution
with open(f'{data_path}/ref_{nb_steps}.pkl', 'rb') as f:
    ref_sol = pickle.load(f)
Xs_ref = ref_sol['Xs_ref']
t_eval = ref_sol['t_eval'].flatten()

#%% Compute best rank approximation
best_error = np.zeros(len(Xs_ref))
for i in range(len(Xs_ref)):
    # Truncated SVD
    best_rank_approx = SVD.truncated_svd(Xs_ref[i], rank)

    # Compute the relative error
    best_error[i] = np.linalg.norm(Xs_ref[i] - best_rank_approx.todense(), 'fro') / np.linalg.norm(Xs_ref[i], 'fro')

#%% Compute the global error for several step sizes
global_errors_dlra = np.zeros((len(stepsizes), len(dlra_names)))
global_errors_dlra_deim = np.zeros((len(stepsizes), len(dlra_deim_names)))

# Loop over the number of steps
for i, nb in enumerate(nb_substeps):
    for j, method in enumerate(dlra_names):
        # Load the solution
        with open(f'{data_path}/{dlra_names[j]}_{nb_steps}_{int(nb)}.pkl', 'rb') as f:
            dlra_sol = pickle.load(f)
        # Compute the relative error
        global_errors_dlra[i, j] = np.linalg.norm(dlra_sol.Xs[-1] - Xs_ref[-1], 'fro') / np.linalg.norm(Xs_ref[-1], 'fro')

    for j, method in enumerate(dlra_deim_names):
        # Load the solution
        with open(f'{data_path}/{dlra_deim_names[j]}_{nb_steps}_{int(nb)}.pkl', 'rb') as f:
            dlra_deim_sol = pickle.load(f)
        # Compute the relative error
        global_errors_dlra_deim[i, j] = np.linalg.norm(dlra_deim_sol.Xs[-1] - Xs_ref[-1], 'fro') / np.linalg.norm(Xs_ref[-1], 'fro')

# %% Save the global errors with pickle
with open(f'{data_path}/global_errors.pkl', 'wb') as f:
    pickle.dump({'global_errors_dlra': global_errors_dlra, 'global_errors_dlra_deim': global_errors_dlra_deim, 'best_error': best_error}, f)

# %%
