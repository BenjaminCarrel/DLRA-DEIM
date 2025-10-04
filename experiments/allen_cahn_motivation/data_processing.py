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

#%% Compute DLRA errors
errors = np.zeros((len(Xs_ref), len(dlra_names)))
times = np.zeros(len(dlra_names))
final_errors = np.zeros(len(dlra_names))

for i in range(len(dlra_names)):
    # Load the solution
    with open(f'{data_path}/{dlra_names[i]}_{nb_steps}_{nb_substeps}.pkl', 'rb') as f:
        dlra_sol = pickle.load(f)

    # Compute the error
    error = np.zeros(len(Xs_ref))
    for j in range(len(Xs_ref)):
        error[j] = np.linalg.norm(Xs_ref[j] - dlra_sol.Xs[j].todense(), 'fro') / np.linalg.norm(Xs_ref[j], 'fro')
    errors[:, i] = error

    # Time of computation and final error
    times[i] = np.sum(dlra_sol['computation_time'])
    final_errors[i] = error[-1]

#%% Compute DLRA-DEIM errors
errors_deim = np.zeros((len(Xs_ref), len(dlra_deim_names)))
times_deim = np.zeros(len(dlra_deim_names))
final_errors_deim = np.zeros(len(dlra_deim_names))

for i in range(len(dlra_deim_names)):
    # Load the solution
    with open(f'{data_path}/{dlra_deim_names[i]}_{nb_steps}_{nb_substeps}.pkl', 'rb') as f:
        dlra_deim_sol = pickle.load(f)

    # Compute the error
    error = np.zeros(len(Xs_ref))
    for j in range(len(Xs_ref)):
        error[j] = np.linalg.norm(Xs_ref[j] - dlra_deim_sol.Xs[j].todense(), 'fro') / np.linalg.norm(Xs_ref[j], 'fro')
    errors_deim[:, i] = error

    # Time of computation and final error
    times_deim[i] = np.sum(dlra_deim_sol['computation_time'])
    final_errors_deim[i] = error[-1]

# %% Save the processed data
with open(f'{data_path}/processed_data.pkl', 'wb') as f:
    pickle.dump({
        'Xs_ref': Xs_ref,
        't_eval': t_eval,
        'best_error': best_error,
        'errors': errors,
        'times': times,
        'final_errors': final_errors,
        'errors_deim': errors_deim,
        'times_deim': times_deim,
        'final_errors_deim': final_errors_deim
    }, f)
print(f"Processed data saved in '{data_path}/processed_data.pkl'")

# %%
