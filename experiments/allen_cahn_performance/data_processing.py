#%% Imports
import numpy as np
from shared_parameters import *
from method_definition import dlra_deim_names, dlra_names, dlra_legends, dlra_deim_legends
import pickle


#%% Compute the final error for all sizes and ranks, and extract the time of computation
final_errors = np.zeros((1+len(dlra_names)+len(dlra_deim_names), len(sizes)*len(ranks[0])))
time_of_computation = np.zeros((1+len(dlra_names)+len(dlra_deim_names), len(sizes)*len(ranks[0])))

# Loop over the sizes and ranks
for i, size in enumerate(sizes):
    # Load the reference solution
    with open(f'{data_path}/ref_{nb_steps}_{size}.pkl', 'rb') as f:
        ref_sol = pickle.load(f)
    Xs_ref = ref_sol['Xs_ref']
    t_eval = ref_sol['t_eval'].flatten()

    for j, rank in enumerate(ranks[i]):

        for k, method in enumerate(dlra_names):
            # Load the solution
            with open(f'{data_path}/{method}_{nb_steps}_{nb_substeps}_{size}_{rank}.pkl', 'rb') as f:
                dlra_sol = pickle.load(f)
            # Compute the relative error
            final_errors[k+1, i*len(ranks[i])+j] = np.linalg.norm(dlra_sol.Xs[-1] - Xs_ref[-1], 'fro') / np.linalg.norm(Xs_ref[-1], 'fro')
            time_of_computation[k+1, i*len(ranks[i])+j] = np.sum(dlra_sol.computation_time['Total time'])

        for k, method in enumerate(dlra_deim_names):
            # Load the solution
            with open(f'{data_path}/{method}_{nb_steps}_{nb_substeps}_{size}_{rank}.pkl', 'rb') as f:
                dlra_deim_sol = pickle.load(f)
            # Compute the relative error
            final_errors[k+1+len(dlra_names), i*len(ranks)+j] = np.linalg.norm(dlra_deim_sol.Xs[-1] - Xs_ref[-1], 'fro') / np.linalg.norm(Xs_ref[-1], 'fro')
            time_of_computation[k+1+len(dlra_names), i*len(ranks)+j] = np.sum(dlra_deim_sol.computation_time)
#%% Save the final errors and time of computation with pickle
with open(f'{data_path}/processed_data.pkl', 'wb') as f:
    pickle.dump({'final_errors': final_errors, 'time_of_computation': time_of_computation}, f)

#%% Print the final errors in a table with the names as rows, sizes and ranks as columns
import pandas as pd

# Create a DataFrame with the final errors
print('Final errors:')
df = pd.DataFrame(final_errors, index=['ref'] + dlra_legends + dlra_deim_legends, columns=[f'{size}_{rank}' for i, size in enumerate(sizes) for rank in ranks[i]])
# Format the DataFrame to display numbers in scientific notation with 4 significant figures
pd.options.display.float_format = '{:.4e}'.format
# Print the DataFrame
print(df)
# Reset the float format for other DataFrames if needed
pd.reset_option('display.float_format')

# Create a DataFrame with the time of computation
print('Time of computation:')
df_time = pd.DataFrame(time_of_computation, index=['ref'] + dlra_legends + dlra_deim_legends, columns=[f'{size}_{rank}' for i, size in enumerate(sizes) for rank in ranks[i]])
# Print the DataFrame
print(df_time)


# %%
