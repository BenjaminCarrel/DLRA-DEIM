#%% Imports
import matplotlib.pyplot as plt
from shared_parameters import *
from method_definition import dlra_deim_names, dlra_names, dlra_deim_linestyles, dlra_linestyles, dlra_deim_legends

import os
if not os.path.exists(f'figures/size_{size}/rank_{rank}/nb_substeps_{nb_substeps}'):
    os.makedirs(f'figures/size_{size}/rank_{rank}/nb_substeps_{nb_substeps}')

import pickle

#%% Load the data
with open(f'{data_path}/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
Xs_ref = data['Xs_ref']
t_eval = data['t_eval'].flatten()
errors = data['errors']
errors_deim = data['errors_deim']
times = data['times']
times_deim = data['times_deim']
final_errors = data['final_errors']
final_errors_deim = data['final_errors_deim']
best_error = data['best_error']

#%% Plotting parameters
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['figure.autolayout'] = True


#%% Print the times in a table
print(f'{"Method":<30}{"Time (s)":<20}{"Rel. error":<20}')
for i, method in enumerate(dlra_names):
    print(f'{dlra_names[i]:<30}{times[i]:<20}{final_errors[i]:<20}')
for i, method in enumerate(dlra_deim_names):
    print(f'{dlra_deim_names[i]:<30}{times_deim[i]:<20}{final_errors_deim[i]:<20}')

#%% Plot the error over time
fig = plt.figure(figsize=(10, 5))
for j, method in enumerate(dlra_names):
    plt.semilogy(t_eval, errors[:, j], dlra_linestyles[j], label=dlra_names[j])
for j, method in enumerate(dlra_deim_names):
    if dlra_deim_names[j] == 'DLRA-sQDEIM' or dlra_deim_names[j] == 'DLRA-OCSS':
        # Plot only a few steps for these methods
        dlra_sqdeim_steps = 2
        plt.semilogy(t_eval[::dlra_sqdeim_steps], errors_deim[::dlra_sqdeim_steps, j], dlra_deim_linestyles[j], label=dlra_deim_legends[j])
    else:
        # Regular plot
        plt.semilogy(t_eval, errors_deim[:, j], dlra_deim_linestyles[j], label=dlra_deim_legends[j])
plt.semilogy(t_eval, best_error, '--', label=f'Best rank {rank} approx.', color='black')
plt.xlabel("Time")
plt.ylabel("Relative error")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Save the figure with timestamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
fig.savefig(f'figures/size_{size}/rank_{rank}/nb_substeps_{nb_substeps}/error_over_time_{timestamp}.pdf', bbox_inches='tight')
# %%
