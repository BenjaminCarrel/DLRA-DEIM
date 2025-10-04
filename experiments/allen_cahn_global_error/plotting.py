#%% Imports
import matplotlib.pyplot as plt
from shared_parameters import *
from method_definition import dlra_deim_names, dlra_names, dlra_deim_linestyles, dlra_linestyles, dlra_colors, dlra_deim_colors

import os
data_path = f"data/size_{size}/rank_{rank}/time_{t_span[1]}_steps_{nb_steps}/substeps_{nb_substeps}"
if not os.path.exists(f"figures/size_{size}/rank_{rank}/time_{t_span[1]}_steps_{nb_steps}/substeps_{nb_substeps}"):
    os.makedirs(f"figures/size_{size}/rank_{rank}/time_{t_span[1]}_steps_{nb_steps}/substeps_{nb_substeps}")

# Import the errors
import pickle
with open(f'{data_path}/global_errors.pkl', 'rb') as f:
    errors = pickle.load(f)
global_errors_dlra = errors['global_errors_dlra']
global_errors_dlra_deim = errors['global_errors_dlra_deim']
best_error = errors['best_error']

#%% Plotting parameters
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10
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
#%% Plot the global error
fig = plt.figure()
for i in range(len(dlra_names)):
    plt.loglog(stepsizes, global_errors_dlra[:, i], dlra_linestyles[i], color=dlra_colors[i], label=dlra_names[i])
for i in range(len(dlra_deim_names)):
    plt.loglog(stepsizes, global_errors_dlra_deim[:, i], dlra_deim_linestyles[i], color=dlra_deim_colors[i], label=dlra_deim_names[i])
plt.loglog(stepsizes, best_error[-1] * np.ones(len(stepsizes)), 'k--', label=f'Best rank {rank} approximation')
plt.xlabel('Number of substeps')
plt.ylabel('Global error')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Save the figure with timestamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
fig.savefig(f'figures/size_{size}/rank_{rank}/time_{t_span[1]}_steps_{nb_steps}/substeps_{nb_substeps}/global_error_{timestamp}.pdf', bbox_inches='tight')
# %%
