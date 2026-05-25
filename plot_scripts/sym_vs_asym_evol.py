import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data files
file_sym = '../rust/occupation_time.csv'
file_asym = '../rust/occupation_time_asymmetric.csv'

# Read CSVs (no headers in the raw output)
data_sym = pd.read_csv(file_sym, header=None)
data_asym = pd.read_csv(file_asym, header=None)

# Extract parameters (first 3 columns: gamma_plus, gamma_minus, omega)
params_sym = data_sym.iloc[:, 0:3]

# Extract time series data
series_sym = data_sym.iloc[:, 3:]
series_asym = data_asym.iloc[:, 3:]

# Calculate the number of time steps (3 observables per step)
N = series_sym.shape[1] // 3

# Define time axis (assuming dt=1e-3, t_final=10.0 based on the Rust code)
time = np.linspace(0, 50.0, N)

# Create a figure with 3 subplots sharing the x-axis
# fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Get a color map to differentiate parameter sets
colors = plt.get_cmap('tab10', len(data_sym))

# Loop through each row (parameter set) and plot
for i in range(1,len(data_sym)):
    gp, gm, omega = params_sym.iloc[i]
    label_base = r"$\gamma_+$={gp}, $\gamma_-$={gm}, $\omega$={omega}".format(gp=gp, gm=gm, omega=omega)
    
    # Extract values for Symmetric
    n_sym = series_sym.iloc[i, 0::3].values
    nn_sym = series_sym.iloc[i, 1::3].values
    fid_sym = series_sym.iloc[i, 2::3].values
    
    # Extract values for Asymmetric
    n_asym = series_asym.iloc[i, 0::3].values
    nn_asym = series_asym.iloc[i, 1::3].values
    fid_asym = series_asym.iloc[i, 2::3].values
    
    # # Plot Occupation Number
    # plt.plot(time, n_sym, color=colors(i), linestyle='-', label=f"Uniform: {label_base}")
    # plt.plot(time, n_asym, color=colors(i), linestyle='--', label=f"Staggered: {label_base}")
    
    # plt.ylabel(r'Occupation Number $\langle n \rangle$')
    # plt.title('Comparison: Uniform vs Staggered Dissipation')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    # plt.grid(True)
    
    # plt.show()
    
    # # Plot NN Correlation
    # plt.plot(time, nn_sym, color=colors(i), linestyle='-')
    # plt.plot(time, nn_asym, color=colors(i), linestyle='--')
    
    # plt.ylabel(r'NN Correlation $\langle n_i n_{i+1} \rangle$')
    # plt.grid(True)

    # plt.show()

    # Plot Fidelity
    plt.ylabel('Fidelity (Néel Overlap)')
    plt.xlabel('Time')
    plt.grid(True)

    plt.plot(time, fid_sym, color=colors(i), linestyle='-', label=f"Uniform: {label_base}")
    plt.plot(time, fid_asym, color=colors(i), linestyle='--', label=f"Staggered: {label_base}")

plt.legend()
plt.show()

# Formatting Top Panel (Occupation)

# Formatting Middle Panel (Correlation)

# Formatting Bottom Panel (Fidelity)

# Adjust layout and show/save
# plt.tight_layout()
# plt.savefig('comparison_plot.png', bbox_inches='tight', dpi=300)
# plt.show()