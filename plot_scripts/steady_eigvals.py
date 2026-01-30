import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('../rust/std_eigenvalues.csv', header=None)
omega = df.iloc[:, 1].values
g_val = df.iloc[0, 0]

# Define column indices for Eigenvalues and Occupations
ev_cols = list(range(2, df.shape[1], 2))
occ_cols = list(range(3, df.shape[1], 2))

# Extract data into numpy arrays for sorting
ev_data = df.iloc[:, ev_cols].values
occ_data = df.iloc[:, occ_cols].values

# Sort eigenvalues at each Omega, and rearrange occupations to match
sort_idx = np.argsort(ev_data, axis=1)
sorted_evs = np.take_along_axis(ev_data, sort_idx, axis=1)
sorted_occs = np.take_along_axis(occ_data, sort_idx, axis=1)

# Plotting
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('cool') # High contrast colormap

# Loop through sorted levels to plot
for i in range(sorted_evs.shape[1]):
    sc = plt.scatter(omega, sorted_evs[:, i], c=sorted_occs[:, i], 
                     cmap=cmap, s=6, vmin=0, vmax=0.5, edgecolors='none')

# Add Colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Occupation', fontsize=12)

# Labels and Formatting
plt.xlabel(r'$\Omega$', fontsize=14)
plt.ylabel('Eigenvalues', fontsize=14)
plt.title(r'Eigenvalue Spectrum vs $\Omega$ (g = {g_val})'.format(g_val=g_val), fontsize=16)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlim(0, 30)
plt.ylim(0, 1)
plt.tight_layout()

plt.show()