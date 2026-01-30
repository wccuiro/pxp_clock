import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
# header=None is used because the CSV does not have a label row
df = pd.read_csv('./../rust/std_eigenvalues.csv', header=None)

# 2. Extract the variables based on your data structure:
# Column 0: Gamma (constant)
# Column 1: Omega (the independent variable)
# Columns 2 to the end: Eigenvalues
gamma_const = df.iloc[0, 0]
omega_vals = df.iloc[:, 1]
eigen_vals = df.iloc[:, 2:]

# 3. Create the Plot
plt.figure(figsize=(12, 7))

# We use a colormap to distinguish the 40+ eigenvalue lines
colors = plt.cm.viridis(np.linspace(0, 1, eigen_vals.shape[1]))

# Optional: Sort eigenvalues at each point to see the band structure clearly
# If you want to see the raw trajectories instead, use `eigen_vals.values`
sorted_eigen = np.sort(eigen_vals.values, axis=1)

for i in range(sorted_eigen.shape[1]):
    plt.plot(omega_vals, sorted_eigen[:, i], color=colors[i], linewidth=1.2, alpha=0.8)

# 4. Formatting the plot
plt.xlabel(r'$\Omega$', fontsize=14)
plt.ylabel('Eigenvalues', fontsize=14)
plt.title(r'Eigenvalue Spectrum vs $\Omega$ (at $\gamma = {gamma_const}$)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Display or Save
plt.tight_layout()
plt.show()
# plt.savefig('eigenvalue_plot.png', dpi=300)