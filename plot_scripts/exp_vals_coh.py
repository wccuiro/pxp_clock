import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv('../rust/occupation.csv')

# Ensure data is sorted for consistent plotting
df = df.sort_values(by=['g', 'omega'])

# Create a Pivot Table (Grid) for 2D/3D plotting
# Rows (index) = g, Columns = omega, Values = n
pivot_table = df.pivot(index='g', columns='omega', values='n')

# Extract arrays for plotting
g_vals = pivot_table.index.values
omega_vals = pivot_table.columns.values
X, Y = np.meshgrid(omega_vals, g_vals)  # X = Omega/gamma-, Y = g
Z = pivot_table.values                  # Z = Occupation n

# 2. Plotting
# ---------------------------------------------------------

# --- Plot A: 2D Heatmap (Phase Diagram) ---
plt.figure(figsize=(10, 8))
# Note: seaborn heatmap index 0 is at the top by default, so we invert y-axis
ax = sns.heatmap(pivot_table, cmap='viridis', 
                 xticklabels=2, yticklabels=2) # Adjust tick frequency if crowded
ax.invert_yaxis() 
plt.title('Phase Diagram: Occupation $n$')
plt.xlabel(r'Scaled Drive $\Omega / \gamma_-$')
plt.ylabel(r'Dissipation Ratio $g = \gamma_+ / \gamma_-$')
plt.tight_layout()
plt.savefig('1_heatmap_n.png')
plt.show()

# --- Plot B: Line Cuts (n vs Omega for fixed g) ---
plt.figure(figsize=(10, 6))
# Select a few evenly spaced values of g to plot
unique_g = df['g'].unique()
selected_g = unique_g[np.linspace(0, len(unique_g)-1, 6, dtype=int)]

for g_val in selected_g:
    subset = df[df['g'] == g_val]
    plt.plot(subset['omega'], subset['n'], label=f'g={g_val:.2f}', marker='.')

plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel('Occupation $n$')
plt.title(r'Occupation vs Drive $\Omega$ (Fixed Dissipation Ratio)')
plt.legend(title='Ratio $g$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_lineplot_n_vs_omega.png')
plt.show()

# --- Plot C: Line Cuts (n vs g for fixed Omega) ---
plt.figure(figsize=(10, 6))
# Select a few evenly spaced values of Omega to plot
unique_omega = df['omega'].unique()
selected_omega = unique_omega[np.linspace(0, len(unique_omega)-1, 6, dtype=int)]

for omega_val in selected_omega:
    subset = df[df['omega'] == omega_val]
    plt.plot(subset['g'], subset['n'], label=f'$\Omega$={omega_val:.2f}', marker='.')

plt.xlabel(r'$g = \gamma_+ / \gamma_-$')
plt.ylabel('Occupation $n$')
plt.title(r'Occupation vs Dissipation Ratio $g$ (Fixed Drive)')
plt.legend(title=r'Drive $\Omega$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_lineplot_n_vs_g.png')
plt.show()

# --- Plot D: Gradient / Susceptibility Map ---
# Calculate the gradient magnitude to find phase transitions
grad_g, grad_omega = np.gradient(Z, g_vals, omega_vals)
susceptibility = np.sqrt(grad_g**2 + grad_omega**2)

plt.figure(figsize=(10, 8))
# We use pcolormesh here for better axis control with raw values
plt.pcolormesh(X, Y, susceptibility, cmap='inferno', shading='auto')
plt.colorbar(label=r'Gradient Magnitude $|\nabla n|$')
plt.title('Susceptibility Map (Phase Boundaries)')
plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel(r'$g = \gamma_+ / \gamma_-$')
plt.tight_layout()
plt.savefig('4_gradient_susceptibility.png')
plt.show()

# --- Plot E: 3D Surface Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_xlabel(r'$\Omega / \gamma_-$')
ax.set_ylabel(r'$g = \gamma_+ / \gamma_-$')
ax.set_zlabel('Occupation $n$')
ax.set_title('3D Surface of Occupation')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Occupation $n$')
ax.view_init(elev=30, azim=225) # Adjust camera angle
plt.tight_layout()
plt.savefig('5_surface_plot.png')
plt.show()

# --- Plot F: Contour Plot ---
plt.figure(figsize=(10, 8))
# Filled contours
cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = plt.colorbar(cp)
cbar.set_label('Occupation $n$')
# Line contours
lines = plt.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
plt.clabel(lines, inline=True, fontsize=8)

plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel(r'$g = \gamma_+ / \gamma_-$')
plt.title('Contour Map of Occupation')
plt.tight_layout()
plt.savefig('6_contour_plot.png')
plt.show()