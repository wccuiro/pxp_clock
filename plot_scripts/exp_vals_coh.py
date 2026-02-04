import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. Configuration / Controls
# ==========================================
# Set the maximum values you want to include in the plots
MAX_OMEGA = 5.0  # Will plot omega up to 5.0
MAX_G = 2.0      # Will plot g up to 2.0

# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv('../rust/occupation_8.csv')

# --- FILTERING STEP ---
# Keep only rows where omega and g are within your desired limits
df = df[(df['omega'] <= MAX_OMEGA) & (df['g'] <= MAX_G)]
# ----------------------

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
plot_data = pivot_table.copy()

# Format labels
plot_data.index = plot_data.index.map('{:.2f}'.format)
plot_data.columns = plot_data.columns.map('{:.2f}'.format)

# Plot using the formatted data
# Note: I removed ax.set_xlim because the data is already filtered!
ax = sns.heatmap(plot_data, cmap='viridis', 
                 xticklabels=5, yticklabels=15, vmin=0, vmax=0.5)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0) 
ax.invert_yaxis() 

plt.title(f'Phase Diagram (Filtered: $\Omega \leq {MAX_OMEGA}, g \leq {MAX_G}$)')
plt.xlabel(r'Scaled Drive $\Omega / \gamma_-$')
plt.ylabel(r'Dissipation Ratio $g = \gamma_+ / \gamma_-$')
plt.tight_layout()
plt.savefig('1_heatmap_n.png')
plt.show()

# --- Plot B: Line Cuts (n vs Omega for fixed g) ---
plt.figure(figsize=(10, 6))
unique_g = df['g'].unique()
# Select 6 evenly spaced values from the filtered range
selected_g = unique_g[np.linspace(0, len(unique_g)-1, 6, dtype=int)]

for g_val in selected_g:
    subset = df[df['g'] == g_val]
    plt.plot(subset['omega'], subset['n'], label=f'g={g_val:.2f}', marker='.')

plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel('Occupation $n$')
# Removed manual xlim, it now scales automatically to your data
plt.title(r'Occupation vs Drive $\Omega$ (Fixed Dissipation Ratio)')
plt.legend(title='Ratio $g$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_lineplot_n_vs_omega.png')
plt.show()

# --- Plot C: Line Cuts (n vs g for fixed Omega) ---
plt.figure(figsize=(10, 6))
unique_omega = df['omega'].unique()
# Select a few evenly spaced values from the filtered range
step_size = max(1, len(unique_omega) // 6) # Ensure we get about 6 lines
selected_omega = unique_omega[::step_size]

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
grad_g, grad_omega = np.gradient(Z, g_vals, omega_vals)
susceptibility = np.sqrt(grad_g**2 + grad_omega**2)

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, susceptibility, cmap='inferno', shading='auto')
plt.colorbar(label=r'Gradient Magnitude $|\nabla n|$')
plt.title('Susceptibility Map')
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
ax.view_init(elev=30, azim=225)
plt.tight_layout()
plt.savefig('5_surface_plot.png')
plt.show()

# --- Plot F: Contour Plot ---
plt.figure(figsize=(10, 8))
cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis', vmin=0, vmax=0.5)
cbar = plt.colorbar(cp)
lines = plt.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
plt.clabel(lines, inline=True, fontsize=8)

plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel(r'$g = \gamma_+ / \gamma_-$')
plt.title('Contour Map of Occupation')
plt.tight_layout()
plt.savefig('6_contour_plot.png')
plt.show()