import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 0. Configuration / Controls
# ==========================================
# Set the maximum values you want to include in the plots
MAX_OMEGA = 5.0  # Will plot omega up to 5.0
MAX_G = 2.0      # Will plot g up to 2.0

# 1. Load Data
# ---------------------------------------------------------
df = pd.read_csv('../rust/occupation_8.csv')

# --- FILTERING STEP ---
# Keep only rows where omega and g are within your desired limits
df = df[(df['omega'] <= MAX_OMEGA) & (df['g'] <= MAX_G)]
# ----------------------

# 2. Calculate Renormalized Deviation
# The incoherent theory predicts: <nn> = [(3g+1)/g]*<n> - 1
# Delta_renorm = g * <nn>_observed - (g * <nn>_theory)

df['deviation_renorm'] = df['g'] * df['nn'] - ((3 * df['g'] + 1) * df['n'] - df['g'])

# Calculate the theoretical prediction for the scatter plot
df['nn_theory'] = ((3 * df['g'] + 1) / df['g']) * df['n'] - 1

# 3. Plotting
# ---------------------------------------------------------

# --- Plot A: Line Cuts (Verification of Zero Limit) ---
plt.figure(figsize=(10, 6))
unique_g = np.sort(df['g'].unique())

# Dynamic selection: Pick 6 evenly spaced values from your FILTERED range
# This is better than unique_g[:6], which only shows the bottom values
indices = np.linspace(0, len(unique_g) - 1, 6, dtype=int)
selected_g = unique_g[indices]

for g_val in selected_g:
    subset = df[df['g'] == g_val].sort_values(by='omega')
    plt.plot(subset['omega'], subset['deviation_renorm'], 
             label=f'g={g_val:.2f}', marker='o', markersize=3)

# Highlight the origin to prove the fix
plt.scatter([0], [0], color='black', s=50, zorder=10, label='Limit $\Omega=0$')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.xlabel(r'Scaled Drive $\Omega / \gamma_-$')
plt.ylabel(r'Renormalized Deviation $g \Delta$')
plt.title(f'Breakdown of Incoherent Relation (Range: $g \leq {MAX_G}$)')
plt.legend(title='Ratio $g$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deviation_renorm_lines.png')
plt.show()

# --- Plot A1: Line Cuts (Verification of Zero Limit) ---
plt.figure(figsize=(10, 6))
unique_om = np.sort(df['omega'].unique())

# Dynamic selection: Pick 6 evenly spaced values from your FILTERED range
# This is better than unique_g[:6], which only shows the bottom values
indices = np.linspace(0, len(unique_om) - 1, 6, dtype=int)
selected_om = unique_om[indices]

for om_val in selected_om:
    subset = df[df['omega'] == om_val].sort_values(by='g')
    plt.plot(subset['g'], subset['deviation_renorm'], 
             label=f'$\Omega$={om_val:.2f}', marker='o', markersize=3)

# Highlight the origin to prove the fix
# plt.scatter([0], [0], color='black', s=50, zorder=10, label='Limit $\Omega=0$')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.xlabel(r'Scaled Drive $\gamma_+ / \gamma_-$')
plt.ylabel(r'Renormalized Deviation $g \Delta$')
plt.title(f'Breakdown of Incoherent Relation (Range: $\Omega \leq {MAX_OMEGA}$)')
plt.legend(title=r'Ratio $\Omega$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deviation_renorm_lines_om.png')
plt.show()

# --- Plot B: Phase Diagram (Heatmap) ---
pivot_renorm = df.pivot(index='g', columns='omega', values='deviation_renorm')

plt.figure(figsize=(10, 8))
# Format tick labels
pivot_renorm.index = pivot_renorm.index.map('{:.2f}'.format)
pivot_renorm.columns = pivot_renorm.columns.map('{:.2f}'.format)

# center=0 ensures white corresponds to the incoherent theory
ax = sns.heatmap(pivot_renorm, cmap='coolwarm', center=0, 
                 xticklabels=5, yticklabels=15, cbar_kws={'label': r'$g \Delta$'})
ax.invert_yaxis() # Put small g at the bottom
plt.title(f'Renormalized Deviation (Filtered: $\Omega \leq {MAX_OMEGA}, g \leq {MAX_G}$)')
plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel(r'$g = \gamma_+ / \gamma_-$')
plt.tight_layout()
plt.savefig('deviation_renorm_heatmap.png')
plt.show()

# --- Plot C: Theory Check (Scatter) ---
plt.figure(figsize=(8, 6))

# Plot the scatter data
sc = plt.scatter(df['nn_theory'], df['nn'], c=df['omega'], 
                 cmap='viridis', alpha=0.7, s=20)

# 1. Set both axes to log scale
plt.xscale('log')
plt.yscale('log')

# 2. Update the identity line
plt.plot([1e-4, 1], [1e-4, 1], 'k--', label='Incoherent Theory')

plt.colorbar(sc, label=r'Drive $\Omega$')
plt.xlabel(r'Theoretical Prediction $\langle n_{i-1}n_{i+1} \rangle$')
plt.ylabel(r'Observed Correlation $\langle n_{i-1}n_{i+1} \rangle$')
plt.title('Correlation Check: Observed vs Theory (Log-Log)')
plt.legend()
plt.grid(True, alpha=0.3, which="both") 
plt.tight_layout()
plt.savefig('correlation_scatter_checked.png')
plt.show()