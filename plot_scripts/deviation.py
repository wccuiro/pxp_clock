import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load Data
df = pd.read_csv('../rust/occupation.csv')

# 2. Calculate Renormalized Deviation
# The incoherent theory predicts: <nn> = [(3g+1)/g]*<n> - 1
# To avoid dividing by small g, we rearrange the deviation formula:
# Delta_renorm = g * <nn>_observed - (g * <nn>_theory)
# Delta_renorm = g * <nn> - [(3g+1)<n> - g]

df['deviation_renorm'] = df['g'] * df['nn'] - ((3 * df['g'] + 1) * df['n'] - df['g'])

# Calculate the theoretical prediction for the scatter plot
df['nn_theory'] = ((3 * df['g'] + 1) / df['g']) * df['n'] - 1

# 3. Plotting
# ---------------------------------------------------------

# --- Plot A: Line Cuts (Verification of Zero Limit) ---
plt.figure(figsize=(10, 6))
unique_g = np.sort(df['g'].unique())
# Select 6 representative values of g
# selected_g = unique_g[np.linspace(0, len(unique_g)-1, 15, dtype=int)]
selected_g = unique_g[:6]

for g_val in selected_g:
    subset = df[df['g'] == g_val].sort_values(by='omega')
    plt.plot(subset['omega'], subset['deviation_renorm'], 
             label=f'g={g_val:.3f}', marker='o', markersize=3)

# Highlight the origin to prove the fix
plt.scatter([0], [0], color='black', s=50, zorder=10, label='Limit $\Omega=0$')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.xlabel(r'Scaled Drive $\Omega / \gamma_-$')
plt.ylabel(r'Renormalized Deviation $g \Delta$')
plt.title(r'Breakdown of Incoherent Relation (Starts at 0)')
plt.legend(title='Ratio $g$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deviation_renorm_lines.png')
plt.show()

# --- Plot B: Phase Diagram (Heatmap) ---
pivot_renorm = df.pivot(index='g', columns='omega', values='deviation_renorm')

plt.figure(figsize=(10, 8))
# center=0 ensures white corresponds to the incoherent theory
ax = sns.heatmap(pivot_renorm, cmap='coolwarm', center=0, 
                 xticklabels=5, yticklabels=5, cbar_kws={'label': r'g \Delta$'})
ax.invert_yaxis() # Put small g at the bottom
plt.title(r'Renormalized Deviation $g \Delta$ (Quantum Correlations)')
plt.xlabel(r'$\Omega / \gamma_-$')
plt.ylabel(r'$g = \gamma_+ / \gamma_-$')
plt.tight_layout()
plt.savefig('deviation_renorm_heatmap.png')
plt.show()

# --- Plot C: Theory Check (Scatter) ---
# Plot the scatter data
sc = plt.scatter(df['nn_theory'], df['nn'], c=df['omega'], 
                 cmap='viridis', alpha=0.7, s=20)

# 1. Set both axes to log scale
plt.xscale('log')
plt.yscale('log')

# 2. Update the identity line
# We cannot start at 0 in log scale. We start at a small number (e.g. 1e-4) 
# or the minimum non-zero value in your dataset.
plt.plot([1e-4, 1], [1e-4, 1], 'k--', label='Incoherent Theory ($\Omega=0$)')

plt.colorbar(sc, label=r'Drive $\Omega$')
plt.xlabel(r'Theoretical Prediction $\langle n_{i-1}n_{i+1} \rangle$')
plt.ylabel(r'Observed Correlation $\langle n_{i-1}n_{i+1} \rangle$')
plt.title('Correlation Check: Observed vs Theory (Log-Log)')
plt.legend()
plt.grid(True, alpha=0.3, which="both") # 'both' adds gridlines for minor log ticks
plt.tight_layout()
plt.savefig('correlation_scatter_checked.png')
plt.show()