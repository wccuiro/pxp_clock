import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the datasets
df_full = pd.read_csv('../python/evals_8.csv', header=None)
df_part = pd.read_csv('../rust/eigenvalues.csv', header=None)

# Process the full spectrum (already in pairs)
full_real = df_full[0]
full_imag = df_full[1]

# Process the partial spectrum
part_vals = df_part.iloc[0].values
part_vals_pi = df_part.iloc[1].values


# Drop the first 4 metadata/parameter values to allow reshaping into pairs
if len(part_vals[1:]) % 2 != 0:
    part_vals = part_vals[4:]

if len(part_vals_pi[1:]) % 2 != 0:
    part_vals_pi = part_vals_pi[4:]


part_complex = part_vals.reshape(-1, 2)
part_real = part_complex[:, 0]
part_imag = part_complex[:, 1]

part_complex_pi = part_vals_pi.reshape(-1, 2)
part_real_pi = part_complex_pi[:, 0]
part_imag_pi = part_complex_pi[:, 1]

# Generate the plot
plt.figure(figsize=(10, 8))

# Plot full spectrum as the background base
plt.scatter(full_real, full_imag, label='Full Spectrum', 
            color='#C7D3D4', alpha=1.0, s=30, edgecolors='none')

# Plot partial spectrum over it
plt.scatter(part_real, part_imag, label='Q=0 Sector', 
            color='#603F83', alpha=1.0, s=30, marker="x", edgecolors='none')

plt.scatter(part_real_pi, part_imag_pi, label='Q=π Sector', 
            color='#1F77B4', alpha=1.0, s=30, marker="x", edgecolors='none')

plt.title(r'$\gamma_+ = 0.5, \gamma_- = 0.25, \Omega = 1.0$, L=8')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
# plt.savefig('eigenvalues_plot.png', dpi=900, bbox_inches='tight')