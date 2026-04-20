import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the 12-site data
df_decay = pd.read_csv('../rust/decay_12.csv', header=None)
df_oee = pd.read_csv('../rust/oee_12.csv', header=None)

# 2. Parse decay_12 into a long format
decay_rows = []
for idx, row in df_decay.iterrows():
    gp, gm, omega = row[0], row[1], row[2]
    # Skip the first 3 columns, then jump in steps of 5
    for i in range(3, len(row), 5):
        if pd.isna(row[i]): break
        decay_rows.append([gp, gm, omega, row[i], row[i+1], row[i+2], row[i+3]])
df_decay_long = pd.DataFrame(decay_rows, columns=['gp', 'gm', 'omega', 'real_eig', 'imag_eig', 'overlap', 'occupation'])

# 3. Parse oee_12 into a long format
oee_rows = []
for idx, row in df_oee.iterrows():
    gp, gm, omega = row[0], row[1], row[2]
    # Skip the first 3 columns, then jump in steps of 3
    for i in range(3, len(row), 3):
        if pd.isna(row[i]): break
        oee_rows.append([gp, gm, omega, row[i], row[i+1], row[i+2]])
df_oee_long = pd.DataFrame(oee_rows, columns=['gp', 'gm', 'omega', 'real_eig', 'imag_eig', 'entropy'])

# 4. Round eigenvalues to 5 decimal places to ensure a clean inner merge
df_decay_long['real_eig_round'] = df_decay_long['real_eig'].round(5)
df_decay_long['imag_eig_round'] = df_decay_long['imag_eig'].round(5)
df_oee_long['real_eig_round'] = df_oee_long['real_eig'].round(5)
df_oee_long['imag_eig_round'] = df_oee_long['imag_eig'].round(5)

# 5. Merge the datasets based on the parameters and matching eigenvalues
df_merged = pd.merge(df_decay_long, df_oee_long, on=['gp', 'gm', 'omega', 'real_eig_round', 'imag_eig_round'], how='inner')
df_merged_clean = df_merged.drop_duplicates(subset=['gp', 'gm', 'omega', 'real_eig_x', 'imag_eig_x'])

params = [[0.001, 0.001], [0.001, 0.2], [0.2, 0.001], [0.2, 0.2]]

# 6. Plot Overlap vs Im[lambda] colored by OEE
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
axes1 = axes1.flatten()

for i, (gp, gm) in enumerate(params):
    ax = axes1[i]
    subset = df_merged_clean[(df_merged_clean['gp'] == gp) & (df_merged_clean['gm'] == gm)]
    
    sc = ax.scatter(subset['imag_eig_x'], subset['overlap'], 
                    c=subset['entropy'], cmap='Reds', s=25, alpha=0.9, edgecolors='none')
    
    ax.set_title(f'γ+={gp:.3f}, γ-={gm:.3f}')
    ax.set_xlabel('Imaginary Part Im[λ] (Oscillation)')
    ax.set_ylabel('Overlap')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f0f0f0') # Grey background to see white dots

cbar1 = fig1.colorbar(sc, ax=axes1.tolist())
cbar1.set_label('Operator Entanglement Entropy (OEE)')
plt.show()
# plt.savefig('overlap_vs_imag_reds.png', bbox_inches='tight')

# 7. Plot Overlap vs OEE colored by Decay Rate
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
axes2 = axes2.flatten()

for i, (gp, gm) in enumerate(params):
    ax = axes2[i]
    subset = df_merged_clean[(df_merged_clean['gp'] == gp) & (df_merged_clean['gm'] == gm)]
    
    sc = ax.scatter(subset['entropy'], subset['overlap'], 
                    c=-subset['real_eig_x'], cmap='Reds', s=25, alpha=0.9, edgecolors='none')
    
    ax.set_title(f'γ+={gp:.3f}, γ-={gm:.3f}')
    ax.set_xlabel('Operator Entanglement Entropy (OEE)')
    ax.set_ylabel('Overlap')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f0f0f0')

cbar2 = fig2.colorbar(sc, ax=axes2.tolist())
cbar2.set_label('Decay Rate (-Re[λ])')
# plt.savefig('overlap_vs_oee_reds.png', bbox_inches='tight')
plt.show()