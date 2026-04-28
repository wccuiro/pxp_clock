import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the 12-site data
df_decay = pd.read_csv('../rust/decay.csv', header=None)
df_oee = pd.read_csv('../rust/oee.csv', header=None)

# 2. Parse decay_12 into a long format
decay_rows = []
for idx, row in df_decay.iterrows():
    q_sector, gp, gm, omega = row[0], row[1], row[2], row[3]
    for i in range(4, len(row), 5):
        if pd.isna(row[i]): break
        decay_rows.append([q_sector, gp, gm, omega, row[i], row[i+1], row[i+2], row[i+3]])
        
df_decay_long = pd.DataFrame(decay_rows, columns=['q_sector', 'gp', 'gm', 'omega', 'real_eig', 'imag_eig', 'overlap', 'occupation'])

# 3. Parse oee_12 into a long format
oee_rows = []
for idx, row in df_oee.iterrows():
    q_sector, gp, gm, omega = row[0], row[1], row[2], row[3]
    for i in range(4, len(row), 3):
        if pd.isna(row[i]): break
        oee_rows.append([q_sector, gp, gm, omega, row[i], row[i+1], row[i+2]])
        
df_oee_long = pd.DataFrame(oee_rows, columns=['q_sector', 'gp', 'gm', 'omega', 'real_eig', 'imag_eig', 'entropy'])

# 4. Round eigenvalues to 5 decimal places to ensure a clean inner merge
df_decay_long['real_eig_round'] = df_decay_long['real_eig'].round(5)
df_decay_long['imag_eig_round'] = df_decay_long['imag_eig'].round(5)
df_oee_long['real_eig_round'] = df_oee_long['real_eig'].round(5)
df_oee_long['imag_eig_round'] = df_oee_long['imag_eig'].round(5)

# 5. Merge the datasets based on the parameters, sector, and matching eigenvalues
df_merged = pd.merge(df_decay_long, df_oee_long, 
                     on=['q_sector', 'gp', 'gm', 'omega', 'real_eig_round', 'imag_eig_round'], 
                     how='inner')
df_merged_clean = df_merged.drop_duplicates(subset=['q_sector', 'gp', 'gm', 'omega', 'real_eig_x', 'imag_eig_x'])

# -------------------------------------------------------------------------
# SETUP: DYNAMIC EXTRACTION OF PARAMETERS AND SECTORS
# -------------------------------------------------------------------------
params = df_merged_clean[['gp', 'gm']].drop_duplicates().sort_values(by=['gp', 'gm']).values
sectors = sorted(df_merged_clean['q_sector'].unique())

marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
marker_map = {sec: marker_list[i % len(marker_list)] for i, sec in enumerate(sectors)}

# Calculate grid size dynamically
n_plots = len(params)
cols = 2
rows = int(np.ceil(n_plots / cols))

# Global color limits
global_ent_min, global_ent_max = df_merged_clean['entropy'].min(), df_merged_clean['entropy'].max()
decay_rate = -df_merged_clean['real_eig_x']
global_dec_min, global_dec_max = decay_rate.min(), decay_rate.max()
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# 6. Plot Overlap vs Im[lambda] colored by OEE
# -------------------------------------------------------------------------
fig1, axes1 = plt.subplots(rows, cols, figsize=(14, 5 * rows), constrained_layout=True)
axes1_flat = np.array(axes1).flatten() if n_plots > 1 else np.array([axes1])
active_axes1 = []

for i, (gp, gm) in enumerate(params):
    ax = axes1_flat[i]
    active_axes1.append(ax)
    subset = df_merged_clean[(df_merged_clean['gp'] == gp) & (df_merged_clean['gm'] == gm)]
    
    for sector in sectors:
        sec_data = subset[subset['q_sector'] == sector]
        if sec_data.empty: continue
            
        sc1 = ax.scatter(sec_data['imag_eig_x'], sec_data['overlap'], 
                         c=sec_data['entropy'], cmap='Reds', vmin=global_ent_min, vmax=global_ent_max,
                         marker=marker_map[sector], label=f'Sector {int(sector)}',
                         s=40, alpha=0.9, edgecolors='gray', linewidth=0.5)
    
    ax.set_title(f'γ+={gp:.3f}, γ-={gm:.3f}')
    ax.set_xlabel('Imaginary Part Im[λ] (Oscillation)')
    ax.set_ylabel('Overlap')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f0f0f0')
    ax.legend(loc='best', fontsize='small')

# Clean up empty subplots
for j in range(n_plots, len(axes1_flat)):
    fig1.delaxes(axes1_flat[j])

# Attach colorbar explicitly to the right of the active axes
cbar1 = fig1.colorbar(sc1, ax=active_axes1, location='right', shrink=0.8, pad=0.02)
cbar1.set_label('Operator Entanglement Entropy (OEE)')
plt.show()


# -------------------------------------------------------------------------
# 7. Plot Overlap vs OEE colored by Decay Rate
# -------------------------------------------------------------------------
fig2, axes2 = plt.subplots(rows, cols, figsize=(14, 5 * rows), constrained_layout=True)
axes2_flat = np.array(axes2).flatten() if n_plots > 1 else np.array([axes2])
active_axes2 = []

for i, (gp, gm) in enumerate(params):
    ax = axes2_flat[i]
    active_axes2.append(ax)
    subset = df_merged_clean[(df_merged_clean['gp'] == gp) & (df_merged_clean['gm'] == gm)]
    
    for sector in sectors:
        sec_data = subset[subset['q_sector'] == sector]
        if sec_data.empty: continue
            
        sc2 = ax.scatter(sec_data['entropy'], sec_data['overlap'], 
                         c=-sec_data['real_eig_x'], cmap='Reds', vmin=global_dec_min, vmax=global_dec_max,
                         marker=marker_map[sector], label=f'Sector {int(sector)}',
                         s=40, alpha=0.9, edgecolors='gray', linewidth=0.5)
    
    ax.set_title(f'γ+={gp:.3f}, γ-={gm:.3f}')
    ax.set_xlabel('Operator Entanglement Entropy (OEE)')
    ax.set_ylabel('Overlap')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f0f0f0')
    ax.legend(loc='best', fontsize='small')

# Clean up empty subplots
for j in range(n_plots, len(axes2_flat)):
    fig2.delaxes(axes2_flat[j])

# Attach colorbar explicitly to the right of the active axes
cbar2 = fig2.colorbar(sc2, ax=active_axes2, location='right', shrink=0.8, pad=0.02)
cbar2.set_label('Decay Rate (-Re[λ])')
plt.show()