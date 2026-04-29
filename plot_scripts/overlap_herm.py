import pandas as pd
import numpy as np
# import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def parse_original_data(filepath):
    """Parses the original decay.csv file into a structured DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
            # Skip empty or incomplete lines
            if len(parts) < 4:
                continue
                
            q_sector = float(parts[0])
            gp = float(parts[1])
            gm = float(parts[2])
            omega = float(parts[3])
            
            # Parse repeating chunks of 7
            idx = 4
            while idx + 6 < len(parts):
                real_eval = float(parts[idx])
                abs_imag = float(parts[idx+1])
                c_k = float(parts[idx+2])
                s_k = float(parts[idx+3])
                occ_c = float(parts[idx+4])
                occ_s = float(parts[idx+5])
                block_size = float(parts[idx+6])
                
                data.append({
                    'q_sector': q_sector,
                    'gp': gp,
                    'gm': gm,
                    'omega': omega,
                    'real_eval': real_eval,
                    'abs_imag': abs_imag,
                    'c_k': c_k,
                    's_k': s_k,
                    'occ_c': occ_c,
                    'occ_s': occ_s,
                    'block_size': block_size
                })
                idx += 7
                
    df = pd.DataFrame(data)
    
    # Calculate physical magnitudes
    df['R_k'] = np.sqrt(df['c_k']**2 + df['s_k']**2)
    df['Occ_R'] = np.sqrt(df['occ_c']**2 + df['occ_s']**2)
    df['phase_c'] = np.arctan2(df['s_k'], df['c_k'])
    df['phase_occ'] = np.arctan2(df['occ_s'], df['occ_c'])
    
    return df

# 1. Parse the original file
df = parse_original_data('../rust/decay.csv')

# 2. Find unique parameter sets
unique_params = df[['gp', 'gm', 'omega']].drop_duplicates()

# 3. Loop through each parameter set and generate the requested plots
for idx, row in unique_params.iterrows():
    gp, gm, omega = row['gp'], row['gm'], row['omega']
    
    # Filter data for this specific parameter set
    subset = df[(df['gp'] == gp) & (df['gm'] == gm) & (df['omega'] == omega)].copy()
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle(f'Parameters: gp={gp}, gm={gm}, omega={omega}', fontsize=16, y=0.98)
    
    # === Row 0: Overlap Amplitudes ===
    
    subset['abs_c_k'] = np.abs(subset['c_k'])
    subset['abs_s_k'] = np.abs(subset['s_k'])
    
    # Overlap C_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='abs_c_k', hue='q_sector', 
                    palette='Set1', ax=axes[0, 0], edgecolor='k', alpha=0.7)
    axes[0, 0].set_title('Overlap $C_k$ vs Real Part')
    axes[0, 0].set_xlabel('Real Part (Decay Rate)')
    
    # Overlap C_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='abs_c_k', hue='q_sector', 
                    palette='Set1', ax=axes[1, 0], edgecolor='k', alpha=0.7)
    deltaE = 1.33  # Assuming this is the fundamental frequency spacing
    x_ticks = [i * deltaE for i in range(9)]
    axes[1, 0].set_xticks(x_ticks)
    axes[1, 0].set_xticklabels(['0', r'$\Delta E$', r'$2\Delta E$', r'$3\Delta E$', r'$4\Delta E$', r'$5\Delta E$', r'$6\Delta E$', r'$7\Delta E$', r'$8\Delta E$'])
    axes[1, 0].set_title('Overlap $C_k$ vs Abs Imaginary Part')
    axes[1, 0].set_xlabel('Absolute Imaginary Part (Freq)')
    for x_val in x_ticks:
        axes[1, 0].axvline(x=x_val, color='gray', linestyle='--', alpha=0.5, zorder=1)
    

    # Overlap S_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='abs_s_k', hue='q_sector', 
                    palette='Set1', ax=axes[0, 1], edgecolor='k', alpha=0.7)
    axes[0, 1].set_title('Overlap $S_k$ vs Real Part')
    axes[0, 1].set_xlabel('Real Part (Decay Rate)')

    # Overlap S_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='abs_s_k', hue='q_sector', 
                    palette='Set1', ax=axes[1, 1], edgecolor='k', alpha=0.7)
    axes[1, 1].set_xticks(x_ticks)
    axes[1, 1].set_xticklabels(['0', r'$\Delta E$', r'$2\Delta E$', r'$3\Delta E$', r'$4\Delta E$', r'$5\Delta E$', r'$6\Delta E$', r'$7\Delta E$', r'$8\Delta E$'])
    axes[1, 1].set_title('Overlap $S_k$ vs Abs Imaginary Part')
    axes[1, 1].set_xlabel('Absolute Imaginary Part (Freq)')
    for x_val in x_ticks:
        axes[1, 1].axvline(x=x_val, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Overlap R_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='R_k', hue='q_sector', 
                    palette='Set1', ax=axes[0, 2], edgecolor='k', alpha=0.7)
    axes[0, 2].set_title('Overlap $R_k$ vs Real Part')
    axes[0, 2].set_xlabel('Real Part (Decay Rate)')
    
    # Overlap R_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='R_k', hue='q_sector', 
                    palette='Set1', ax=axes[1, 2], edgecolor='k', alpha=0.7)
    axes[1, 2].set_xticks(x_ticks)
    axes[1, 2].set_xticklabels(['0', r'$\Delta E$', r'$2\Delta E$', r'$3\Delta E$', r'$4\Delta E$', r'$5\Delta E$', r'$6\Delta E$', r'$7\Delta E$', r'$8\Delta E$'])
    axes[1, 2].set_title('Overlap $R_k$ vs Abs Imaginary Part')
    axes[1, 2].set_xlabel('Absolute Imaginary Part (Freq)')
    for x_val in x_ticks:
        axes[1, 2].axvline(x=x_val, color='gray', linestyle='--', alpha=0.5, zorder=1)

    # axes[2, 0].remove()
    # axes[2, 0] = fig.add_subplot(3, 4, 9, projection='polar') # 9 is the 1-based index for row 3, col 1
    # sc1 = axes[2, 0].scatter(subset['phase_c'], subset['R_k'], c=subset['abs_imag'], cmap='viridis', alpha=0.8, edgecolor='k', s=40)
    # axes[2, 0].set_title("Polar: Overlap $R_k$ & $\phi_k$", pad=15)
    # plt.colorbar(sc1, ax=axes[2, 0], label='Frequency ($|\omega|$)')

    # Real vs Phase 
    sns.scatterplot(data=subset, x='phase_c', y='real_eval', hue='q_sector', 
                    palette='Set1', ax=axes[2, 0], edgecolor='k', alpha=0.7)
    axes[2, 0].set_title(r'Real Part vs Overlap Phase $\phi_k$')
    axes[2, 0].set_xlabel(r'Phase $\phi_k$ (rad)')

    # Imaginary vs Phase 
    sns.scatterplot(data=subset, x='phase_c', y='abs_imag', hue='q_sector', 
                    palette='Set1', ax=axes[2, 1], edgecolor='k', alpha=0.7)
    axes[2, 1].set_title(r'Abs Imaginary Part vs Overlap Phase $\phi_k$')
    axes[2, 1].set_xlabel(r'Phase $\phi_k$ (rad)')

    # Overlap R_k vs Phase    
    sns.scatterplot(data=subset, x='phase_c', y='R_k', hue='q_sector', 
                    palette='Set1', ax=axes[2, 2], edgecolor='k', alpha=0.7)
    axes[2, 2].set_title(r'Overlap $R_k$ vs Overlap Phase $\phi_k$')
    axes[2, 2].set_xlabel(r'Phase $\phi_k$ (rad)')

    
    # Apply grid styling to all subplots
    for ax in axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Adjust layout to prevent overlap and save the figure
    plt.tight_layout()
    plt.show()

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Parameters: gp={gp}, gm={gm}, omega={omega}', fontsize=16, y=0.98)

    # === Row 1: Occupation Traces ===

    subset['abs_occ_c'] = np.abs(subset['occ_c'])
    subset['abs_occ_s'] = np.abs(subset['occ_s'])

    # Occupation C_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='abs_occ_c', hue='q_sector', 
                    palette='Set2', ax=axes[0, 0], edgecolor='k', alpha=0.7)
    axes[0, 0].set_title('Occupation $C_k$ vs Real Part')
    axes[0, 0].set_xlabel('Real Part (Decay Rate)')
    
    # Occupation C_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='abs_occ_c', hue='q_sector', 
                    palette='Set2', ax=axes[1, 0], edgecolor='k', alpha=0.7)
    axes[1, 0].set_title('Occupation $C_k$ vs Imaginary Part')
    axes[1, 0].set_xlabel('Absolute Imaginary Part (Freq)')

    # Occupation S_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='abs_occ_s', hue='q_sector', 
                    palette='Set2', ax=axes[0, 1], edgecolor='k', alpha=0.7)
    axes[0, 1].set_title('Occupation $S_k$ vs Real Part')
    axes[0, 1].set_xlabel('Real Part (Decay Rate)')

    # Occupation S_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='abs_occ_s', hue='q_sector', 
                    palette='Set2', ax=axes[1, 1], edgecolor='k', alpha=0.7)
    axes[1, 1].set_title('Occupation $S_k$ vs Abs Imaginary Part')
    axes[1, 1].set_xlabel('Absolute Imaginary Part (Freq)')
    
    # Occupation R_k vs Real
    sns.scatterplot(data=subset, x='real_eval', y='Occ_R', hue='q_sector', 
                    palette='Set2', ax=axes[0, 2], edgecolor='k', alpha=0.7)
    axes[0, 2].set_title('Occupation $R_k$ vs Real Part')
    axes[0, 2].set_xlabel('Real Part (Decay Rate)')
    
    # Occupation R_k vs Imaginary
    sns.scatterplot(data=subset, x='abs_imag', y='Occ_R', hue='q_sector', 
                    palette='Set2', ax=axes[1, 2], edgecolor='k', alpha=0.7)
    axes[1, 2].set_title('Occupation $R_k$ vs Abs Imaginary Part')
    axes[1, 2].set_xlabel('Absolute Imaginary Part (Freq)')

    # Apply grid styling to all subplots
    for ax in axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Adjust layout to prevent overlap and save the figure
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'detailed_plots_gp{gp}_gm{gm}_w{omega}.png', dpi=300, bbox_inches='tight')
    # plt.close()

print("All plots generated successfully.")