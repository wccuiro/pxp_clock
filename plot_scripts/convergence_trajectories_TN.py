import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

"""
This script plots the time evolution of the occupation number (n), nearest-neighbor correlation (nnn_corr), 
and fidelity, comparing exact dynamics with the ensemble average of multiple quantum trajectories. 

It generates a separate figure for each variable. Each figure contains a 2x2 grid of subplots 
corresponding to combinations of the gamma_plus and gamma_minus decay parameters.

Data Format Expectations:
1. Exact time evolution (Single CSV file):
   gamma_plus, gamma_minus, omega, n_0, nn_0, fidelity_0, n_1, nn_1, fidelity_1, ...
2. Individual Trajectory data (Multiple CSV files inside parameter-specific directories):
   time, jump_type, n, nnn_corr, fidelity
"""

# Global parameters for easy modification
L = 10
gamma_plus_vals = [0.001, 0.2]
gamma_minus_vals = [0.001, 0.2]
dt_trajectories = 0.1
dt_exact = 0.01  # Time step for exact dynamics, should match the one used in the exact data generation
exact_file = '../rust/occupation_time.csv'

def plot_exact_dynamics(filename, ax, target_gp, target_gm, var_name, dt_exact):
    """
    Reads the exact data CSV. Extracts the specific variable using an offset,
    filtering for the target gamma parameters.
    """
    # Offset mapping to extract the correct variable from the repeating triplets
    offset_map = {'n': 0, 'nnn_corr': 1, 'fidelity': 2}
    if var_name not in offset_map:
        return 0
    
    offset = offset_map[var_name]
    lines_plotted = 0
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) < 3:
                    continue
                    
                g_plus = float(parts[0])
                g_minus = float(parts[1])
                omega = float(parts[2])
                
                # Filter by the current subplot's gamma parameters
                # Using a small tolerance to handle floating point comparisons
                if abs(g_plus - target_gp) < 1e-5 and abs(g_minus - target_gm) < 1e-5:
                    # Start at index 3 + offset, jump by 3
                    values = [float(parts[i]) for i in range(3 + offset, len(parts), 3)]
                    time_axis = [i * dt_exact for i in range(len(values))]
                    
                    ax.plot(time_axis, values, label=fr'Exact ($\omega$={omega})', 
                            alpha=1, linewidth=2.0, linestyle='--')
                    lines_plotted += 1
    except FileNotFoundError:
        pass # Handle silently, will just show ensemble if exact is missing
        
    return lines_plotted

def plot_average_trajectory(folder_path, var_name, ax):
    """
    Reads all trajectory CSVs in the specified folder and plots the ensemble average 
    and standard deviation for the given variable.
    """
    search_pattern = os.path.join(folder_path, 'traj_*.csv')
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        return 0

    all_vals = []
    time_array = None

    for file in file_list:
        df = pd.read_csv(file)
        
        # Check if necessary columns exist
        if not {'time', var_name}.issubset(df.columns):
            continue
            
        if time_array is None:
            time_array = df['time'].values

        # Ensure lengths match before appending
        if len(df[var_name]) != len(time_array):
            continue

        all_vals.append(df[var_name].values)

    if not all_vals:
        return 0

    val_matrix = np.array(all_vals)
    val_average = np.mean(val_matrix, axis=0)
    val_std = np.std(val_matrix, axis=0)
    
    ax.plot(time_array, val_average, color='blue', linewidth=2.0, label='Ensemble Avg')
    ax.fill_between(time_array, val_average - val_std, val_average + val_std, 
                    color='blue', alpha=0.2, label=r'$\pm 1\sigma$ Variance')
    
    return len(all_vals)

if __name__ == "__main__":
    variables = ['n', 'nnn_corr', 'fidelity']
    
    # Creates the 4 combinations: (0.001, 0.001), (0.001, 0.2), (0.2, 0.001), (0.2, 0.2)
    combinations = list(itertools.product(gamma_plus_vals, gamma_minus_vals))
    
    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Time Evolution of {var} (L={L}, dt={dt_trajectories})', fontsize=16)
        axes = axes.flatten()
        
        for idx, (gp, gm) in enumerate(combinations):
            ax = axes[idx]
            
            # Construct the dynamic path for the current parameter combination
            folder_path = f"../data/trajectoriesTN_L{L}_gamma+{gp}_gamma-{gm}_dt{dt_trajectories}"
            
            # Plot exact data and ensemble data
            exact_count = plot_exact_dynamics(exact_file, ax, gp, gm, var, dt_exact=dt_exact)
            ensemble_count = plot_average_trajectory(folder_path, var, ax)
            
            # Format subplot titles and labels
            title_str = fr"$\gamma_+$={gp}, $\gamma_-$={gm}"
            if ensemble_count > 0:
                title_str += f" (Averaging {ensemble_count} runs)"
            else:
                title_str += " (No trajectory data found)"
                
            ax.set_title(title_str)
            ax.set_xlabel('Time')
            ax.set_ylabel(var)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add legend only if lines were plotted
            if exact_count > 0 or ensemble_count > 0:
                ax.legend(loc='best')
                
        plt.tight_layout()
        plt.show()