import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This script provides functions to plot the time evolution of the occupation number n, the correlation and the fidelity and compares them with the ensamble average of multiple trajectories. 
The data of the exact time evolution is expected to be in a CSV file with the following format:
gamma_plus, gamma_minus, omega, n_0, nn_0, fidelity_0, n_1, nn_1, fidelity_1, ..., n_N, nn_N, fidelity_N

the data of each trajectory is expected to be in a CSV file with the following format:
time, n, nn, fidelity
"""


def plot_occupation_dynamics(filename, ax, dt=1e-2):
    lines_plotted = 0
    
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',') if p.strip()]
            
            if len(parts) < 3:
                continue
                
            g = float(parts[0])
            omega = float(parts[1])
            
            n_values = [float(parts[i]) for i in range(3, len(parts), 3)]
            
            # Scale the index by dt to align with the actual time axis
            time_axis = [i * dt for i in range(len(n_values))]
            
            ax.plot(time_axis, n_values, label=f'g={g}, omega={omega}', 
                    alpha=1, linewidth=3.0, linestyle='--')
            lines_plotted += 1

    if lines_plotted == 0:
        print(f"Warning: No valid data found in {filename}.")
        
    return lines_plotted

def plot_average_trajectory_uniform(folder_path, ax):
    search_pattern = os.path.join(folder_path, 'traj_*.csv')
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"Error: No files matching 'traj_*.csv' found in {folder_path}")
        return 0

    all_n = []
    time_array = None

    for file in file_list:
        df = pd.read_csv(file)
        
        if not {'time', 'n'}.issubset(df.columns):
            continue
            
        if time_array is None:
            time_array = df['time'].values

        if len(df['n']) != len(time_array):
            continue

        all_n.append(df['n'].values)

    if not all_n:
        return 0

    n_matrix = np.array(all_n)
    n_average = np.mean(n_matrix, axis=0)
    n_std = np.std(n_matrix, axis=0)
    
    ax.plot(time_array, n_average, color='blue', linewidth=2.5, label='Ensemble Average $\\langle n \\rangle$')
    ax.fill_between(time_array, n_average - n_std, n_average + n_std, color='blue', alpha=0.2, label=r'$\pm 1\sigma$ Variance')
    
    return len(all_n)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # The integration_dt variable is removed since time is explicit in the new file format.
    
    individual_count = plot_occupation_dynamics('../rust/occupation_time.csv', ax, dt=1e-2)
    ensemble_count = plot_average_trajectory_uniform('../data/trajectoriesTN', ax)
    
    ax.set_title(f'Time Evolution of n (Averaging {ensemble_count} runs)')
    ax.set_xlabel('Time')
    ax.set_ylabel('n')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    
    # Kept your legend logic, though with only 3 likely handles now (Julia, Ensemble, Variance) 
    # the first block will rarely trigger unless the data folder behavior changes.
    if len(handles) > 15:
        ensemble_handles = handles[-2:]
        ensemble_labels = labels[-2:]
        ax.legend(ensemble_handles, ensemble_labels, loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.show()