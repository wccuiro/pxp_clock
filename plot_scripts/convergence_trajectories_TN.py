import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_occupation_dynamics(filename, ax):
    """
    Reads the updated occupation_dynamics.txt format using explicit 
    Time and AvgOccupation columns.
    """
    try:
        # Use a regex separator \s+ to handle any combination of spaces/tabs
        df = pd.read_csv(filename, sep=r'\s+')
        
        # Verify the necessary columns exist to prevent silent failures
        if 'Time' not in df.columns or 'AvgOccupation' not in df.columns:
            print(f"Error: Expected columns 'Time' and 'AvgOccupation' missing in {filename}.")
            return 0
            
        ax.plot(df['Time'], df['AvgOccupation'], label='Lindblad Trajectory', 
                alpha=1, linewidth=3.0, linestyle='--', color='orange')
        
        return 1 # Represents one trajectory plotted

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return 0
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0

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
    
    individual_count = plot_occupation_dynamics('../julia/occupation_dynamics.txt', ax)
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