import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def calculate_waiting_times_custom(jump_times, jump_types, a_minus, a_plus, M):
    """
    Calculates waiting times based on the accumulated count N(t).
    """
    wait_times = []
    if len(jump_times) == 0:
        return wait_times
        
    last_tick_time = jump_times.iloc[0]
    accumulator = 0
    
    for t, j_type in zip(jump_times, jump_types):
        if j_type == 1:
            accumulator += a_minus
        elif j_type == 0:
            accumulator += a_plus
            
        if abs(accumulator) >= M:
            wait_times.append(t - last_tick_time)
            last_tick_time = t
            accumulator = 0  
            
    return wait_times

def process_uncertainty_scaling(folder_path, t_burn=30.0, max_M=50):
    file_pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No CSV files found in directory: {folder_path}")
        return {}

    # Define the M thresholds we want to test
    # We use a logarithmic spread to get even spacing on the log-log plot
    M_values = np.unique(np.logspace(0, np.log10(max_M), num=20, dtype=int))
    
    # Store aggregate wait times for each M
    wait_times = {
        'emissions': {m: [] for m in M_values},
        'activity': {m: [] for m in M_values},
        'heat': {m: [] for m in M_values}
    }

    print(f"Processing {len(files)} files for uncertainty scaling...")
    print(f"Testing M values: {M_values}")

    for file in files:
        df = pd.read_csv(file)
        df_steady = df[df['time'] >= t_burn]
        df_jumps = df_steady[df_steady['jump_type'].isin([0, 1])]

        if df_jumps.empty:
            continue

        j_times = df_jumps['time']
        j_types = df_jumps['jump_type']

        for m in M_values:
            # Case (i)
            wait_times['emissions'][m].extend(
                calculate_waiting_times_custom(j_times, j_types, 1, 0, m)
            )
            # Case (ii)
            wait_times['activity'][m].extend(
                calculate_waiting_times_custom(j_times, j_types, 1, 1, m)
            )
            # Case (iii)
            wait_times['heat'][m].extend(
                calculate_waiting_times_custom(j_times, j_types, 1, -1, m)
            )

    # Calculate epsilon^2 = Var(dt) / Mean(dt)^2
    uncertainties = {'M': M_values, 'emissions': [], 'activity': [], 'heat': []}
    
    for m in M_values:
        for key in ['emissions', 'activity', 'heat']:
            wt_array = np.array(wait_times[key][m])
            
            # We need at least 2 ticks to compute variance
            if len(wt_array) > 1 and np.mean(wt_array) > 0:
                var = np.var(wt_array, ddof=1)
                mean_sq = np.mean(wt_array)**2
                uncertainties[key].append(var / mean_sq)
            else:
                uncertainties[key].append(np.nan)

    return uncertainties

if __name__ == '__main__':
    DATA_FOLDER = "../data/trajectoriesTN_L20_gamma+0.2_gamma-0.001_dt0.05" 
    T_TRANSIENT = 30.0 
    MAX_THRESHOLD = 50 # Calculate up to M=50
    
    results = process_uncertainty_scaling(DATA_FOLDER, t_burn=T_TRANSIENT, max_M=MAX_THRESHOLD)
    
    if results:
        M_vals = results['M']
        eps2_emi = results['emissions']
        eps2_act = results['activity']
        eps2_heat = results['heat']
        
        plt.figure(figsize=(9, 6))
        
        # Plot the calculated uncertainties
        plt.loglog(M_vals, eps2_emi, 'o-', alpha=0.7, label='(i) Accumulated Emissions')
        plt.loglog(M_vals, eps2_act, 's-', alpha=0.7, label='(ii) Dynamical Activity')
        plt.loglog(M_vals, eps2_heat, '^-', alpha=0.7, label='(iii) Dissipated Heat Current')
        
        # Plot theoretical 1/M reference line 
        # Shifted slightly for visual alignment with the activity/heat lines
        ref_M = np.array([M_vals[0], M_vals[-1]])
        ref_line = 1.0 / ref_M
        
        # Adjust constant to align the reference line with the data visually if needed
        # We use the first valid activity point to anchor the reference line
        first_valid_idx = np.where(~np.isnan(eps2_act))[0][0]
        anchor = eps2_act[first_valid_idx] * M_vals[first_valid_idx]
        plt.loglog(ref_M, anchor / ref_M, 'k--', label=r'Reference $\propto 1/M$')
        
        plt.title('Clock Precision: Relative Uncertainty vs Threshold')
        plt.xlabel('Threshold (M)')
        plt.ylabel(r'Relative Uncertainty $\epsilon^2 = \text{Var}(\Delta t) / \langle \Delta t \rangle^2$')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()