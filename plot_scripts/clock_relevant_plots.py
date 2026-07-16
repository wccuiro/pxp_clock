import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def extract_wait_times(jump_times, M):
    """Calculates waiting times for the dynamical activity clock (total jumps)."""
    if len(jump_times) < M:
        return []
    tick_times = jump_times.iloc[M-1::M].values
    return np.diff(np.insert(tick_times, 0, jump_times.iloc[0])).tolist()

def analyze_clock_performance(folder_path, t_burn=30.0, max_M=100):
    """Calculates resolution and accuracy across a range of thresholds."""
    file_pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found in {folder_path}")
        return None, None
        
    M_values = np.arange(1, max_M + 1)
    wait_times_dict = {m: [] for m in M_values}
    total_time_steady = 0.0
    total_jumps_steady = 0
    
    for file in files:
        df = pd.read_csv(file)
        df_steady = df[df['time'] >= t_burn]
        df_jumps = df_steady[df_steady['jump_type'].isin([0, 1])]
        
        if df_jumps.empty:
            continue
            
        total_time_steady += (df_steady['time'].max() - t_burn)
        total_jumps_steady += len(df_jumps)
        
        j_times = df_jumps['time']
        for m in M_values:
            wait_times_dict[m].extend(extract_wait_times(j_times, m))
            
    # Calculate global jump rate for classical Poisson bound
    gamma_tot = total_jumps_steady / total_time_steady if total_time_steady > 0 else 0
    
    results = {'M': [], 'R': [], 'A': [], 'mean_T': [], 'var_T': [], 'wait_times': []}
    
    for m in M_values:
        wt = np.array(wait_times_dict[m])
        if len(wt) > 1 and np.mean(wt) > 0:
            mean_T = np.mean(wt)
            var_T = np.var(wt, ddof=1)
            
            results['M'].append(m)
            results['R'].append(1.0 / mean_T)
            results['A'].append((mean_T**2) / var_T)
            results['mean_T'].append(mean_T)
            results['var_T'].append(var_T)
            results['wait_times'].append(wt)
            
    return results, gamma_tot

def generate_plots(base_data_folder, t_burn=30.0):
    # Analyze the main dataset
    res, gamma_tot = analyze_clock_performance(base_data_folder, t_burn=t_burn, max_M=100)
    
    if not res:
        return
        
    M_arr = np.array(res['M'])
    R_arr = np.array(res['R'])
    A_arr = np.array(res['A'])
    
    # Find Optimal Threshold (Max Accuracy)
    opt_idx = np.argmax(A_arr)
    M_opt = M_arr[opt_idx]
    A_opt = A_arr[opt_idx]
    R_opt = R_arr[opt_idx]
    wt_opt = res['wait_times'][opt_idx]
    
    print(f"Optimal Threshold Found: M = {M_opt}")
    print(f"Max Accuracy: {A_opt:.2f}, Resolution: {R_opt:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---------------------------------------------------------
    # Plot 1: Accuracy-Resolution Trade-off
    # ---------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(R_arr, A_arr, 'o-', color='darkorange', label='Dynamical Activity Clock')
    ax.plot(R_opt, A_opt, 's', color='red', markersize=10, label=f'Optimal (M={M_opt})')
    
    # Poisson Bound: A = Gamma / R
    R_smooth = np.linspace(min(R_arr), max(R_arr), 100)
    ax.plot(R_smooth, gamma_tot / R_smooth, 'k--', label=r'Poisson Bound $\mathcal{A} = \Gamma_{tot} / \mathcal{R}$')
    
    ax.set_title('(1) Accuracy-Resolution Trade-off')
    ax.set_xlabel(r'Resolution $\mathcal{R}$')
    ax.set_ylabel(r'Accuracy $\mathcal{A}$')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    # ---------------------------------------------------------
    # Plot 2: Waiting Time Distribution at Optimal M
    # ---------------------------------------------------------
    ax = axes[0, 1]
    ax.hist(wt_opt, bins=30, color='darkorange', alpha=0.7, density=True, edgecolor='black')
    ax.axvline(np.mean(wt_opt), color='red', linestyle='dashed', linewidth=2, label=r'Mean $\langle \mathcal{T} \rangle$')
    
    ax.set_title(f'(2) Waiting Time Distribution at M={M_opt}')
    ax.set_xlabel(r'Waiting Time $\mathcal{T}$')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    # ---------------------------------------------------------
    # Plot 3: Kinetic Uncertainty Relation (KUR) Saturation
    # ---------------------------------------------------------
    ax = axes[1, 0]
    ax.loglog(M_arr, A_arr, 'o-', color='darkorange', label='Measured Accuracy')
    
    # For a dynamical activity clock, <K_tick> is exactly M.
    ax.loglog(M_arr, M_arr, 'k--', label=r'KUR Bound $\mathcal{A} \leq \langle \mathcal{K}_{tick} \rangle$')
    
    ax.set_title('(3) Kinetic Uncertainty Relation')
    ax.set_xlabel(r'Dynamical Activity per Tick $\langle \mathcal{K}_{tick} \rangle = M$')
    ax.set_ylabel(r'Accuracy $\mathcal{A}$')
    ax.legend()
    ax.grid(True, which="both", ls='--', alpha=0.5)

    # ---------------------------------------------------------
    # Plot 4: Accuracy vs. Drive (Requires Multiple Datasets)
    # ---------------------------------------------------------
    ax = axes[1, 1]
    
    # This logic assumes you run trajectories for different Omegas and save them
    # in directories named like "../data/trajectoriesTN_Omega1.0"
    parent_dir = os.path.dirname(base_data_folder)
    all_folders = glob.glob(os.path.join(parent_dir, "trajectoriesTN_Omega*"))
    
    omegas = []
    max_accuracies = []
    
    if len(all_folders) > 1:
        for folder in all_folders:
            try:
                # Extract Omega from folder name string
                omega_str = folder.split("Omega")[1].split("_")[0]
                omega_val = float(omega_str)
                folder_res, _ = analyze_clock_performance(folder, t_burn, max_M=50)
                if folder_res:
                    omegas.append(omega_val)
                    max_accuracies.append(np.max(folder_res['A']))
            except Exception:
                continue
                
        # Sort for plotting
        sort_idx = np.argsort(omegas)
        omegas = np.array(omegas)[sort_idx]
        max_accuracies = np.array(max_accuracies)[sort_idx]
        
        ax.plot(omegas, max_accuracies, 's-', color='purple', markersize=8)
        ax.set_title('(4) Clock Accuracy vs Coherent Drive')
        ax.set_xlabel(r'Rabi Frequency $\Omega$')
    else:
        # Fallback if you only have one dataset right now
        ax.plot([1.0], [A_opt], 's', color='purple', markersize=10)
        ax.set_title('(4) Clock Accuracy vs Coherent Drive (Single Data Point)')
        ax.set_xlabel('Drive (Need multiple Omega folders to plot curve)')
        
    ax.set_ylabel(r'Maximum Accuracy $\mathcal{A}_{opt}$')
    ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Point this to your existing dataset folder
    DATA_FOLDER = "../data/trajectoriesTN_L20_gamma+0.2_gamma-0.001_dt0.05" 
    T_BURN = 30.0 
    
    generate_plots(DATA_FOLDER, t_burn=T_BURN)