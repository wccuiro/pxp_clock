import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def calculate_waiting_times_custom(jump_times, jump_types, a_minus, a_plus, M):
    """
    Calculates waiting times based on the accumulated count N(t).
    N(t) = a_minus * N_minus(t) + a_plus * N_plus(t)
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

def process_trajectories(folder_path, t_burn=30.0, M_standard=5, M_rare=1):
    file_pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No CSV files found in directory: {folder_path}")
        return [], [], []

    wait_times_emissions = []
    wait_times_activity = []
    wait_times_heat = []

    print(f"Processing {len(files)} files. Discarding all data before t = {t_burn}...")

    for file in files:
        df = pd.read_csv(file)
        
        # Isolate the steady state based on the ensemble plot observation
        df_steady = df[df['time'] >= t_burn]
        df_jumps = df_steady[df_steady['jump_type'].isin([0, 1])]

        if df_jumps.empty:
            continue

        jump_times = df_jumps['time']
        jump_types = df_jumps['jump_type']

        # Case (i): Accumulated emissions (a_minus = 1, a_plus = 0)
        # Using M_rare because gamma_minus is extremely small
        w_emissions = calculate_waiting_times_custom(jump_times, jump_types, a_minus=1, a_plus=0, M=M_rare)
        wait_times_emissions.extend(w_emissions)

        # Case (ii): Dynamical activity (a_minus = 1, a_plus = 1)
        w_activity = calculate_waiting_times_custom(jump_times, jump_types, a_minus=1, a_plus=1, M=M_standard)
        wait_times_activity.extend(w_activity)

        # Case (iii): Dissipated heat current (a_minus = 1, a_plus = -1)
        w_heat = calculate_waiting_times_custom(jump_times, jump_types, a_minus=1, a_plus=-1, M=M_standard)
        wait_times_heat.extend(w_heat)
        
    print("-" * 50)
    print(f"Total Wait Times Extracted (Steady State > {t_burn}):")
    print(f"Case (i)   Accumulated Emissions (M={M_rare}):  {len(wait_times_emissions)}")
    print(f"Case (ii)  Dynamical Activity    (M={M_standard}):  {len(wait_times_activity)}")
    print(f"Case (iii) Heat Current          (M={M_standard}):  {len(wait_times_heat)}")

    return wait_times_emissions, wait_times_activity, wait_times_heat

if __name__ == '__main__':
    DATA_FOLDER = "../data/trajectoriesTN_L20_gamma+0.2_gamma-0.001_dt0.05" 
    
    # Hardcoded transient cutoff based on the ensemble plot
    T_TRANSIENT = 30.0 
    
    w_emissions, w_activity, w_heat = process_trajectories(
        DATA_FOLDER, 
        t_burn=T_TRANSIENT, 
        M_standard=10, 
        M_rare=10 # Lowered threshold for the rare emission channel
    )
    
    if w_activity:
        plt.figure(figsize=(10, 6))
        if w_emissions:
            plt.hist(w_emissions, bins=30, alpha=0.5, label='(i) Accumulated Emissions (M=10)', density=True)
        plt.hist(w_activity, bins=30, alpha=0.5, label='(ii) Dynamical Activity (M=10)', density=True)
        if w_heat:
            plt.hist(w_heat, bins=30, alpha=0.5, label='(iii) Dissipated Heat Current (M=10)', density=True)
            
        plt.title('Waiting Time Distributions (Steady State)')
        plt.xlabel('Waiting Time (Δt between ticks)')
        plt.ylabel('Density')
        plt.legend()
        plt.show()