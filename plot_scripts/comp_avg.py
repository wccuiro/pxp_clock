import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

# --- 1. Settings ---
data_dir = "../data/trajectories/"
output_image = "pxp_trajectory_corrected.png"
expected_value = -0.6425108401711591 
common_dt = 0.05  # The resolution for the shared time grid
max_time = 400.0  # Should match your Rust simulation total_time

# --- 2. Define Common Time Grid ---
# We create one master clock that all files will conform to
common_time = np.arange(0, max_time, common_dt)
num_steps = len(common_time)

# Arrays to accumulate interpolated data
# Shape: (Number of Files, Number of Time Steps)
all_sz_interpolated = []

# --- 3. Load and Resample Data ---
print(f"Reading and resampling data from: {data_dir}")
file_list = glob.glob(os.path.join(data_dir, "traj_*.csv"))

if not file_list:
    print("Error: No files found.")
    exit()

for f in file_list:
    # Read raw data (which has random/irregular time points)
    df = pd.read_csv(f)
    
    # Sort just in case times are out of order
    df = df.sort_values('time')
    
    t_raw = df['time'].values
    sz_raw = df['sz'].values
    
    # --- CRITICAL STEP: Interpolation ---
    # We map the raw irregular (t, sz) onto our fixed 'common_time'
    # np.interp performs linear interpolation
    sz_interp = np.interp(common_time, t_raw, sz_raw)
    
    all_sz_interpolated.append(sz_interp)

# Convert list to a 2D Matrix (N_traj x N_time)
sz_matrix = np.array(all_sz_interpolated)

# --- 4. Calculate Statistics on the Aligned Grid ---
# Now we can safely average "column by column" because the columns represent the EXACT same time.
mean_sz = np.mean(sz_matrix, axis=0)
std_sz = np.std(sz_matrix, axis=0)

# --- 5. Plotting ---
plt.figure(figsize=(10, 6), dpi=150)

# A. Plot Individual Trajectories (Thin transparent lines)
# We plot the first 50 RESAMPLED trajectories
for i in range(min(100, len(sz_matrix))):
    plt.plot(common_time, sz_matrix[i], color='red', alpha=0.03, linewidth=1)

# B. Plot Standard Deviation (Shaded)
plt.fill_between(common_time, mean_sz - std_sz, mean_sz + std_sz, 
                 color='orange', alpha=0.2, label=r'Standard Deviation ($\sigma$)')

# C. Plot Ensemble Average
plt.plot(common_time, mean_sz, color='#D35400', linewidth=2, label=r'Ensemble Average $\langle S^z \rangle$')

# D. Theory Line
plt.axhline(y=expected_value, color='black', linestyle='--', linewidth=2, label='Theory')

# Formatting
plt.title(f"PXP Model: Aligned Trajectories (N={len(file_list)})")
plt.xlabel("Time ($1/\Omega$)")
plt.ylabel(r"Magnetization $\langle S^z \rangle$")
plt.ylim(-1.1, 1.1)
plt.xlim(0, max_time)
plt.grid(True, linestyle=':', alpha=0.6)

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.savefig(output_image)
print(f"Plot saved to '{output_image}'")
plt.show()