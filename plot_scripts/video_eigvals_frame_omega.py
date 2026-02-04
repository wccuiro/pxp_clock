import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your data
# Structure assumed: [Gamma, Omega, E1, Occ1, E2, Occ2, ...]
df = pd.read_csv('../rust/std_eigenvalues_8.csv', header=None)

# 2. Setup Frames (Iterate over unique Omega values)
unique_omegas = np.sort(df.iloc[:, 1].unique())

# Define columns
ev_cols = list(range(2, df.shape[1], 2))
occ_cols = list(range(3, df.shape[1], 2))


print(f"Generating {len(unique_omegas)} frames...")

for i, omega in enumerate(unique_omegas):
    # Filter for the current frame (Omega)
    subset = df[df[1] == omega].sort_values(by=0)
    
    # Extract data for plotting
    gamma_axis = subset.iloc[:, 0].values  # X-axis
    ev_data = subset.iloc[:, ev_cols].values
    occ_data = subset.iloc[:, occ_cols].values
    
    # Sort eigenvalues (and corresponding occupations) for clean bands
    sort_idx = np.argsort(ev_data, axis=1)
    sorted_evs = np.take_along_axis(ev_data, sort_idx, axis=1)
    sorted_occs = np.take_along_axis(occ_data, sort_idx, axis=1)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('cool')
    
    # Scatter plot for each eigen-level
    for j in range(sorted_evs.shape[1]):
        sc = plt.scatter(gamma_axis, sorted_occs[:, j], c=-sorted_evs[:, j]*np.log(sorted_evs[:, j]), 
                         cmap=cmap, s=15, vmin=0, vmax=0.4, edgecolors='none')
    
    # --- LOG SCALE & LABELS ---
    # plt.yscale('log')  # <--- Log scale set here
    plt.xlabel(r'$\gamma$', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    plt.title(r'Spectrum vs $\gamma$ at $\Omega = {omega:.3f}$'.format(omega=omega), fontsize=16)
    
    # --- FIXED AXIS LIMITS ---
    plt.ylim(-0.1, 0.6)  # Example fixed y-limits
    plt.xlim(0, 2)
    
    # --- COLORBAR ---
    cbar = plt.colorbar(sc)
    cbar.set_label('S', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    
    # --- SAVING THE FRAME ---
    filename = f"../data/video_ss_vals/omega_per_frame/frame_{i:03d}.png"
    # filename = f'frame_{i:03d}.png'
    # plt.tight_layout()
    plt.savefig(filename, dpi=150) # <--- File saved here
    plt.close() # Close memory
    
    if i % 10 == 0:
        print(f"Saved {filename}")

print("All frames generated.")