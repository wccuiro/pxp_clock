import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the dataset
# Assumes structure: gamma, omega, ev1, occ1, ev2, occ2, ...
df = pd.read_csv('../rust/std_eigenvalues_8.csv', header=None)

# 2. Get unique gamma values and sort them to ensure a smooth video
unique_gammas = np.sort(df.iloc[:, 0].unique())

# Define column indices for Eigenvalues and Occupations
ev_cols = list(range(2, df.shape[1], 2))
occ_cols = list(range(3, df.shape[1], 2))

print(f"Found {len(unique_gammas)} unique gamma values. Starting frame generation...")

# 3. Loop through each gamma and create a plot
for i, gamma in enumerate(unique_gammas):
    # Filter data for this specific gamma
    subset = df[df[0] == gamma]
    omega = subset.iloc[:, 1].values
    
    # Extract and sort eigenvalues/occupations for this frame
    ev_data = subset.iloc[:, ev_cols].values
    occ_data = subset.iloc[:, occ_cols].values
    sort_idx = np.argsort(ev_data, axis=1)
    
    sorted_evs = np.take_along_axis(ev_data, sort_idx, axis=1)
    sorted_occs = np.take_along_axis(occ_data, sort_idx, axis=1)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('cool')
    
    # Plot each level
    for j in range(sorted_evs.shape[1]):
        sc = plt.scatter(omega, sorted_occs[:, j], c=-sorted_evs[:, j]*np.log(sorted_evs[:, j]), 
                         cmap=cmap, s=3, vmin=0, vmax=0.4, edgecolors='none')
    
    # Add persistent colorbar and labels
    cbar = plt.colorbar(sc)
    cbar.set_label('S', fontsize=12)
    plt.xlabel(r'$\Omega$', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    # plt.yscale('log')
    plt.xlim(0, 5)
    plt.ylim(-0.1, 0.6)
    plt.title(r'Eigenvalue Spectrum vs $\Omega$ ($\gamma = {gamma:.3f}$)'.format(gamma=gamma), fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Set axis limits to be constant across all frames (very important for video!)
    # You might want to adjust these based on your data range
    plt.ylim(df.iloc[:, ev_cols].min().min(), df.iloc[:, ev_cols].max().max())
    
    # Save the frame with a padded index for ffmpeg compatibility
    filename = f"../data/video_ss_vals/gamma_per_frame/frame_{i:03d}.png"
    # plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close() # Close figure to free memory
    
    print(f"Saved {filename} for gamma = {gamma}")

print("Done! You can now use ffmpeg to create the video.")