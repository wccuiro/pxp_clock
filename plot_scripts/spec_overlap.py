import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def load_data(filename):
    """
    Parses the decay.csv file robustly, handling variable line lengths.
    Returns structured numpy array.
    """
    data_list = []
    print(f"Loading {filename}...")
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3: continue
            try:
                gp = float(parts[0])
                gm = float(parts[1])
                omega = float(parts[2])
                
                # Iterate through eigenvalues (groups of 5: real, imag, overlap, occupation, k)
                for i in range(3, len(parts), 5):
                    if i+4 >= len(parts): break
                    real_val = float(parts[i])
                    imag_val = float(parts[i+1])
                    overlap_val = float(parts[i+2])
                    occupation_val = float(parts[i+3])
                    k_val = float(parts[i+4])
                    
                    # Store as: g, omega, decay_rate (-real), imag, overlap, occupation
                    data_list.append((gp, gm, omega, -real_val, imag_val, overlap_val, occupation_val))
            except ValueError:
                continue

    # Convert to structured array for easy filtering
    dtype = [('gp', 'f8'), ('gm', 'f8'), ('omega', 'f8'), ('decay', 'f8'), ('imag', 'f8'), ('overlap', 'f8'), ('occupation', 'f8')]
    data = np.array(data_list, dtype=dtype)
    return data

# --- Main Script ---

# 1. Load Data
filename = '../rust/decay.csv' # Make sure this file is in the same directory
data = load_data(filename)

# Get unique values for snapping
unique_gps = np.sort(np.unique(data['gp']))
unique_gms = np.sort(np.unique(data['gm']))
unique_omegas = np.sort(np.unique(data['omega']))

# 2. Setup the Plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25) # Make room for sliders

# Initial values (start from the middle)
init_gp = unique_gps[len(unique_gps)//2]
init_gm = unique_gms[len(unique_gms)//2]
init_w = unique_omegas[len(unique_omegas)//2]

# Helper to get subset
def get_subset(gp_val, gm_val, w_val):
    # Find nearest available gp, gm, and omega in the dataset
    nearest_gp = unique_gps[np.argmin(np.abs(unique_gps - gp_val))]
    nearest_gm = unique_gms[np.argmin(np.abs(unique_gms - gm_val))]
    nearest_w = unique_omegas[np.argmin(np.abs(unique_omegas - w_val))]

    mask = (np.abs(data['gp'] - nearest_gp) < 1e-5) & \
           (np.abs(data['gm'] - nearest_gm) < 1e-5) & \
           (np.abs(data['omega'] - nearest_w) < 1e-5)
    return data[mask], nearest_gp, nearest_gm, nearest_w

# Initial plot
subset, current_gp, current_gm, current_w = get_subset(init_gp, init_gm, init_w)

# We plot two layers: 
# 1. Grey background dots (context) - optional, can be noisy if too many points
# 2. Colored main scatter plot
scatter = ax.scatter(subset['decay'], subset['imag'], 
                     c=subset['overlap'], cmap='Oranges', 
                     s=subset['overlap']*500, alpha=0.8, edgecolors='k',
                     vmin=0, vmax=np.max(data['overlap'])/2) # Fix color scale globally

# Add colorbar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Overlap Magnitude')

# Set labels and fixed limits (optional, allows seeing movement better)
ax.set_xlabel(r'Dissipation Rate ($-\mathrm{Re}[\lambda]$)')
ax.set_ylabel(r'Oscillation Frequency ($\mathrm{Im}[\lambda]$)')
title_text = ax.set_title(f'Spectrum at $g_p={current_gp:.4f}, g_m={current_gm:.4f}, \omega={current_w:.4f}$')
ax.grid(True, linestyle='--', alpha=0.5)

# Fix axes limits to the global data range so the window doesn't jump around
ax.set_xlim(np.min(data['decay']), np.max(data['decay']))
ax.set_ylim(np.min(data['imag']), np.max(data['imag']))

# 3. Create Sliders
axcolor = 'lightgoldenrodyellow'
ax_gp = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_gm = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_w = plt.axes([0.2, 0.0, 0.65, 0.03], facecolor=axcolor)

s_gp = Slider(ax_gp, 'Dissipation (g_p)', unique_gps.min(), unique_gps.max(), valinit=init_gp)
s_gm = Slider(ax_gm, 'Dissipation (g_m)', unique_gms.min(), unique_gms.max(), valinit=init_gm)
s_w = Slider(ax_w, 'Frequency ($\omega$)', unique_omegas.min(), unique_omegas.max(), valinit=init_w)

# 4. Update Function
def update(val):
    # Get slider values
    target_gp = s_gp.val
    target_gm = s_gm.val
    target_w = s_w.val
    
    # Get new data subset (snapped to nearest grid point)
    new_subset, near_gp, near_gm, near_w = get_subset(target_gp, target_gm, target_w)
    
    # Update scatter plot data
    # scatter.set_offsets takes a (N, 2) array of xy coordinates
    new_offsets = np.column_stack((new_subset['decay'], new_subset['imag']))
    scatter.set_offsets(new_offsets)
    
    # Update colors and sizes
    scatter.set_array(new_subset['overlap'])
    scatter.set_sizes(new_subset['overlap']*500)
    
    # Update title to show the actual values we snapped to
    title_text.set_text(f'Spectrum at $g_p={near_gp:.4f}, g_m={near_gm:.4f}, \omega={near_w:.4f}$')
    
    fig.canvas.draw_idle()

# Attach update function to sliders
s_gp.on_changed(update)
s_gm.on_changed(update)
s_w.on_changed(update)

plt.show()