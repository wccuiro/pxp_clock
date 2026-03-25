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
            if len(parts) < 2: continue
            try:
                g = float(parts[0])
                omega = float(parts[1])
                
                # Iterate through eigenvalues (groups of 4: real, imag, overlap, k)
                for i in range(2, len(parts), 4):
                    if i+3 >= len(parts): break
                    real_val = float(parts[i])
                    imag_val = float(parts[i+1])
                    overlap_val = float(parts[i+2])
                    
                    # Store as: g, omega, decay_rate (-real), imag, overlap
                    data_list.append((g, omega, -real_val, imag_val, overlap_val))
            except ValueError:
                continue

    # Convert to structured array for easy filtering
    dtype = [('g', 'f8'), ('omega', 'f8'), ('decay', 'f8'), ('imag', 'f8'), ('overlap', 'f8')]
    data = np.array(data_list, dtype=dtype)
    print(f"Loaded {len(data)} eigenvalues.")
    return data

# --- Main Script ---

# 1. Load Data
filename = '../rust/decay_8_80x80.csv' # Make sure this file is in the same directory
data = load_data(filename)

# Get unique values for snapping
unique_gs = np.sort(np.unique(data['g']))
unique_omegas = np.sort(np.unique(data['omega']))

# 2. Setup the Plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25) # Make room for sliders

# Initial values (start from the middle)
init_g = unique_gs[len(unique_gs)//2]
init_w = unique_omegas[len(unique_omegas)//2]

# Helper to get subset
def get_subset(g_val, w_val):
    # Find nearest available g and omega in the dataset
    nearest_g = unique_gs[np.argmin(np.abs(unique_gs - g_val))]
    nearest_w = unique_omegas[np.argmin(np.abs(unique_omegas - w_val))]
    
    mask = (np.abs(data['g'] - nearest_g) < 1e-5) & \
           (np.abs(data['omega'] - nearest_w) < 1e-5)
    return data[mask], nearest_g, nearest_w

# Initial plot
subset, current_g, current_w = get_subset(init_g, init_w)

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
title_text = ax.set_title(f'Spectrum at $g={current_g:.4f}, \omega={current_w:.4f}$')
ax.grid(True, linestyle='--', alpha=0.5)

# Fix axes limits to the global data range so the window doesn't jump around
ax.set_xlim(np.min(data['decay']), np.max(data['decay']))
ax.set_ylim(np.min(data['imag']), np.max(data['imag']))

# 3. Create Sliders
axcolor = 'lightgoldenrodyellow'
ax_g = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_w = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

s_g = Slider(ax_g, 'Dissipation (g)', unique_gs.min(), unique_gs.max(), valinit=init_g)
s_w = Slider(ax_w, 'Frequency ($\omega$)', unique_omegas.min(), unique_omegas.max(), valinit=init_w)

# 4. Update Function
def update(val):
    # Get slider values
    target_g = s_g.val
    target_w = s_w.val
    
    # Get new data subset (snapped to nearest grid point)
    new_subset, near_g, near_w = get_subset(target_g, target_w)
    
    # Update scatter plot data
    # scatter.set_offsets takes a (N, 2) array of xy coordinates
    new_offsets = np.column_stack((new_subset['decay'], new_subset['imag']))
    scatter.set_offsets(new_offsets)
    
    # Update colors and sizes
    scatter.set_array(new_subset['overlap'])
    scatter.set_sizes(new_subset['overlap']*500)
    
    # Update title to show the actual values we snapped to
    title_text.set_text(f'Spectrum at $g={near_g:.4f}, \omega={near_w:.4f}$')
    
    fig.canvas.draw_idle()

# Attach update function to sliders
s_g.on_changed(update)
s_w.on_changed(update)

plt.show()