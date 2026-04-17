import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import os

def load_data(filename):
    """
    Parses the variable-length CSV where each row is:
    gp, gm, omega, re1, im1, oee1, re2, im2, oee2, ...
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    dataset = {}
    gps, gms, omegas = set(), set(), set()

    with open(filename, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(',') if x]
            if len(vals) < 3:
                continue
                
            gp, gm, omega = vals[0], vals[1], vals[2]
            
            # Extract the triplets (Re[lambda], Im[lambda], OEE)
            # Reshape into an N x 3 array
            triplets = np.array(vals[3:]).reshape(-1, 3)
            
            # Store in a dictionary keyed by the parameter tuple
            dataset[(gp, gm, omega)] = {
                're': triplets[:, 0],
                'im': triplets[:, 1],
                'oee': triplets[:, 2]
            }
            
            gps.add(gp)
            gms.add(gm)
            omegas.add(omega)

    return dataset, sorted(list(gps)), sorted(list(gms)), sorted(list(omegas))

# --- Load Data ---
data_file = '../rust/oee.csv'
dataset, gps, gms, omegas = load_data(data_file)

if not dataset:
    print("No valid data found in the file.")
    sys.exit(1)

# --- Initial Parameters ---
init_gp = gps[0]
init_gm = gms[0]
init_omega = omegas[0]

# --- Setup Plot ---
fig, (ax_im, ax_re) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.3)

# Initial scatter plots
scatter_im = ax_im.scatter([], [], c='royalblue', alpha=0.7, edgecolors='none')
ax_im.set_xlabel('Energy Oscillation Im[λ]')
ax_im.set_ylabel('Operator Entanglement Entropy (OEE)')
ax_im.grid(True, linestyle='--', alpha=0.6)

scatter_re = ax_re.scatter([], [], c='royalblue', alpha=0.7, edgecolors='none')
ax_re.set_xlabel('Decay Rate -Re[λ]')
ax_re.set_ylabel('Operator Entanglement Entropy (OEE)')
ax_re.grid(True, linestyle='--', alpha=0.6)

# Title for the figure
title_text = fig.suptitle(f"γ+ = {init_gp:.4f}  |  γ- = {init_gm:.4f}  |  ω = {init_omega:.4f}", fontsize=12)

def set_axes_limits(ax, x_data, y_data):
    """Helper to explicitly set plot limits with a 5% margin."""
    if len(x_data) == 0 or len(y_data) == 0:
        return
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
    y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

def update_plot(gp, gm, omega):
    # Find the closest matching parameters in the dataset to avoid floating point mismatch
    closest_gp = min(gps, key=lambda x: abs(x - gp))
    closest_gm = min(gms, key=lambda x: abs(x - gm))
    closest_omega = min(omegas, key=lambda x: abs(x - omega))
    
    key = (closest_gp, closest_gm, closest_omega)
    
    if key in dataset:
        d = dataset[key]
        
        # Update Im[lambda] vs OEE
        scatter_im.set_offsets(np.c_[d['im'], d['oee']])
        set_axes_limits(ax_im, d['im'], d['oee'])
        
        # Update -Re[lambda] vs OEE (Decay rate is positive)
        re_vals = -d['re']
        scatter_re.set_offsets(np.c_[re_vals, d['oee']])
        set_axes_limits(ax_re, re_vals, d['oee'])
        
        title_text.set_text(f"γ+ = {closest_gp:.4f}  |  γ- = {closest_gm:.4f}  |  ω = {closest_omega:.4f}")
    else:
        title_text.set_text(f"Data not found for γ+={closest_gp:.3f}, γ-={closest_gm:.3f}, ω={closest_omega:.3f}")
        scatter_im.set_offsets(np.empty((0, 2)))
        scatter_re.set_offsets(np.empty((0, 2)))
    
    fig.canvas.draw_idle()

# --- Setup Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_gp = plt.axes([0.15, 0.12, 0.75, 0.03], facecolor=axcolor)
ax_gm = plt.axes([0.15, 0.08, 0.75, 0.03], facecolor=axcolor)
ax_omega = plt.axes([0.15, 0.04, 0.75, 0.03], facecolor=axcolor)

# Helper to avoid identical min/max bounds when a parameter only has one unique value
def get_bounds(vals):
    return (vals[0] - 0.1, vals[0] + 0.1) if len(vals) == 1 else (min(vals), max(vals))

gp_min, gp_max = get_bounds(gps)
gm_min, gm_max = get_bounds(gms)
omega_min, omega_max = get_bounds(omegas)

# Use valstep to snap the sliders exclusively to the discrete values found in the CSV
slider_gp = Slider(ax_gp, 'γ+', gp_min, gp_max, valinit=init_gp, valstep=gps)
slider_gm = Slider(ax_gm, 'γ-', gm_min, gm_max, valinit=init_gm, valstep=gms)
slider_omega = Slider(ax_omega, 'ω', omega_min, omega_max, valinit=init_omega, valstep=omegas)

def on_change(val):
    update_plot(slider_gp.val, slider_gm.val, slider_omega.val)

slider_gp.on_changed(on_change)
slider_gm.on_changed(on_change)
slider_omega.on_changed(on_change)

# Initialize the first view
update_plot(init_gp, init_gm, init_omega)

plt.show()