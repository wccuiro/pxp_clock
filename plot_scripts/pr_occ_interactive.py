import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------------------------
# 1. Load and Pre-process Data
# ---------------------------------------------------------
print("Loading data...")
try:
    # Read the file (assuming no header)
    df = pd.read_csv('../rust/std_eigenvalues_server.csv', header=None)
except FileNotFoundError:
    print("Error: 'std_eigenvalues_server.csv' not found. Please ensure the file is in the working directory.")
    exit()

# Extract Columns
# Col 0: Gamma, Col 1: Omega
gamma_col = df[0].values
omega_col = df[1].values

# Calculate PR for every row
# PR = 1 / sum(lambda^2)
# lambda values are in columns 2, 4, 6, ...
eigenvalues = df.iloc[:, 2::2].values # Numpy array of all eigenvalues
# Square and sum
sum_sq_eigenvalues = np.sum(eigenvalues**2, axis=1)
# Avoid division by zero
pr_values = np.divide(1.0, sum_sq_eigenvalues, where=sum_sq_eigenvalues > 0)

# Create a simplified DataFrame for plotting
plot_df = pd.DataFrame({
    'gamma': gamma_col,
    'omega': omega_col,
    'PR': pr_values
})

# Get unique sorted values for slider and axes
available_omegas = np.sort(plot_df['omega'].unique())
available_gammas = np.sort(plot_df['gamma'].unique())

# ---------------------------------------------------------
# 2. Setup Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25) # Leave space at bottom for slider

# Function to get data for a specific Omega
def get_y_data(target_omega):
    # Find closest actual omega in data
    idx = np.argmin(np.abs(available_omegas - target_omega))
    actual_omega = available_omegas[idx]
    
    # Filter and sort by Gamma
    subset = plot_df[np.isclose(plot_df['omega'], actual_omega)].sort_values('gamma')
    return subset['gamma'].values, subset['PR'].values, actual_omega

# Initial Plot (Start at middle Omega)
init_omega = available_omegas[len(available_omegas)//2]
x_data, y_data, current_omega = get_y_data(init_omega)

line, = ax.plot(x_data, y_data, 'b.-', linewidth=1.5, label='Participation Ratio')

# Reference line at Gamma = 1 (The pinch point)
ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='$\gamma=1$')

# Axis limits and labels
ax.set_xlim(min(available_gammas), max(available_gammas))
# Set Y limit with some headroom (PR >= 1)
ax.set_ylim(0.9, max(plot_df['PR']) * 1.05) 

ax.set_xlabel(r'$\gamma$ (Parameter)')
ax.set_ylabel('Participation Ratio ($1 / \sum p_i^2$)')
ax.set_title(f'PR vs. $\gamma$ at $\Omega \\approx {current_omega:.3f}$')
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------------------------------------------------
# 3. Create Slider
# ---------------------------------------------------------
ax_omega = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_omega = Slider(
    ax=ax_omega,
    label='Omega ($\Omega$)',
    valmin=min(available_omegas),
    valmax=max(available_omegas),
    valinit=init_omega,
)

# ---------------------------------------------------------
# 4. Update Function
# ---------------------------------------------------------
def update(val):
    target_o = slider_omega.val
    new_x, new_y, actual_o = get_y_data(target_o)
    
    # Update line data
    line.set_data(new_x, new_y)
    
    # Update title
    ax.set_title(f'PR vs. $\gamma$ at $\Omega \\approx {actual_o:.3f}$')
    
    # Optional: Rescale Y axis if PR changes drastically?
    # ax.set_ylim(0.9, max(new_y) * 1.1) 
    
    fig.canvas.draw_idle()

slider_omega.on_changed(update)

print("Interactive PR vs Gamma plot started.")
plt.show()