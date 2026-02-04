import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load Data
# ---------------------------------------------------------
print("Loading data...")
try:
    df = pd.read_csv('../rust/std_eigenvalues_10.csv', header=None)
except FileNotFoundError:
    print("Error: 'std_eigenvalues_server.csv' not found.")
    exit()

# Extract unique parameter values for lookup
available_gammas = np.sort(df[0].unique())
available_omegas = np.sort(df[1].unique())

# Pre-calculate global limits for consistent axes
# (Sampling a few rows to estimate, or scanning all if fast enough)
# For safety, let's set reasonable fixed limits or let it autoscaling initially
# Based on your previous plots: X [0, 1.0], Y [0, 0.4] roughly
X_LIMITS = (-0.05, 0.55) 
Y_LIMITS = (0, 0.4)

# 2. Helper Function to Get Data
# ---------------------------------------------------------
def get_plot_data(gamma_val, omega_val):
    # Find closest available values in the dataset
    closest_g = available_gammas[np.argmin(np.abs(available_gammas - gamma_val))]
    closest_o = available_omegas[np.argmin(np.abs(available_omegas - omega_val))]
    
    # Filter dataframe
    # Using float comparison tolerance
    row = df[np.isclose(df[0], closest_g) & np.isclose(df[1], closest_o)]
    
    if row.empty:
        return None, closest_g, closest_o
        
    row = row.iloc[0]
    
    # Extract Eigenvalues (cols 2, 4, ...) and Occupations (cols 3, 5, ...)
    eigenvalues = row[2::2].values.astype(float)
    occupations = row[3::2].values.astype(float)
    
    # Calculate Entropy (-lambda * ln(lambda))
    entropy = np.zeros_like(eigenvalues)
    mask = eigenvalues > 0
    entropy[mask] = -eigenvalues[mask] * np.log(eigenvalues[mask])
    
    # Group for multiplicity (bubble sizes)
    data = pd.DataFrame({'Occupation': occupations, 'Entropy': entropy})
    grouped = data.groupby(['Occupation', 'Entropy']).size().reset_index(name='Count')
    
    return grouped, closest_g, closest_o

# 3. Setup Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25) # Make room for sliders

# Initial Plot (start with middle values)
init_gamma = 1.0
init_omega = 1.0

data, actual_g, actual_o = get_plot_data(init_gamma, init_omega)

scatter = ax.scatter(data['Occupation'], data['Entropy'], 
                     s=data['Count']*50, # Scale bubble size
                     alpha=0.6, edgecolors='b', label='States')

ax.set_xlabel("Occupation")
ax.set_ylabel(r"Entropy ($-\lambda \ln \lambda$)")
ax.set_title(r'Steady State Distribution $\gamma \approx {actual_g:.3f}, \Omega \approx {actual_o:.3f}$'.format(actual_g=actual_g, actual_o=actual_o))
ax.set_xlim(X_LIMITS)
ax.set_ylim(Y_LIMITS)
ax.grid(True, alpha=0.3)

# 4. Create Sliders
# ---------------------------------------------------------
# Define axes for sliders [left, bottom, width, height]
ax_gamma = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_omega = plt.axes([0.2, 0.05, 0.65, 0.03])

slider_gamma = Slider(
    ax_gamma, r'Gamma ($\gamma$)', 
    valmin=min(available_gammas), 
    valmax=max(available_gammas), 
    valinit=init_gamma
)

slider_omega = Slider(
    ax_omega, r'Omega ($\Omega$)', 
    valmin=min(available_omegas), 
    valmax=max(available_omegas), 
    valinit=init_omega
)

# 5. Update Function
# ---------------------------------------------------------
def update(val):
    target_g = slider_gamma.val
    target_o = slider_omega.val
    
    new_data, new_g, new_o = get_plot_data(target_g, target_o)
    
    if new_data is not None:
        # Update positions
        # scatter.set_offsets expects an (N, 2) array
        points = np.column_stack((new_data['Occupation'], new_data['Entropy']))
        scatter.set_offsets(points)
        
        # Update sizes
        scatter.set_sizes(new_data['Count'] * 50)
        
        # Update title
        ax.set_title(r'Steady State Distribution $\gamma \approx {new_g:.3f}, \Omega \approx {new_o:.3f}$'.format(new_g=new_g, new_o=new_o))
        
        # Redraw
        fig.canvas.draw_idle()

# Attach update function to sliders
slider_gamma.on_changed(update)
slider_omega.on_changed(update)

print("Interactive plot started.")
plt.show()