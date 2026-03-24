import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load the dataset without headers
# Pandas will automatically assign integer column names: 0, 1, 2, 3...
df = pd.read_csv('../rust/decay.csv', header=None)

# 2. Reshape the data from "wide" to "long" format
# Column 0 is g, Column 1 is omega
g_vals = df[0].values
omega_vals = df[1].values

# Extract real (even indices from 2 onwards) and imag (odd indices from 3 onwards)
real_cols = df.columns[2::4]
imag_cols = df.columns[3::4]
overlap_cols = df.columns[4::4]
blockSize_cols = df.columns[5::4]

# Ensure we only take complete pairs in case the CSV has a trailing comma
min_len = min(len(real_cols), len(imag_cols))
real_vals = df[real_cols[:min_len]].values
imag_vals = df[imag_cols[:min_len]].values
overlap_vals = df[overlap_cols[:min_len]].values
blockSize_vals = df[blockSize_cols[:min_len]].values

# Repeat g and omega so they align with the flattened arrays
num_val_pairs = real_vals.shape[1]
g_long = np.repeat(g_vals, num_val_pairs)
omega_long = np.repeat(omega_vals, num_val_pairs)

# Build a clean dataframe handling ALL values seamlessly
long_df = pd.DataFrame({
    'g': g_long,
    'omega': omega_long,
    'val_real': real_vals.flatten(),
    'val_imag': imag_vals.flatten(),
    'overlap': overlap_vals.flatten(),
    'blockSize': blockSize_vals.flatten()
}).dropna() # Drops empty pairs if the CSV has rows of varying lengths

# 3. Normalize to project onto the unit circle
iphi = long_df['val_real'] + 1j * long_df['val_imag']


long_df['x_norm'] = np.real(np.exp(iphi))
long_df['y_norm'] = np.imag(np.exp(iphi))


# Extract unique, sorted parameters for the sliders
unique_g = np.sort(long_df['g'].unique())
unique_omega = np.sort(long_df['omega'].unique())

# 4. Setup the base plot
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)

theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', alpha=0.5)

# Initialize scatter plot with ALL matching points for the first slider positions
initial_data = long_df[(long_df['g'] == unique_g[0]) & (long_df['omega'] == unique_omega[0])]
point = ax.scatter(initial_data['x_norm'].values, 
                 initial_data['y_norm'].values,
                 c = initial_data['overlap'].values, 
                 cmap='viridis', 
                 s=25, alpha=0.7)

cbar = plt.colorbar(point, ax=ax)
cbar.set_label('Overlap with Neel') 

# Formatting
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xlabel(r"Re($e^{-i \phi}$)")
ax.set_ylabel(r"Im($e^{-i \phi}$)")
ax.grid(True, alpha=0.3)
ax.set_title(f"g = {unique_g[0]:.4f}, $\omega$ = {unique_omega[0]:.4f}")

# 5. Interactive sliders
ax_g = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_omega = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_g = Slider(ax=ax_g, label='g', valmin=unique_g.min(), valmax=unique_g.max(), valinit=unique_g[0], valstep=unique_g)
slider_omega = Slider(ax=ax_omega, label=r'$\omega$', valmin=unique_omega.min(), valmax=unique_omega.max(), valinit=unique_omega[0], valstep=unique_omega)

def update(val):
    current_g = slider_g.val
    current_omega = slider_omega.val
    
    # Isolate ALL points matching the current slider states
    row_data = long_df[(np.isclose(long_df['g'], current_g)) & (np.isclose(long_df['omega'], current_omega))]
    
    if not row_data.empty:
        point.set_offsets(np.c_[row_data['x_norm'].values, row_data['y_norm'].values])
        ax.set_title(f"g = {current_g:.4f}, $\omega$ = {current_omega:.4f}")
        point.set_array(row_data['overlap'].values)
    
    fig.canvas.draw_idle()

slider_g.on_changed(update)
slider_omega.on_changed(update)

plt.show()