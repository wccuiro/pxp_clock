import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load the dataset without headers
# Pandas will automatically assign integer column names: 0, 1, 2, 3...
df = pd.read_csv('../rust/decay.csv', header=None)

# 2. Reshape the data from "wide" to "long" format
# Column 0 is gp, Column 1 is gm, Column 2 is omega
gp_vals = df[0].values
gm_vals = df[1].values
omega_vals = df[2].values

# Extract real (even indices from 3 onwards) and imag (odd indices from 4 onwards)
real_cols = df.columns[3::5]
imag_cols = df.columns[4::5]
overlap_cols = df.columns[5::5]
occupation_cols = df.columns[6::5]
blockSize_cols = df.columns[7::5]

# Ensure we only take complete pairs in case the CSV has a trailing comma
min_len = min(len(real_cols), len(imag_cols))
real_vals = df[real_cols[:min_len]].values
imag_vals = df[imag_cols[:min_len]].values
overlap_vals = df[overlap_cols[:min_len]].values
occupation_vals = df[occupation_cols[:min_len]].values
blockSize_vals = df[blockSize_cols[:min_len]].values

# Repeat gp and gm and omega so they align with the flattened arrays
num_val_pairs = real_vals.shape[1]
gp_long = np.repeat(gp_vals, num_val_pairs)
gm_long = np.repeat(gm_vals, num_val_pairs)
omega_long = np.repeat(omega_vals, num_val_pairs)

# Build a clean dataframe handling ALL values seamlessly
long_df = pd.DataFrame({
    'gp': gp_long,
    'gm': gm_long,
    'omega': omega_long,
    'val_real': real_vals.flatten(),
    'val_imag': imag_vals.flatten(),
    'overlap': overlap_vals.flatten(),
    'occupation': occupation_vals.flatten(),
    'blockSize': blockSize_vals.flatten()
}).dropna() # Drops empty pairs if the CSV has rows of varying lengths

# 3. Store the base complex phase. Time evolution will be applied dynamically.
long_df['iphi'] = long_df['val_real'] + 1j * long_df['val_imag']

# Extract unique, sorted parameters for the sliders
unique_gp = np.sort(long_df['gp'].unique())
unique_gm = np.sort(long_df['gm'].unique())
unique_omega = np.sort(long_df['omega'].unique())

# 4. Setup the base plot
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.30)

theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', alpha=0.5)

# Initialize scatter plot with ALL matching points for the first slider positions
initial_data = long_df[(long_df['gp'] == unique_gp[0]) & (long_df['gm'] == unique_gm[0]) & (long_df['omega'] == unique_omega[0])]
initial_t = 1.0 # Must match slider_t valinit
initial_phase = np.exp(initial_data['iphi'].values * initial_t)

point = ax.scatter(np.real(initial_phase), 
                   np.imag(initial_phase),
                   c=initial_data['overlap'].values, 
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
ax.set_title(r"$\gamma_+$ = {unique_gp[0]:.4f}, $\gamma_-$ = {unique_gm[0]:.4f}, $\omega$ = {unique_omega[0]:.4f}".format(unique_gp=unique_gp, unique_gm=unique_gm, unique_omega=unique_omega))

# 5. Interactive sliders
ax_t = plt.axes([0.15, 0.2, 0.7, 0.03])
ax_gp = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_gm = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_omega = plt.axes([0.15, 0.05, 0.7, 0.03])


slider_t = Slider(ax=ax_t, label='t', valmin=0.0, valmax=10.0, valinit=1.0, valstep=0.01)
slider_gp = Slider(ax=ax_gp, label=r'$\gamma_+$', valmin=unique_gp.min(), valmax=unique_gp.max(), valinit=unique_gp[0], valstep=unique_gp)
slider_gm = Slider(ax=ax_gm, label=r'$\gamma_-$', valmin=unique_gm.min(), valmax=unique_gm.max(), valinit=unique_gm[0], valstep=unique_gm)
slider_omega = Slider(ax=ax_omega, label=r'$\omega$', valmin=unique_omega.min(), valmax=unique_omega.max(), valinit=unique_omega[0], valstep=unique_omega)

def update(val):
    current_gp = slider_gp.val
    current_gm = slider_gm.val
    current_omega = slider_omega.val
    current_t = slider_t.val
    
    # Isolate ALL points matching the current slider states
    row_data = long_df[(np.isclose(long_df['gp'], current_gp)) & (np.isclose(long_df['gm'], current_gm)) & (np.isclose(long_df['omega'], current_omega))]
    
    if not row_data.empty:
        # Calculate time evolution dynamically for this frame
        evolved_phase = np.exp(row_data['iphi'].values * current_t)
        new_x = np.real(evolved_phase)
        new_y = np.imag(evolved_phase)
        
        point.set_offsets(np.c_[new_x, new_y])
        ax.set_title(r"$\gamma_+$ = {current_gp:.4f}, $\gamma_-$ = {current_gm:.4f}, $\omega$ = {current_omega:.4f}, t = {current_t:.1f}".format(current_gp=current_gp, current_gm=current_gm, current_omega=current_omega, current_t=current_t))
        point.set_array(row_data['overlap'].values)
    
    fig.canvas.draw_idle()

slider_t.on_changed(update)
slider_gp.on_changed(update)
slider_gm.on_changed(update)
slider_omega.on_changed(update)

plt.show()