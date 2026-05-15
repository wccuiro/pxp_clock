import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 1. DATA LOADING (Updated for New Rust Format with alpha)
# ---------------------------------------------------------
def load_data(filename):
    data_points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                
                # New format requires 5 header parameters: q_sector, gp, gm, omega, alpha
                if len(parts) < 5: 
                    continue
                
                try:
                    q_sector = float(parts[0])
                    gp = float(parts[1])
                    gm = float(parts[2])
                    omega = float(parts[3])
                    alpha = float(parts[4])
                    
                    raw_data = parts[5:]
                    
                    # New Rust chunks are 7 elements long:
                    # real, abs_imag, c_k, s_k, occ_c, occ_s, block_size
                    num_modes = len(raw_data) // 7
                    
                    for i in range(num_modes):
                        idx = i * 7
                        
                        real_eval = float(raw_data[idx])
                        abs_imag_eval = float(raw_data[idx+1])
                        c_k = float(raw_data[idx+2])
                        s_k = float(raw_data[idx+3])
                        occ_c = float(raw_data[idx+4])
                        occ_s = float(raw_data[idx+5])
                        block_size = int(float(raw_data[idx+6]))

                        # Physics convention: Decay Rate = -Re(Eigenvalue)
                        decay_rate = -real_eval
                        energy_oscillation = abs_imag_eval
                        
                        # Calculate magnitudes from cosine/sine components
                        overlap = np.sqrt(c_k**2 + s_k**2)
                        occupation = np.sqrt(occ_c**2 + occ_s**2)

                        data_points.append({
                            'q_sector': q_sector,
                            'gp': gp,
                            'gm': gm,
                            'omega': omega,
                            'alpha': alpha,
                            'decay_rate': decay_rate,
                            'energy_oscillation': energy_oscillation,
                            'overlap': overlap,
                            'occupation': occupation,
                            'size': block_size
                        })
                except ValueError: 
                    continue
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return pd.DataFrame()
        
    return pd.DataFrame(data_points)

# ---------------------------------------------------------
# 2. INTERACTIVE PLOTTER
# ---------------------------------------------------------
def interactive_plot(df):
    if df.empty: 
        print("No data found."); return

    # Extract unique values for the sliders
    unique_qs = sorted(df['q_sector'].unique())
    unique_gps = sorted(df['gp'].unique())
    unique_gms = sorted(df['gm'].unique())
    unique_ws = sorted(df['omega'].unique())
    unique_alphas = sorted(df['alpha'].unique())
    
    init_q = unique_qs[0]
    init_gp = unique_gps[0]
    init_gm = unique_gms[0]
    init_w = unique_ws[0]
    init_alpha = unique_alphas[-1] # Default to alpha=1.0 if it exists

    # sharex=False because Energy and Decay Rate have different units/ranges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    
    # Increase bottom padding to fit 5 sliders
    plt.subplots_adjust(bottom=0.35, hspace=0.3)
    
    axes_list = [ax1, ax2]
    x_cols = ['energy_oscillation', 'decay_rate']

    # -- Global Limits Calculation --
    y_min, y_max = df[df['overlap'] > 0]['overlap'].min(), df['overlap'].max()
    
    # X-axis limits for Plot 1 (Energy - mirrored to be symmetric)
    e_max = df['energy_oscillation'].max()
    e_max = abs(e_max)
    pad_e = 0.1 * (e_max if e_max != 0 else 1.0)
    
    # X-axis limits for Plot 2 (Decay Rate)
    d_min, d_max = df['decay_rate'].min(), df['decay_rate'].max()
    pad_d = 0.1 * (abs(d_max) if d_max != 0 else 1.0)

    # -- Setup Lists to hold plot objects --
    scats_normal = []
    scats_jordan = []
    
    cmap = cm.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=1, vmax=5)

    for i, ax in enumerate(axes_list):
        ax.set_ylabel("Overlap Magnitude")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_ylim(max(y_min * 0.5, 0), y_max * 1.5)

        if x_cols[i] == 'energy_oscillation':
            ax.set_xlabel(r"Energy Oscillation ($\mathrm{Im}[\lambda]$)")
            ax.set_xlim(-e_max - pad_e, e_max + pad_e)
        else:
            ax.set_xlabel(r"Decay Rate ($-\mathrm{Re}[\lambda]$)")
            ax.set_xlim(d_min - pad_d, d_max + pad_d)

        sn = ax.scatter([], [], s=40, c='royalblue', alpha=0.6, label='Normal (k=1)')
        scats_normal.append(sn)
        
        sj = ax.scatter([], [], marker='*', edgecolors='black', linewidth=0.5, label='Jordan (k>1)')
        scats_jordan.append(sj)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='Normal (k=1)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Jordan (k>1)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    annotations = []

    def update(val):
        for txt in annotations:
            txt.remove()
        annotations.clear()

        # Retrieve Slider values
        target_q = min(unique_qs, key=lambda x: abs(x - slider_q.val))
        target_gp = min(unique_gps, key=lambda x: abs(x - slider_gp.val))
        target_gm = min(unique_gms, key=lambda x: abs(x - slider_gm.val))
        target_w = min(unique_ws, key=lambda x: abs(x - slider_w.val))
        target_alpha = min(unique_alphas, key=lambda x: abs(x - slider_alpha.val))
        
        subset = df[(df['q_sector'] == target_q) &
                    (df['gp'] == target_gp) & 
                    (df['gm'] == target_gm) & 
                    (df['omega'] == target_w) &
                    (df['alpha'] == target_alpha)]

        normal = subset[subset['size'] == 1]
        jordan = subset[subset['size'] > 1]

        for i, ax in enumerate(axes_list):
            current_x_col = x_cols[i] 

            # Mirror the 'Energy Oscillation' (Imaginary Part) to show symmetric spectrum
            if current_x_col == 'energy_oscillation':
                # Normal Modes
                if not normal.empty:
                    nx = np.concatenate((normal[current_x_col], -normal[current_x_col]))
                    ny = np.concatenate((normal['overlap'], normal['overlap']))
                    scats_normal[i].set_offsets(np.column_stack((nx, ny)))
                else:
                    scats_normal[i].set_offsets(np.zeros((0, 2)))

                # Jordan Blocks
                if not jordan.empty:
                    jx = np.concatenate((jordan[current_x_col], -jordan[current_x_col]))
                    jy = np.concatenate((jordan['overlap'], jordan['overlap']))
                    jsize = np.concatenate((jordan['size'], jordan['size']))
                    
                    scats_jordan[i].set_offsets(np.column_stack((jx, jy)))
                    
                    sizes = 100 * ((jsize * 0 + 1) ** 1.5)
                    scats_jordan[i].set_sizes(sizes)
                    colors = cmap(norm(jsize))
                    scats_jordan[i].set_color(colors)
                    
                    for _, row in jordan.iterrows():
                        # Positive side
                        txt1 = ax.text(row[current_x_col], row['overlap'] * 1.3, f"k={int(row['size'])}", 
                                       fontsize=9, fontweight='bold', color='darkred', ha='center')
                        annotations.append(txt1)
                        # Negative side
                        if row[current_x_col] != 0:
                            txt2 = ax.text(-row[current_x_col], row['overlap'] * 1.3, f"k={int(row['size'])}", 
                                           fontsize=9, fontweight='bold', color='darkred', ha='center')
                            annotations.append(txt2)
                else:
                    scats_jordan[i].set_offsets(np.zeros((0, 2)))
                    
            else:
                # Decay Rate Plot (Do NOT Mirror)
                if not normal.empty:
                    scats_normal[i].set_offsets(np.column_stack((normal[current_x_col], normal['overlap'])))
                else:
                    scats_normal[i].set_offsets(np.zeros((0, 2)))

                if not jordan.empty:
                    scats_jordan[i].set_offsets(np.column_stack((jordan[current_x_col], jordan['overlap'])))
                    sizes = 100 * ((jordan['size']*0+1 )** 1.5) 
                    scats_jordan[i].set_sizes(sizes)
                    colors = cmap(norm(jordan['size']))
                    scats_jordan[i].set_color(colors)
                    
                    for _, row in jordan.iterrows():
                        txt = ax.text(row[current_x_col], row['overlap'] * 1.3, f"k={int(row['size'])}", 
                                      fontsize=9, fontweight='bold', color='darkred', ha='center')
                        annotations.append(txt)
                else:
                    scats_jordan[i].set_offsets(np.zeros((0, 2)))

        fig.suptitle(r"Q-Sector={}, $\gamma_+$={:.4f}, $\gamma_-$={:.4f}, $\omega$={:.4f}, $\alpha$={:.4f}".format(
            int(target_q), target_gp, target_gm, target_w, target_alpha), fontsize=14)
        fig.canvas.draw_idle()

    # -- 5 Sliders Layout --
    ax_q = plt.axes([0.2, 0.25, 0.6, 0.03])
    ax_gp = plt.axes([0.2, 0.20, 0.6, 0.03])
    ax_gm = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_w = plt.axes([0.2, 0.10, 0.6, 0.03])
    ax_alpha = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    slider_q = Slider(ax_q, r'$Q$-Sector', min(unique_qs), max(unique_qs), valinit=init_q, valstep=unique_qs)
    slider_gp = Slider(ax_gp, r'$\gamma_+$', min(unique_gps), max(unique_gps), valinit=init_gp, valstep=unique_gps)
    slider_gm = Slider(ax_gm, r'$\gamma_-$', min(unique_gms), max(unique_gms), valinit=init_gm, valstep=unique_gms)
    slider_w = Slider(ax_w, r'$\omega$', min(unique_ws), max(unique_ws), valinit=init_w, valstep=unique_ws)
    slider_alpha = Slider(ax_alpha, r'$\alpha$', min(unique_alphas), max(unique_alphas), valinit=init_alpha, valstep=unique_alphas)

    slider_q.on_changed(update)
    slider_gp.on_changed(update)
    slider_gm.on_changed(update)
    slider_w.on_changed(update)
    slider_alpha.on_changed(update)

    update(0)
    plt.show()

# Run
df = load_data('../rust/decay_alpha.csv')
interactive_plot(df)