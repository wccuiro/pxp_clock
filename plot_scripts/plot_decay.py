import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 1. DATA LOADING (Unchanged)
# ---------------------------------------------------------
def load_data(filename):
    data_points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) < 3: continue
                try:
                    g = float(parts[0])
                    omega = float(parts[1])
                    raw_data = parts[2:]
                    num_modes = len(raw_data) // 4
                    for i in range(num_modes):
                        idx = i * 4
                        # Physics convention: Decay Rate = -Re(Eigenvalue)
                        decay_rate = -float(raw_data[idx])
                        energy_oscillation = float(raw_data[idx+1])
                        overlap = float(raw_data[idx+2])
                        size = int(float(raw_data[idx+3]))
                        
                        data_points.append({
                            'g': g, 
                            'omega': omega, 
                            'decay_rate': decay_rate,
                            'energy_oscillation': energy_oscillation,
                            'overlap': overlap, 
                            'size': size
                        })
                except ValueError: continue
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found."); return pd.DataFrame()
    return pd.DataFrame(data_points)

# ---------------------------------------------------------
# 2. INTERACTIVE PLOTTER
# ---------------------------------------------------------
def interactive_plot(df):
    if df.empty: print("No data."); return

    unique_gs = sorted(df['g'].unique())
    unique_ws = sorted(df['omega'].unique())
    init_g = unique_gs[0]
    init_w = unique_ws[0]

    # CHANGED: sharex=False because Energy and Decay Rate have different units/ranges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    axes_list = [ax1, ax2]
    # Define which column to plot on X-axis for each subplot
    x_cols = ['energy_oscillation', 'decay_rate']

    # -- Global Limits Calculation --
    # Y-axis is shared (Overlap)
    y_min, y_max = df[df['overlap'] > 0]['overlap'].min(), df['overlap'].max()
    
    # X-axis limits for Plot 1 (Energy)
    e_min, e_max = df['energy_oscillation'].min(), df['energy_oscillation'].max()
    pad_e = 0.1 * (abs(e_max) if e_max != 0 else 1.0)
    
    # X-axis limits for Plot 2 (Decay)
    d_min, d_max = df['decay_rate'].min(), df['decay_rate'].max()
    pad_d = 0.1 * (abs(d_max) if d_max != 0 else 1.0)

    # -- Setup Lists to hold plot objects --
    scats_normal = []
    scats_jordan = []
    
    # Styling
    cmap = cm.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=1, vmax=5)

    # -- Initialize plots --
    for i, ax in enumerate(axes_list):
        ax.set_ylabel("Overlap Magnitude")
        # ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_ylim(y_min * 0.5, y_max * 1.5)

        # Set specific X limits and labels based on the column
        if x_cols[i] == 'energy_oscillation':
            ax.set_xlabel(r"Energy Oscillation ($\mathrm{Im}[\lambda]$)")
            ax.set_xlim(e_min - pad_e, e_max + pad_e)
        else:
            ax.set_xlabel(r"Decay Rate ($-\mathrm{Re}[\lambda]$)")
            ax.set_xlim(d_min - pad_d, d_max + pad_d)

        # Create Scatter Objects
        sn = ax.scatter([], [], s=40, c='royalblue', alpha=0.6, label='Normal (k=1)')
        scats_normal.append(sn)
        
        sj = ax.scatter([], [], marker='*', edgecolors='black', linewidth=0.5, label='Jordan (k>1)')
        scats_jordan.append(sj)
        
        # Legend (only needs to be added once really, but adding to both is fine)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='Normal (k=1)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Jordan (k>1)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # Annotations list
    annotations = []

    def update(val):
        # 1. Clear old annotations
        for txt in annotations:
            txt.remove()
        annotations.clear()

        # 2. Filter Data
        target_g = min(unique_gs, key=lambda x: abs(x - slider_g.val))
        target_w = min(unique_ws, key=lambda x: abs(x - slider_w.val))
        subset = df[(df['g'] == target_g) & (df['omega'] == target_w)]
        
        normal = subset[subset['size'] == 1]
        jordan = subset[subset['size'] > 1]

        # 3. Update BOTH plots
        for i, ax in enumerate(axes_list):
            current_x_col = x_cols[i] # 'energy_oscillation' for ax1, 'decay_rate' for ax2

            # Update Normal Modes
            if not normal.empty:
                scats_normal[i].set_offsets(np.column_stack((normal[current_x_col], normal['overlap'])))
            else:
                scats_normal[i].set_offsets(np.zeros((0, 2)))

            # Update Jordan Blocks
            if not jordan.empty:
                scats_jordan[i].set_offsets(np.column_stack((jordan[current_x_col], jordan['overlap'])))
                
                # Styles
                sizes = 100 * ((jordan['size']*0+1 )** 1.5) 
                scats_jordan[i].set_sizes(sizes)
                colors = cmap(norm(jordan['size']))
                scats_jordan[i].set_color(colors)
                
                # Annotations
                for _, row in jordan.iterrows():
                    txt = ax.text(
                        row[current_x_col], 
                        row['overlap'] * 1.3, 
                        f"k={int(row['size'])}", 
                        fontsize=9, 
                        fontweight='bold', 
                        color='darkred',
                        ha='center'
                    )
                    annotations.append(txt)
            else:
                scats_jordan[i].set_offsets(np.zeros((0, 2)))

        fig.suptitle(r"Spectrum at g={:.4f}, $\omega$={:.4f}".format(target_g, target_w), fontsize=14)
        fig.canvas.draw_idle()

    # -- Sliders --
    ax_g = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_w = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider_g = Slider(ax_g, 'g', min(unique_gs), max(unique_gs), valinit=init_g, valstep=unique_gs)
    slider_w = Slider(ax_w, 'omega', min(unique_ws), max(unique_ws), valinit=init_w, valstep=unique_ws)

    slider_g.on_changed(update)
    slider_w.on_changed(update)

    update(0)
    plt.show()

# Run
df = load_data('../rust/decay_12_merged.csv')
interactive_plot(df)