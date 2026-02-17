import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 1. DATA LOADING
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
                    num_modes = len(raw_data) // 3
                    for i in range(num_modes):
                        idx = i * 3
                        # Physics convention: Decay Rate = -Re(Eigenvalue)
                        decay_rate = -float(raw_data[idx]) 
                        overlap = float(raw_data[idx+1])
                        size = int(float(raw_data[idx+2]))
                        
                        data_points.append({
                            'g': g, 
                            'omega': omega, 
                            'decay_rate': decay_rate, 
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

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    # -- Static Elements --
    ax.set_xlabel(r"Decay Rate ($-\mathrm{Re}[\lambda]$)")
    ax.set_ylabel("Overlap Magnitude")
    # ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Set fixed limits based on global data extrema
    x_min, x_max = df['decay_rate'].min(), df['decay_rate'].max()
    y_min, y_max = df[df['overlap']>0]['overlap'].min(), df['overlap'].max()
    
    # Add padding to limits
    ax.set_xlim(x_min - 0.1*abs(x_min), x_max + 0.1*abs(x_max))
    ax.set_ylim(y_min * 0.5, y_max * 1.5)

    # -- Plot Objects --
    # 1. Normal Modes (Size = 1) - Blue dots
    scat_normal = ax.scatter([], [], s=40, c='royalblue', alpha=0.6, label='Normal (k=1)')
    
    # 2. Jordan Blocks (Size > 1) - Red Stars
    # We will initialize this with a colormap to distinguish sizes visually
    cmap = cm.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=1, vmax=5) # Assuming max block size ~5 usually
    scat_jordan = ax.scatter([], [], marker='*', edgecolors='black', linewidth=0.5, label='Jordan (k>1)')

    # List to hold text annotations so we can clear them each update
    annotations = []

    def update(val):
        # 1. Clear old text annotations
        for txt in annotations:
            txt.remove()
        annotations.clear()

        # 2. Get Data Slice
        target_g = min(unique_gs, key=lambda x: abs(x - slider_g.val))
        target_w = min(unique_ws, key=lambda x: abs(x - slider_w.val))
        subset = df[(df['g'] == target_g) & (df['omega'] == target_w)]
        
        # 3. Update Normal Modes
        normal = subset[subset['size'] == 1]
        if not normal.empty:
            scat_normal.set_offsets(np.column_stack((normal['decay_rate'], normal['overlap'])))
        else:
            scat_normal.set_offsets(np.zeros((0, 2)))

        # 4. Update Jordan Blocks (The important part)
        jordan = subset[subset['size'] > 1]
        if not jordan.empty:
            # Set positions
            scat_jordan.set_offsets(np.column_stack((jordan['decay_rate'], jordan['overlap'])))
            
            # Dynamic Sizes: Size 2 = 200px, Size 3 = 400px...
            sizes = 100 * ((jordan['size']*0+1 )** 1.5) 
            scat_jordan.set_sizes(sizes)
            
            # Dynamic Colors: Larger blocks = Darker Red
            colors = cmap(norm(jordan['size']))
            scat_jordan.set_color(colors)
            
            # Explicit Text Annotations
            for _, row in jordan.iterrows():
                # Place text slightly above the star
                txt = ax.text(
                    row['decay_rate'], 
                    row['overlap'] * 1.3, 
                    f"k={int(row['size'])}", 
                    fontsize=10, 
                    fontweight='bold', 
                    color='darkred',
                    ha='center'
                )
                annotations.append(txt)
        else:
            scat_jordan.set_offsets(np.zeros((0, 2)))

        ax.set_title(r"Spectrum at g={target_g:.4f}, $\omega$={target_w:.4f}".format(target_g=target_g, target_w=target_w), fontsize=14)
        fig.canvas.draw_idle()

    # -- Sliders --
    ax_g = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_w = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider_g = Slider(ax_g, 'g', min(unique_gs), max(unique_gs), valinit=init_g, valstep=unique_gs)
    slider_w = Slider(ax_w, 'omega', min(unique_ws), max(unique_ws), valinit=init_w, valstep=unique_ws)

    slider_g.on_changed(update)
    slider_w.on_changed(update)

    # Add legend manually to explain markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='Normal (k=1)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Jordan (k>1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    update(0)
    plt.show()

# Run
df = load_data('../rust/decay_12.csv')
interactive_plot(df)