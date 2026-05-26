import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PRL Document Formatting Requirements ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,          # Legible for a two-column document layout
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (3.4, 8.5), # Standard PRL single-column width (3.4 inches)
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True
})

def plot_symmetry_comparison():
    # Load data files
    file_sym = '../rust/occupation_time.csv'
    file_asym = '../rust/occupation_time_asymmetric.csv'

    try:
        data_sym = pd.read_csv(file_sym, header=None)
        data_asym = pd.read_csv(file_asym, header=None)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Extract parameters (first 3 columns: gamma_plus, gamma_minus, omega)
    params_sym = data_sym.iloc[:, 0:3]

    # Extract time series data
    series_sym = data_sym.iloc[:, 3:]
    series_asym = data_asym.iloc[:, 3:]

    # Calculate the number of time steps (3 observables per step)
    N = series_sym.shape[1] // 3
    time = np.linspace(0, 20.0, N)

    # We assume 4 parameter sets based on the desired 4 subplots
    num_plots = min(4, len(data_sym))
    
    fig, axs = plt.subplots(num_plots, 1, sharex=True)
    if num_plots == 1:
        axs = [axs]

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # Standard colors/linestyles for clarity in print and grayscale
    color_sym = 'black'
    ls_sym = '-'
    color_asym = 'red'
    ls_asym = '--'

    for i in range(num_plots):
        ax = axs[i]
        
        # Extract parameters for this row
        gp, gm, omega = params_sym.iloc[i]
        
        # Extract Fidelity values (index 2, 5, 8...)
        fid_sym = series_sym.iloc[i, 2::3].values
        fid_asym = series_asym.iloc[i, 2::3].values
        
        ax.plot(time[:N//2], fid_sym[:N//2], color=color_sym, linestyle=ls_sym, label="Uniform")
        ax.plot(time[:N//2], fid_asym[:N//2], color=color_asym, linestyle=ls_asym, label="Staggered")
        
        # --- Standard Subplot Labels ---
        label_text = subplot_labels[i]
        ax.text(-0.25, 0.93, label_text, transform=ax.transAxes,
                fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='left')
        
        # Parameter label box shifted right to clear the subplot letter identifier
        textstr = rf"$\gamma_+={gp}, \gamma_-={gm}$"
        ax.text(0.18, 0.93, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        ax.set_ylabel("Fidelity $F(t)$")
        ax.grid(True, alpha=0.2)
        
        # Place legend in the top subplot only to avoid cluttering the figure
        if i == 0:
            ax.legend(loc='upper right', frameon=True, handlelength=2.5)

    # Shared x-axis label on the bottom plot
    axs[-1].set_xlabel("Time $t$")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08) # Compact spacing for shared x-axes
    
    # Save as high-res PDF for LaTeX inclusion
    plt.savefig('symmetry_vs_asymmetry_fidelity.pdf', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_symmetry_comparison()