import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- PRL Document Formatting Requirements ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,      # Large enough for a two-column document
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (3.4, 8.5), # 3.4 inches is standard PRL single-column width
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# [Keep your other functions like plot_eigenvalue_overlaps, get_tower_mask, etc. here]

def plot_normalized_dynamics(filename="occupation_time_alpha.csv", dt=1e-3):
    """Plots the normalized observables, with one plot per (gp, gm) set stacked vertically."""
    
    # Dictionary to group data by (gp, gm)
    # Format: grouped_data[(gp, gm)] = [{'alpha': alpha, 'omega': omega, 'time': time, 'norm_n': norm_n, ...}]
    grouped_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 5: 
                    continue
                
                alpha = float(vals[0])
                gp = float(vals[1])
                gm = float(vals[2])
                omega = float(vals[3])
                
                data = np.array(vals[4:], dtype=float).reshape(-1, 3)
                
                n_t = data[:, 0]
                nn_t = data[:, 1]
                fid_t = data[:, 2] 
                
                time = np.arange(len(n_t)) * dt
                
                n_ss, nn_ss, fid_ss = n_t[-1], nn_t[-1], fid_t[-1]
                n_0, nn_0, fid_0 = n_t[0], nn_t[0], fid_t[0]
                
                norm_n = (n_t - n_ss) / (n_0 - n_ss + 1e-16)
                norm_nn = (nn_t - nn_ss) / (nn_0 - nn_ss + 1e-16)
                norm_fid = fid_t 
                
                grouped_data[(gp, gm)].append({
                    'alpha': alpha,
                    'omega': omega,
                    'time': time,
                    'norm_n': norm_n,
                    'norm_nn': norm_nn,
                    'norm_fid': norm_fid
                })
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    if not grouped_data:
        print("No valid data found in file.")
        return

    # Sort groups by gp and gm for consistent subplot ordering
    sorted_groups = sorted(grouped_data.keys())
    num_groups = len(sorted_groups)
    
    # Define observables to iterate over
    observables = [
        ('norm_n', r"$\frac{\langle n(t) \rangle - \langle n \rangle_{ss}}{\langle n(0) \rangle - \langle n \rangle_{ss}}$", "Normalized Occupation Dynamics"),
        ('norm_nn', r"$\frac{\langle nn(t) \rangle - \langle nn \rangle_{ss}}{\langle nn(0) \rangle - \langle nn \rangle_{ss}}$", "Normalized NN Correlation"),
        ('norm_fid', r"Fidelity $F(t)$", "Fidelity Dynamics")
    ]
    
    # Standard matplotlib line styles to combine with colors
    linestyles = ['-', '--', '-.', ':']
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Generate one figure (4 subplots) per observable
    for obs_key, ylabel, title in observables:
        fig, axs = plt.subplots(num_groups, 1, sharex=True)
        if num_groups == 1:
            axs = [axs]
            
        for i, (gp, gm) in enumerate(sorted_groups):
            ax = axs[i]
            datasets = grouped_data[(gp, gm)]
            
            # Sort datasets within the subplot by alpha
            datasets.sort(key=lambda x: x['alpha'])
            
            for j, d in enumerate(datasets):
                ls = linestyles[j % len(linestyles)]
                ax.plot(d['time'], d[obs_key], label=rf"$\alpha={d['alpha']:.1f}$", linestyle=ls)
                
            ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            # --- Standard Subplot Labels for Figure Captions ---
            label_text = subplot_labels[i] if i < len(subplot_labels) else ""
            ax.text(-0.25, 1., label_text, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='left')
            
            # Add parameter label box inside the plot (top right or top left)
            textstr = rf"$\gamma_+={gp}, \gamma_-={gm}$"
            ax.text(0.05, 0.85, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
            
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            
            # Use a clean legend layout
            ax.legend(loc='upper right', frameon=True, handlelength=2.5)

        axs[-1].set_xlabel("Time $t$")
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.08) # Minimal space between shared x-axis subplots
        
        # Save as high-res PDF for LaTeX inclusion
        save_name = f"{title.replace(' ', '_').lower()}.pdf"
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == "__main__":
    plot_normalized_dynamics("../rust/occupation_time_alpha.csv", dt=1e-3)