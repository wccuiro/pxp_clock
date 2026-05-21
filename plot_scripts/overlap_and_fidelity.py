import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_overlap_vs_imaginary(filename="decay.csv"):
    """Plots Overlap vs Imaginary parts of the eigenvalues in a 2x2 grid."""
    grouped_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                sector, gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                data = np.array(vals[4:], dtype=float).reshape(-1, 8)
                
                # Reconstruct complex numbers for c_k and o_k
                c_k = data[:, 2] + 1j * data[:, 3]
                o_k = data[:, 4] + 1j * data[:, 5]
                w_k_mag = np.abs(c_k * np.conj(o_k))
                
                grouped_data[(gp, gm, omega)].append({
                    'sector': sector,
                    'imag': data[:, 1],
                    'overlap_mag': w_k_mag
                })
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    if not grouped_data:
        print("No data found.")
        return

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    fig.suptitle("Eigenspectrum Overlaps vs Imaginary Part", fontsize=16, fontweight='bold')
    
    for idx, ((gp, gm, omega), sector_list) in enumerate(grouped_data.items()):
        if idx >= 4:
            print("Warning: More than 4 parameter combinations found. Only plotting the first 4.")
            break
            
        ax = axes[idx]
        
        # Sort by sector to keep colors/labels consistent
        sector_list.sort(key=lambda x: x['sector'])
        
        for s_data in sector_list:
            sec_val = int(s_data['sector'])
            ax.scatter(s_data['imag'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', 
                       linewidth=0.5, label=f"Sector {sec_val}")
            
        ax.set_title(rf"$\gamma_+={gp}$, $\gamma_-={gm}$, $\Omega={omega}$")
        ax.set_xlabel("Imaginary(Eigenvalue)")
        ax.set_ylabel(r"$|W_k|$ (Overlap Magnitude)")
        ax.grid(True, alpha=0.3)
        
        # Prevent the legend from crowding the plot if there are too many sectors
        if len(sector_list) <= 10:
            ax.legend(fontsize='small')
            
    # Clean up any empty subplots if there happens to be fewer than 4 combinations
    for i in range(len(grouped_data), 4):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()


def plot_fidelity_in_time(filename="occupation_time.csv", dt=1e-3):
    """Plots only the exact numerical fidelity dynamics from Lindblad dynamics."""
    datasets = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2])
                data = np.array(vals[3:], dtype=float).reshape(-1, 3)
                
                # Extract numerical fidelity directly (3rd element in the 3-element chunks)
                fid_t = data[:, 2] 
                time = np.arange(len(fid_t)) * dt
                
                datasets.append({
                    'label': rf"$\gamma_+={gp}, \gamma_-={gm}, \Omega={omega}$",
                    'time': time,
                    'fid_t': fid_t
                })
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    if not datasets:
        print("No valid data found in file.")
        return

    # Plot all extracted fidelities on a single figure
    plt.figure(figsize=(10, 6))
    for d in datasets:
        plt.plot(d['time'], d['fid_t'], label=d['label'], linewidth=2, alpha=0.8)
        
    plt.xlabel("Time")
    plt.ylabel("Fidelity $F(t)$")
    plt.title("Exact Numerical Lindblad Fidelity Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_overlap_vs_imaginary("../rust/decay_12.csv")
    plot_fidelity_in_time("../rust/occupation_time.csv", dt=1e-3)