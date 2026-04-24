import numpy as np
import matplotlib.pyplot as plt

def plot_eigenvalue_overlaps(filename="decay.csv"):
    """Plots Overlap vs Real and Imaginary parts of the eigenvalues, one figure per parameter set."""
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2])
                
                # Reshape into blocks of 5: [Re, Im, Overlap, Occ, Block]
                data = np.array(vals[3:], dtype=float).reshape(-1, 5)
                
                real_evals = data[:, 0]
                imag_evals = data[:, 1]
                overlaps = data[:, 2]
                
                # Create a new figure specifically for THIS parameter set
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"Eigenspectrum Overlaps: gp={gp}, gm={gm}, $\\omega$={omega}", fontsize=14, fontweight='bold')
                
                # Plot 1: Real vs Overlap
                ax1.scatter(real_evals, overlaps, alpha=0.7, color='blue', edgecolors='w', linewidth=0.5)
                ax1.set_xlabel("Real(Eigenvalue)")
                ax1.set_ylabel("Overlap")
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Imaginary vs Overlap
                ax2.scatter(imag_evals, overlaps, alpha=0.7, color='red', edgecolors='w', linewidth=0.5)
                ax2.set_xlabel("Imaginary(Eigenvalue)")
                ax2.set_ylabel("Overlap")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
    except FileNotFoundError:
        print(f"File {filename} not found.")

def plot_normalized_dynamics(filename="occupation_time.csv", dt=1e-4):
    """Plots the normalized observables grouped by observable to compare parameter sets."""
    datasets = []
    
    # 1. Read all the data into memory first
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2])
                
                # Reshape into blocks of 3: [n, nn, fid]
                data = np.array(vals[3:], dtype=float).reshape(-1, 3)
                
                n_t = data[:, 0]
                nn_t = data[:, 1]
                fid_t = data[:, 2] 
                
                time = np.arange(len(n_t)) * dt
                
                # Calculate Steady state and Initial state
                n_ss, nn_ss, fid_ss = n_t[-1], nn_t[-1], fid_t[-1]
                n_0, nn_0, fid_0 = n_t[0], nn_t[0], fid_t[0]
                
                # Normalize (adding 1e-16 to avoid division by zero)
                norm_n = (n_t - n_ss) / (n_0 - n_ss + 1e-16)
                norm_nn = (nn_t - nn_ss) / (nn_0 - nn_ss + 1e-16)
                norm_fid = (fid_t - fid_ss) / (fid_0 - fid_ss + 1e-16)
                
                # Store the processed data
                datasets.append({
                    'label': r"$\gamma_{+}=" + str(gp) + r", \gamma_{-}=" + str(gm) + r", \Omega=" + str(omega) + r"$",
                    'time': time,
                    'norm_n': norm_n,
                    'norm_nn': norm_nn,
                    'norm_fid': norm_fid
                })
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    if not datasets:
        print("No valid data found in file.")
        return

    # 2. Plotting: One figure per observable
    
    # --- Figure 1: Occupation ---
    plt.figure(figsize=(10, 6))
    for d in datasets[1::]:
        plt.plot(d['time'], d['norm_n'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{\langle n(t) \rangle - \langle n \rangle_{ss}}{\langle n(0) \rangle - \langle n \rangle_{ss}}$")
    plt.title("Normalized Occupation Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Figure 2: Nearest-Neighbor Correlation ---
    plt.figure(figsize=(10, 6))
    for d in datasets[1::]:
        plt.plot(d['time'], d['norm_nn'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{\langle nn(t) \rangle - \langle nn \rangle_{ss}}{\langle nn(0) \rangle - \langle nn \rangle_{ss}}$")
    plt.title("Normalized NN Correlation Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Figure 3: Fidelity (Overlap) ---
    plt.figure(figsize=(10, 6))
    for d in datasets[1::]:
        plt.plot(d['time'], d['norm_fid'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{O(t) - O_{ss}}{O(0) - O_{ss}}$")
    plt.title("Normalized Fidelity / Overlap Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()


# Execute the plotters
if __name__ == "__main__":
    plot_eigenvalue_overlaps("../rust/decay.csv")
    plot_normalized_dynamics("../rust/occupation_time.csv", dt=1e-4)