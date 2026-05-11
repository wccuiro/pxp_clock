import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_eigenvalue_overlaps(filename="decay.csv"):
    """Plots Overlap vs Real and Imaginary parts of the eigenvalues."""
    grouped_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                sector, gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                data = np.array(vals[4:], dtype=float).reshape(-1, 8)
                
                c_k = data[:, 2] + 1j * data[:, 3]
                o_k = data[:, 4] + 1j * data[:, 5]
                w_k_mag = np.abs(c_k * np.conj(o_k))
                
                grouped_data[(gp, gm, omega)].append({
                    'sector': sector,
                    'real': data[:, 0],
                    'imag': data[:, 1],
                    'overlap_mag': w_k_mag
                })
                
        for (gp, gm, omega), sector_list in grouped_data.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Eigenspectrum Overlaps: gp={gp}, gm={gm}, $\\omega$={omega}", fontsize=14, fontweight='bold')
            
            sector_list.sort(key=lambda x: x['sector'])
            
            for s_data in sector_list:
                sec_val = s_data['sector']
                ax1.scatter(s_data['real'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', linewidth=0.5, label=f"Sector {sec_val}")
                ax2.scatter(s_data['imag'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', linewidth=0.5, label=f"Sector {sec_val}")
            
            ax1.set_xlabel("Real(Eigenvalue)")
            ax1.set_ylabel("$|W_k|$ (Overlap Magnitude)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel("Imaginary(Eigenvalue)")
            ax2.set_ylabel("$|W_k|$ (Overlap Magnitude)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
                
    except FileNotFoundError:
        print(f"File {filename} not found.")


def get_tower_mask(imag_eval, W_mag, omega_0=1.33, threshold_ratio=0.015, freq_tolerance=0.25):
    """
    Identifies distinct frequency towers using the strict physical constant omega_0 = 1.33.
    """
    threshold = np.max(W_mag) * threshold_ratio
    is_prominent = W_mag > threshold
    
    # Check distance to the nearest integer multiple of omega_0
    remainder = np.abs(imag_eval) % omega_0
    
    # A mode is in the tower if the remainder is close to 0 OR close to omega_0
    in_tower = (remainder < freq_tolerance) | ((omega_0 - remainder) < freq_tolerance)
    
    return in_tower & is_prominent


def plot_scar_decomposition(filename="decay.csv", t_max=10.0, dt=1e-3):
    """
    Analytically reconstructs the fidelity, applying a strict frequency-domain comb filter.
    """
    grouped_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                sector, gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                data = np.array(vals[4:], dtype=float).reshape(-1, 8)
                grouped_data[(gp, gm, omega)].append(data)
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    time = np.arange(0, t_max, dt)

    for (gp, gm, omega), sector_data_list in grouped_data.items():
        all_data = np.vstack(sector_data_list)
        
        lambda_k = all_data[:, 0] + 1j * all_data[:, 1]
        c_k = all_data[:, 2] + 1j * all_data[:, 3]
        o_k = all_data[:, 4] + 1j * all_data[:, 5]
        
        W_k = c_k * np.conj(o_k)
        W_mag = np.abs(W_k)
        
        # Apply the explicit 1.33 physical frequency tower isolation
        mask_scars = get_tower_mask(all_data[:, 1], W_mag, omega_0=1.33)
        mask_bulk = ~mask_scars
        
        F_exact = np.zeros_like(time, dtype=float)
        F_scars = np.zeros_like(time, dtype=float)
        F_bulk = np.zeros_like(time, dtype=float)
        
        for w, lam in zip(W_k[mask_bulk], lambda_k[mask_bulk]):
            F_bulk += np.real(w * np.exp(lam * time))
            
        for w, lam in zip(W_k[mask_scars], lambda_k[mask_scars]):
            F_scars += np.real(w * np.exp(lam * time))
            
        F_exact = F_bulk + F_scars

        plt.figure(figsize=(10, 6))
        plt.plot(time, F_exact, label="Exact Reconstruction", color='black', linewidth=2.5, zorder=3)
        plt.plot(time, F_scars, label=f"Scar Towers $\\omega_0=1.33$ ({np.sum(mask_scars)} states)", color='darkorange', linestyle='--', linewidth=2.5, zorder=4)
        plt.plot(time, F_bulk, label=f"Bulk Background ({np.sum(mask_bulk)} states)", color='steelblue', alpha=0.8, linewidth=1.5, zorder=2)
        
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
        plt.xlabel("Time")
        plt.ylabel("Reconstructed Fidelity $F(t)$")
        plt.title(f"Analytical Fidelity Decomposition ($\\omega_0=1.33$): gp={gp}, gm={gm}, $\\Omega$={omega}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_fidelity_comparison(decay_filename="decay.csv", occ_filename="occupation_time.csv", dt=1e-3):
    """
    Cross-references the analytical tower reconstruction with the numerical Lindblad evolution.
    """
    numerical_data = {}
    try:
        with open(occ_filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: continue
                gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2])
                data = np.array(vals[3:], dtype=float).reshape(-1, 3)
                numerical_data[(gp, gm, omega)] = data[:, 2] 
    except FileNotFoundError:
        print(f"File {occ_filename} not found.")
        return

    analytical_data = defaultdict(list)
    try:
        with open(decay_filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: continue
                sector, gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                data = np.array(vals[4:], dtype=float).reshape(-1, 8)
                analytical_data[(gp, gm, omega)].append(data)
    except FileNotFoundError:
        print(f"File {decay_filename} not found.")
        return

    for params in numerical_data.keys():
        if params not in analytical_data:
            print(f"Missing analytical data for parameters {params}")
            continue
            
        gp, gm, omega = params
        fid_num = numerical_data[params]
        time = np.arange(len(fid_num)) * dt
        
        all_data = np.vstack(analytical_data[params])
        lambda_k = all_data[:, 0] + 1j * all_data[:, 1]
        c_k = all_data[:, 2] + 1j * all_data[:, 3]
        o_k = all_data[:, 4] + 1j * all_data[:, 5]
        
        W_k = c_k * np.conj(o_k)
        W_mag = np.abs(W_k)
        
        # Apply the explicit 1.33 physical frequency tower isolation
        mask_scars = get_tower_mask(all_data[:, 1], W_mag, omega_0=1.33)
        
        F_ana = np.zeros_like(time, dtype=float)
        F_scars = np.zeros_like(time, dtype=float)
        
        for w, lam in zip(W_k, lambda_k):
            F_ana += np.real(w * np.exp(lam * time))
            
        for w, lam in zip(W_k[mask_scars], lambda_k[mask_scars]):
            F_scars += np.real(w * np.exp(lam * time))
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(time, fid_num, label="Numerical Lindblad (Exact)", color='black', linewidth=3, zorder=1)
        plt.plot(time, F_ana, label="Analytical Reconstruction", color='red', linestyle='--', linewidth=2, zorder=2)
        plt.plot(time, F_scars, label="Analytical Scars ($\\omega_0=1.33$ Only)", color='orange', linestyle=':', linewidth=2.5, zorder=3)
        
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel("Fidelity $F(t)$")
        plt.title(f"Numerical vs Analytical Fidelity: gp={gp}, gm={gm}, $\\Omega$={omega}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_normalized_dynamics(filename="occupation_time.csv", dt=1e-4):
    """Plots the normalized observables grouped by observable to compare parameter sets."""
    datasets = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 4: 
                    continue
                
                gp, gm, omega = float(vals[0]), float(vals[1]), float(vals[2])
                data = np.array(vals[3:], dtype=float).reshape(-1, 3)
                
                n_t = data[:, 0]
                nn_t = data[:, 1]
                fid_t = data[:, 2] 
                
                time = np.arange(len(n_t)) * dt
                
                n_ss, nn_ss, fid_ss = n_t[-1], nn_t[-1], fid_t[-1]
                n_0, nn_0, fid_0 = n_t[0], nn_t[0], fid_t[0]
                
                norm_n = (n_t - n_ss) / (n_0 - n_ss + 1e-16)
                norm_nn = (nn_t - nn_ss) / (nn_0 - nn_ss + 1e-16)
                norm_fid = fid_t 
                
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

if __name__ == "__main__":
    plot_eigenvalue_overlaps("../rust/decay.csv")
    plot_scar_decomposition("../rust/decay.csv", t_max=10.0, dt=1e-3)
    plot_fidelity_comparison("../rust/decay.csv", "../rust/occupation_time.csv", dt=1e-3)
    plot_normalized_dynamics("../rust/occupation_time.csv", dt=1e-3)