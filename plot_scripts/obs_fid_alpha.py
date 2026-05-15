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
                # decay.csv format: q_sector, gp, gm, omega, alpha, [7 items per mode]...
                if len(vals) < 6: 
                    continue
                
                sector = float(vals[0])
                gp = float(vals[1])
                gm = float(vals[2])
                omega = float(vals[3])
                alpha = float(vals[4])
                
                # New Rust format has 7 elements per mode
                data = np.array(vals[5:], dtype=float).reshape(-1, 7)
                
                c_k = data[:, 2] + 1j * data[:, 3]
                o_k = data[:, 4] + 1j * data[:, 5]
                w_k_mag = np.abs(c_k * np.conj(o_k))
                
                grouped_data[(gp, gm, omega, alpha)].append({
                    'sector': sector,
                    'real': data[:, 0],
                    'imag': data[:, 1],
                    'overlap_mag': w_k_mag
                })
                
        for (gp, gm, omega, alpha), sector_list in grouped_data.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Eigenspectrum Overlaps: gp={gp}, gm={gm}, $\\Omega$={omega}, $\\alpha$={alpha}", fontsize=14, fontweight='bold')
            
            sector_list.sort(key=lambda x: x['sector'])
            
            for s_data in sector_list:
                sec_val = s_data['sector']
                ax1.scatter(s_data['real'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', linewidth=0.5, label=f"Sector {sec_val}")
                # Plot both positive and negative frequencies for symmetry
                ax2.scatter(s_data['imag'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', linewidth=0.5, label=f"Sector {sec_val}")
                ax2.scatter(-s_data['imag'], s_data['overlap_mag'], alpha=0.7, edgecolors='w', linewidth=0.5)
            
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


def get_tower_mask(imag_eval, W_mag, omega_0=1.33, local_ratio=0.05, global_ratio=0.01, freq_tolerance=0.25):
    mask_scars = np.zeros(len(W_mag), dtype=bool)
    
    global_max = np.max(W_mag)
    absolute_min_floor = global_max * global_ratio
    
    max_n = int(np.max(np.abs(imag_eval)) / omega_0) + 1
    
    for n in range(max_n + 1):
        target_freq = n * omega_0
        in_this_tower = np.abs(np.abs(imag_eval) - target_freq) <= freq_tolerance
        
        if np.any(in_this_tower):
            local_max = np.max(W_mag[in_this_tower])
            local_threshold = local_max * local_ratio
            
            is_prominent = (W_mag > local_threshold) & (W_mag > absolute_min_floor)
            mask_scars |= (in_this_tower & is_prominent)
            
    return mask_scars


def draw_tolerance_bands(ax_inset, max_imag, omega_0=1.33, freq_tolerance=0.25):
    max_n = int(max_imag / omega_0) + 1
    for n in range(max_n + 1):
        center = n * omega_0
        ax_inset.axvspan(center - freq_tolerance, center + freq_tolerance, color='gray', alpha=0.15, zorder=0, lw=0)
        if center != 0:
            ax_inset.axvspan(-center - freq_tolerance, -center + freq_tolerance, color='gray', alpha=0.15, zorder=0, lw=0)


def plot_scar_decomposition(filename="decay.csv", t_max=10.0, dt=1e-3):
    grouped_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 6: 
                    continue
                
                sector = float(vals[0])
                gp = float(vals[1])
                gm = float(vals[2])
                omega = float(vals[3])
                alpha = float(vals[4])
                
                data = np.array(vals[5:], dtype=float).reshape(-1, 7)
                grouped_data[(gp, gm, omega, alpha)].append(data)
                
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    time = np.arange(0, t_max, dt)

    for (gp, gm, omega, alpha), sector_data_list in grouped_data.items():
        all_data = np.vstack(sector_data_list)
        
        imag_eval = all_data[:, 1]
        lambda_k = all_data[:, 0] + 1j * imag_eval
        c_k = all_data[:, 2] + 1j * all_data[:, 3]
        o_k = all_data[:, 4] + 1j * all_data[:, 5]
        
        W_k = c_k * np.conj(o_k)
        W_mag = np.abs(W_k)
        
        mask_scars = get_tower_mask(imag_eval, W_mag, omega_0=1.33)
        mask_bulk = ~mask_scars
        
        F_exact = np.zeros_like(time, dtype=float)
        F_scars = np.zeros_like(time, dtype=float)
        F_bulk = np.zeros_like(time, dtype=float)
        
        for w, lam in zip(W_k[mask_bulk], lambda_k[mask_bulk]):
            F_bulk += np.real(w * np.exp(lam * time))
            
        for w, lam in zip(W_k[mask_scars], lambda_k[mask_scars]):
            F_scars += np.real(w * np.exp(lam * time))
            
        F_exact = F_bulk + F_scars

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time, F_exact, label="Exact Reconstruction", color='black', linewidth=2.5, zorder=3)
        ax.plot(time, F_scars, label=f"Scar Towers $\\omega_0=1.33$ ({np.sum(mask_scars)} states)", color='darkorange', linestyle='--', linewidth=2.5, zorder=4)
        ax.plot(time, F_bulk, label=f"Bulk Background ({np.sum(mask_bulk)} states)", color='steelblue', alpha=0.8, linewidth=1.5, zorder=2)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Reconstructed Fidelity $F(t)$")
        ax.set_title(f"Analytical Fidelity Decomposition ($\\omega_0=1.33$): gp={gp}, gm={gm}, $\\Omega$={omega}, $\\alpha$={alpha}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # --- Create the Inset Plot ---
        ax_inset = ax.inset_axes([0.55, 0.15, 0.40, 0.40])
        
        draw_tolerance_bands(ax_inset, np.max(np.abs(imag_eval)), omega_0=1.33, freq_tolerance=0.25)
        
        ax_inset.scatter(imag_eval[mask_bulk], W_mag[mask_bulk], color='steelblue', alpha=0.1, s=15, edgecolor='none')
        ax_inset.scatter(imag_eval[mask_scars], W_mag[mask_scars], color='darkorange', alpha=0.9, s=25, edgecolor='white', linewidth=0.5)
        
        ax_inset.set_xlabel("Imag(Eigenvalue)", fontsize=8)
        ax_inset.set_ylabel("$|W_k|$", fontsize=8)
        ax_inset.tick_params(axis='both', which='major', labelsize=7)
        ax_inset.set_title("Spectral Filter Selection", fontsize=9)
        ax_inset.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_fidelity_comparison(decay_filename="decay.csv", occ_filename="occupation_time_alpha.csv", dt=1e-3):
    numerical_data = {}
    try:
        with open(occ_filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                # format: alpha, gp, gm, omega, [n, nn, fid]...
                if len(vals) < 5: continue
                alpha = float(vals[0])
                gp = float(vals[1])
                gm = float(vals[2])
                omega = float(vals[3])
                
                data = np.array(vals[4:], dtype=float).reshape(-1, 3)
                numerical_data[(gp, gm, omega, alpha)] = data[:, 2] # 3rd column is fid
    except FileNotFoundError:
        print(f"File {occ_filename} not found.")
        return

    analytical_data = defaultdict(list)
    try:
        with open(decay_filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                if len(vals) < 6: continue
                sector = float(vals[0])
                gp = float(vals[1])
                gm = float(vals[2])
                omega = float(vals[3])
                alpha = float(vals[4])
                
                data = np.array(vals[5:], dtype=float).reshape(-1, 7)
                analytical_data[(gp, gm, omega, alpha)].append(data)
    except FileNotFoundError:
        print(f"File {decay_filename} not found.")
        return

    for params in numerical_data.keys():
        if params not in analytical_data:
            print(f"Missing analytical data for parameters {params}")
            continue
            
        gp, gm, omega, alpha = params
        fid_num = numerical_data[params]
        time = np.arange(len(fid_num)) * dt
        
        all_data = np.vstack(analytical_data[params])
        
        imag_eval = all_data[:, 1]
        lambda_k = all_data[:, 0] + 1j * imag_eval
        c_k = all_data[:, 2] + 1j * all_data[:, 3]
        o_k = all_data[:, 4] + 1j * all_data[:, 5]
        
        W_k = c_k * np.conj(o_k)
        W_mag = np.abs(W_k)
        
        mask_scars = get_tower_mask(imag_eval, W_mag, omega_0=1.33)
        mask_bulk = ~mask_scars
        
        F_ana = np.zeros_like(time, dtype=float)
        F_scars = np.zeros_like(time, dtype=float)
        
        for w, lam in zip(W_k, lambda_k):
            F_ana += np.real(w * np.exp(lam * time))
            
        for w, lam in zip(W_k[mask_scars], lambda_k[mask_scars]):
            F_scars += np.real(w * np.exp(lam * time))
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time, fid_num, label="Numerical Lindblad (Exact)", color='black', linewidth=3, zorder=1)
        ax.plot(time, F_ana, label="Analytical Reconstruction", color='red', linestyle='--', linewidth=2, zorder=2)
        ax.plot(time, F_scars, label="Analytical Scars ($\\omega_0=1.33$ Only)", color='orange', linestyle=':', linewidth=2.5, zorder=3)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Fidelity $F(t)$")
        ax.set_title(f"Numerical vs Analytical Fidelity: gp={gp}, gm={gm}, $\\Omega$={omega}, $\\alpha$={alpha}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # --- Create the Inset Plot ---
        ax_inset = ax.inset_axes([0.55, 0.15, 0.40, 0.40])
        
        draw_tolerance_bands(ax_inset, np.max(np.abs(imag_eval)), omega_0=1.33, freq_tolerance=0.25)
        
        ax_inset.scatter(imag_eval[mask_bulk], W_mag[mask_bulk], color='steelblue', alpha=0.1, s=15, edgecolor='none')
        ax_inset.scatter(imag_eval[mask_scars], W_mag[mask_scars], color='darkorange', alpha=0.9, s=25, edgecolor='white', linewidth=0.5)
        
        ax_inset.set_xlabel("Imag(Eigenvalue)", fontsize=8)
        ax_inset.set_ylabel("$|W_k|$", fontsize=8)
        ax_inset.tick_params(axis='both', which='major', labelsize=7)
        ax_inset.set_title("Spectral Filter Selection", fontsize=9)
        ax_inset.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def plot_normalized_dynamics(filename="occupation_time_alpha.csv", dt=1e-3):
    """Plots the normalized observables grouped by observable to compare parameter sets."""
    datasets = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                # format: alpha, gp, gm, omega, [n, nn, fid]...
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
                
                datasets.append({
                    'label': r"$\gamma_{+}=" + str(gp) + r", \gamma_{-}=" + str(gm) + r", \alpha=" + str(alpha) + r"$",
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
    for d in datasets:
        plt.plot(d['time'], d['norm_n'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{\langle n(t) \rangle - \langle n \rangle_{ss}}{\langle n(0) \rangle - \langle n \rangle_{ss}}$")
    plt.title("Normalized Occupation Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    for d in datasets:
        plt.plot(d['time'], d['norm_nn'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{\langle nn(t) \rangle - \langle nn \rangle_{ss}}{\langle nn(0) \rangle - \langle nn \rangle_{ss}}$")
    plt.title("Normalized NN Correlation Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    for d in datasets:
        plt.plot(d['time'], d['norm_fid'], label=d['label'], linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\frac{O(t) - O_{ss}}{O(0) - O_{ss}}$")
    plt.title("Normalized Fidelity / Overlap Dynamics")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Make sure to update the file names below to match your outputs
    plot_eigenvalue_overlaps("../rust/decay_alpha.csv")
    plot_scar_decomposition("../rust/decay_alpha.csv", t_max=10.0, dt=1e-3)
    plot_fidelity_comparison("../rust/decay_alpha.csv", "../rust/occupation_time_alpha.csv", dt=1e-3)
    plot_normalized_dynamics("../rust/occupation_time_alpha.csv", dt=1e-3)