import numpy as np
import matplotlib.pyplot as plt

# PRL Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10
})

filename = "../rust/decay.csv"

# 1. Read and parse the complex overlap data
data_by_param = {}
with open(filename, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 4: continue
        q_sector, gp, gm, omega = parts[0:4]
        
        # Only process Q=0 and Q=pi (4) sectors
        if q_sector not in ['0', '4']: continue
        
        param_key = (gp, gm, omega)
        if param_key not in data_by_param:
            data_by_param[param_key] = {'lam': [], 'w': []}
            
        eigen_data = parts[4:]
        for i in range(0, len(eigen_data), 8):
            if i + 7 < len(eigen_data):
                try:
                    re_val = float(eigen_data[i])
                    im_val = float(eigen_data[i+1])
                    c_re = float(eigen_data[i+2])
                    c_im = float(eigen_data[i+3])
                    o_re = float(eigen_data[i+4])
                    o_im = float(eigen_data[i+5])
                    
                    c_k = complex(c_re, c_im)
                    o_k = complex(o_re, o_im)
                    
                    # Strictly gauge-invariant overlap logic (c_k * o_k^*)
                    w_k = c_k * np.conj(o_k)
                    
                    data_by_param[param_key]['lam'].append(complex(re_val, im_val))
                    data_by_param[param_key]['w'].append(w_k)
                except ValueError:
                    pass

# 2. Process and plot each unique parameter set
for (gp, gm, omega), pdata in data_by_param.items():
    lambdas = np.array(pdata['lam'])
    weights = np.array(pdata['w'])
    weights_mag = np.abs(weights)
    
    ss_idx = np.argmin(np.abs(lambdas))
    mask = np.ones(len(lambdas), dtype=bool)
    mask[ss_idx] = False
    
    non_ss_indices = np.where(mask)[0]
    sorted_non_ss = non_ss_indices[np.argsort(weights_mag[mask])]
    top_8_idx = sorted_non_ss[-8:]
    
    # --- SMART GROUPING LOGIC FOR LABELS ---
    groups = {}
    for idx in top_8_idx:
        val = np.abs(lambdas[idx].imag)
        found_key = None
        for k in groups.keys():
            if abs(k - val) < 1e-4:
                found_key = k
                break
        if found_key is None:
            groups[val] = []
            found_key = val
        groups[found_key].append(idx)
        
    sorted_keys = sorted(groups.keys())
    
    labels = {}         # Holds "1a", "1b"
    base_labels = {}    # Holds just "1"
    label_counter = 1
    
    for k in sorted_keys:
        members = groups[k]
        if k < 1e-4: 
            # Purely real state (no conjugate pair)
            for idx in members:
                labels[idx] = str(label_counter)
                base_labels[idx] = str(label_counter)
                label_counter += 1
        else:
            # Complex conjugate pair
            members_sorted = sorted(members, key=lambda x: lambdas[x].imag, reverse=True)
            chars = ['a', 'b', 'c', 'd']
            for i, idx in enumerate(members_sorted):
                labels[idx] = f"{label_counter}{chars[i]}"
                base_labels[idx] = str(label_counter) # Store only the number
            label_counter += 1
    # ---------------------------------------

    fig = plt.figure(figsize=(3.375, 5.5))
    
    # =======================================================
    # Subplot 1: Overlap vs Imaginary Axis (Labels: 1a, 1b)
    # =======================================================
    ax1 = fig.add_subplot(2, 1, 1)
    
    ax1.scatter(lambdas.imag, weights_mag, color='gray', alpha=0.4, s=5, zorder=1)
    ax1.scatter(lambdas[ss_idx].imag, weights_mag[ss_idx], marker='*', color='gold', edgecolor='black', s=150, zorder=3, label='SS')
    ax1.text(lambdas[ss_idx].imag, weights_mag[ss_idx]*1.05 + 0.01, 'SS', color='black', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for idx in top_8_idx:
        ax1.scatter(lambdas[idx].imag, weights_mag[idx], facecolors='none', edgecolors='red', s=60, lw=1.2, zorder=2)
    #     ax1.text(lambdas[idx].imag, weights_mag[idx]+0.01, labels[idx], color='red', fontsize=8, fontweight='bold', ha='center', va='bottom')
        
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')
        
    ax1.set_xlabel(r'Imaginary Eigenvalue Im($\lambda$)')
    ax1.set_ylabel(r'Overlap')
    ax1.set_title(rf'$\gamma_+={gp}, \gamma_-={gm}$')
    ax1.set_ylim(-0.02, np.max(weights_mag)*1.3)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # =======================================================
    # Subplot 2: Overlap vs Real Axis (Labels: Base integer)
    # =======================================================
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax2.scatter(lambdas.real, weights_mag, color='gray', alpha=0.4, s=5, zorder=1)
    ax2.scatter(lambdas[ss_idx].real, weights_mag[ss_idx], marker='*', color='gold', edgecolor='black', s=150, zorder=3, label='SS')
    ax2.text(lambdas[ss_idx].real, weights_mag[ss_idx]*1.05 + 0.01, 'SS', color='black', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    bottom_points = {}
    for idx in top_8_idx:
        x = lambdas[idx].real
        y = weights_mag[idx]
        found = False
        for (bx, by) in bottom_points.keys():
            if abs(bx - x) < 1e-4 and abs(by - y) < 1e-4:
                # Merge logic using only the base number
                if base_labels[idx] not in bottom_points[(bx, by)]:
                    bottom_points[(bx, by)].append(base_labels[idx])
                found = True
                break
        if not found:
            bottom_points[(x, y)] = [base_labels[idx]]
            
    for (x, y), lbls in bottom_points.items():
        combined_label = ", ".join(sorted(lbls))
        ax2.scatter(x, y, facecolors='none', edgecolors='red', s=60, lw=1.2, zorder=2)
        # ax2.text(x, y*1.05 + 0.01, combined_label, color='red', fontsize=8, fontweight='bold', ha='center', va='bottom')
        
    ax2.set_xlabel(r'Real Eigenvalue Re($\lambda$) [Decay Rate]')
    ax2.set_ylabel(r'Overlap')
    ax2.set_xlim(np.min(lambdas.real)*1.05, 0.05)
    ax2.set_ylim(-0.02, np.max(weights_mag)*1.3)
    
    # ax2.set_yscale('log')
    
    # ax2.set_xscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'prl_re_im_pairs_gp{gp}_gm{gm}.png', dpi=300, bbox_inches='tight')
    # plt.close()