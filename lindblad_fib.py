#!/usr/bin/env python3
"""
pxp_fib_master_plots.py

Reduced (Fibonacci) PXP Lindblad master equation:
 - Fibonacci basis (no adjacent 1s)
 - PXP Hamiltonian H = Omega * sum_i P_{i-1} X_i P_{i+1} (reduced basis)
 - Two incoherent channels:
      L_-^i = sqrt(gamma_minus) |0><1|_i  (decay)
      L_+^i = sqrt(gamma_plus)  |1><0|_i  (pump)  -- only when result is valid
 - Master equation only (qutip.mesolve)
 - Initial state: Néel (1010...)
 - Outputs: fidelity(t), mean magnetization(t), heatmaps of <n_i n_j> and connected C_ij at final time
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
import qutip as qt
import time

# ---------------------------
# Utilities
# ---------------------------
def bit_at(x, i):
    return (x >> i) & 1

def flip_bit(x, i):
    return x ^ (1 << i)

def fibonacci_basis(L):
    """Return list of integers (bitstrings) without adjacent 1s, and index map."""
    states = []
    for s in range(1 << L):
        if (s & (s << 1)) == 0:
            states.append(s)
    index = {s: i for i, s in enumerate(states)}
    return states, index

def neel_bitstring(L, start_with_one=True):
    """Return integer bitstring for Néel pattern: start_with_one=True => 1010..."""
    s = 0
    for i in range(L):
        bit = (i % 2) if start_with_one else ((i + 1) % 2)
        if bit:
            s |= (1 << i)
    return s

# ---------------------------
# Operator constructors (reduced)
# ---------------------------
def build_pxp_hamiltonian_reduced(L, states, index, Omega=1.0, periodic=False):
    """
    Build reduced PXP Hamiltonian in Fibonacci basis.
    Convention: Off-diagonal matrix element for flipping site i is 'Omega',
    so that construction matches many-body sigmax convention H = Omega * sum P X P.
    """
    d = len(states)
    rows, cols, data = [], [], []
    for idx, s in enumerate(states):
        for i in range(L):
            left = (i - 1) % L if periodic else i - 1
            right = (i + 1) % L if periodic else i + 1
            left_ok = (left < 0 or left >= L) or (bit_at(s, left) == 0)
            right_ok = (right < 0 or right >= L) or (bit_at(s, right) == 0)
            if left_ok and right_ok:
                s2 = flip_bit(s, i)
                jdx = index.get(s2)
                if jdx is not None:
                    val = Omega
                    rows.append(idx); cols.append(jdx); data.append(val)
                    rows.append(jdx); cols.append(idx); data.append(val)
    H = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
    return H

def build_plus_minus_ops_reduced(L, states, index, gamma_minus=0.0, gamma_plus=0.0, periodic=False):
    """
    Build lists of sparse operators in reduced basis:
    - minus_ops[i] : sqrt(gamma_minus) |0><1|_i (always valid if starting from valid state)
    - plus_ops[i]  : sqrt(gamma_plus)  |1><0|_i (only if neighbors allow creation of 1)
    Returns: (list_minus_sparse, list_plus_sparse, list_all_qobj_c_ops)
    """
    d = len(states)
    sqrt_minus = np.sqrt(gamma_minus)
    sqrt_plus = np.sqrt(gamma_plus)
    minus_sparse = []
    plus_sparse = []
    for i in range(L):
        rows_m, cols_m, data_m = [], [], []
        rows_p, cols_p, data_p = [], [], []
        for idx, s in enumerate(states):
            # minus: if bit i == 1 -> flip to 0
            if bit_at(s, i) == 1:
                s2 = s & ~(1 << i)
                jdx = index.get(s2)
                if jdx is not None:
                    rows_m.append(jdx); cols_m.append(idx); data_m.append(sqrt_minus)
            # plus: if bit i == 0 and neighbors allow placing a 1
            if bit_at(s, i) == 0:
                left = (i - 1) % L if periodic else i - 1
                right = (i + 1) % L if periodic else i + 1
                left_ok = (left < 0 or left >= L) or (bit_at(s, left) == 0)
                right_ok = (right < 0 or right >= L) or (bit_at(s, right) == 0)
                if left_ok and right_ok:
                    s2 = s | (1 << i)
                    jdx = index.get(s2)
                    if jdx is not None:
                        rows_p.append(jdx); cols_p.append(idx); data_p.append(sqrt_plus)
        minus_sparse.append(coo_matrix((data_m, (rows_m, cols_m)), shape=(d, d)).tocsr() if data_m else csr_matrix((d,d)))
        plus_sparse.append(coo_matrix((data_p, (rows_p, cols_p)), shape=(d, d)).tocsr() if data_p else csr_matrix((d,d)))
    # combine into qobj list (only include nonzero ones)
    c_ops_q = []
    if gamma_minus > 0:
        for m in minus_sparse:
            c_ops_q.append(qt.Qobj(m, dims=[[d],[d]]))
    if gamma_plus > 0:
        for p in plus_sparse:
            c_ops_q.append(qt.Qobj(p, dims=[[d],[d]]))
    return minus_sparse, plus_sparse, c_ops_q

def build_number_ops_reduced(L, states):
    """Return list of sparse n_i operators in reduced basis."""
    d = len(states)
    ops = []
    for i in range(L):
        rows, cols, data = [], [], []
        for idx, s in enumerate(states):
            if bit_at(s, i) == 1:
                rows.append(idx); cols.append(idx); data.append(1.0)
        ops.append(coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr() if data else csr_matrix((d,d)))
    return ops

# ---------------------------
# Master evolution and plotting
# ---------------------------
def run_fib_master_and_plot(L=8, Omega=1.0, gamma_minus=0.2, gamma_plus=0.0,
                            t_final=10.0, n_steps=201, periodic=False, start_with_one=True):
    states, index = fibonacci_basis(L)
    d = len(states)
    print(f"L={L}, reduced dimension d={d} (2^{L} = {2**L})")

    # Build reduced Hamiltonian and collapse ops
    t0 = time.time()
    H_sparse = build_pxp_hamiltonian_reduced(L, states, index, Omega=Omega, periodic=periodic)
    minus_sparse, plus_sparse, c_ops_q = build_plus_minus_ops_reduced(L, states, index,
                                                                      gamma_minus=gamma_minus,
                                                                      gamma_plus=gamma_plus,
                                                                      periodic=periodic)
    n_sparse_list = build_number_ops_reduced(L, states)
    t1 = time.time()
    print(f"Built operators in {t1-t0:.3f} s")

    # wrap H into Qobj (keep sparse -> Qobj handles it)
    H = qt.Qobj(H_sparse, dims=[[d],[d]])
    c_ops = c_ops_q  # list of qutip Qobj collapse ops

    # build n_i and Z_i as Qobj
    n_qops = [qt.Qobj(m, dims=[[d],[d]]) for m in n_sparse_list]
    I = qt.qeye(d)
    z_qops = [2 * n - I for n in n_qops]

    # Build pair operators n_i n_j for i<j (we need these to compute <n_i n_j>)
    pair_indices = []
    pair_nn_qops = []
    for i in range(L):
        for j in range(i+1, L):
            pair_indices.append((i, j))
            pair_nn_qops.append(n_qops[i] * n_qops[j])

    # initial Néel state in reduced basis
    neel_int = neel_bitstring(L, start_with_one=start_with_one)
    if neel_int not in index:
        raise ValueError("Néel bitstring not in reduced basis (check L and pattern).")
    psi0_red = qt.basis(d, index[neel_int])
    P0 = psi0_red * psi0_red.dag()
    rho0 = P0

    # Prepare e_ops for mesolve:
    # order chosen so parsing is straightforward: [P0, Z_total, n_i (L), pair_nn ...]
    Z_total = sum(z_qops) / float(L)   # mean magnetization operator
    e_ops = [P0, Z_total]
    e_ops.extend(n_qops)               # L elements
    e_ops.extend(pair_nn_qops)         # L*(L-1)/2 elements

    # safety warning if too many expectations requested
    max_eops_warn = 1000
    if len(e_ops) > max_eops_warn:
        print(f"Warning: you're requesting {len(e_ops)} expectation ops. This may be slow / memory-heavy.")

    # time list
    tlist = np.linspace(0.0, t_final, n_steps)

    print("Running qutip.mesolve (master equation)...")
    tstart = time.time()
    # no need to store states — we asked for expectations via e_ops
    result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops, options=qt.Options(nsteps=10000))
    tend = time.time()
    print(f"mesolve done in {tend - tstart:.3f} s")

    expect = np.array(result.expect)   # shape (len(e_ops), n_steps)

    # Parse expectations
    idx = 0
    fidelity = np.real_if_close(expect[idx]); idx += 1  # fidelity vs time
    mean_mag = np.real_if_close(expect[idx]); idx += 1  # mean magnetization vs time

    # n_i expectations
    n_expect = np.real_if_close(expect[idx: idx + L]); idx += L

    # pair nn expectations (flat)
    n_pairs = len(pair_nn_qops)
    nn_expect_flat = np.real_if_close(expect[idx: idx + n_pairs]); idx += n_pairs

    # reshape nn_expect into (L, L, n_steps) symmetric matrix
    nn_expect = np.zeros((L, L, len(tlist)))
    # diagonal
    for i in range(L):
        nn_expect[i, i, :] = n_expect[i]
    k = 0
    for (i, j) in pair_indices:
        nn_expect[i, j, :] = nn_expect_flat[k]
        nn_expect[j, i, :] = nn_expect_flat[k]
        k += 1

    # connected correlator C_ij(t) = <n_i n_j> - <n_i><n_j>
    C_nn = np.zeros_like(nn_expect)
    for t_idx in range(len(tlist)):
        ni = n_expect[:, t_idx]
        C_nn[:, :, t_idx] = nn_expect[:, :, t_idx] - np.outer(ni, ni)

    # Plot results: fidelity, mean magnetization, and correlator heatmaps at final time
    fig = plt.figure(figsize=(11, 8))

    # fidelity
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(tlist, fidelity, lw=2)
    ax1.set_ylabel("Fidelity with initial Néel")
    ax1.set_title(f"PXP reduced (Fibonacci), L={L}, d={d}, gamma-={gamma_minus}, gamma+={gamma_plus}")
    ax1.grid(True)

    # mean magnetization
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
    ax2.plot(tlist, mean_mag, lw=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"Mean magnetization $\langle Z\rangle$")
    ax2.grid(True)

    # # (optional) small diagnostics: show first few n_i as faint lines (not required but sometimes useful)
    # ax3 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
    # max_show = min(6, L)
    # for i in range(max_show):
    #     ax3.plot(tlist, n_expect[i], lw=1.1, alpha=0.9, label=f"n_{i}")
    # ax3.set_xlabel("Time")
    # ax3.set_ylabel(r"$\langle n_i\rangle$ (diagnostic)")
    # ax3.legend(fontsize="small", ncol=2)
    # ax3.grid(True)

    # heatmaps at final time
    t_final_idx = -1
    nn_final = nn_expect[:, :, t_final_idx]
    C_final = C_nn[:, :, t_final_idx]

    ax4 = plt.subplot2grid((3, 2), (2, 0))
    im1 = ax4.imshow(nn_final, origin='lower', interpolation='nearest', aspect='auto')
    ax4.set_title(rf"$\langle n_i n_j\rangle$ at t={tlist[t_final_idx]:.3f}")
    ax4.set_xlabel("j"); ax4.set_ylabel("i")
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot2grid((3, 2), (2, 1))
    im2 = ax5.imshow(C_final, origin='lower', interpolation='nearest', aspect='auto')
    ax5.set_title(rf"Connected $C_{{ij}}$ at t={tlist[t_final_idx]:.3f}")
    ax5.set_xlabel("j"); ax5.set_ylabel("i")
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Return relevant data in case the caller wants to analyze further
    return {
        "L": L,
        "d": d,
        "states": states,
        "index": index,
        "H_sparse": H_sparse,
        "c_ops": c_ops,
        "tlist": tlist,
        "fidelity": fidelity,
        "mean_magnetization": mean_mag,
        "n_expect": n_expect,
        "nn_expect": nn_expect,
        "C_nn": C_nn,
        "mesolve_result": result
    }

# ---------------------------
# Main guard
# ---------------------------
if __name__ == "__main__":
    # parameters (edit as needed)
    L = 8
    Omega = 10.0
    gamma_minus = 0.0   # decay rate
    gamma_plus  = 0.0   # pump rate
    t_final = 15.0
    n_steps = 10000
    periodic = True
    start_with_one = True   # Néel = 1010...

    out = run_fib_master_and_plot(L=L, Omega=Omega,
                                  gamma_minus=gamma_minus, gamma_plus=gamma_plus,
                                  t_final=t_final, n_steps=n_steps,
                                  periodic=periodic, start_with_one=start_with_one)
