#!/usr/bin/env python3
"""
pxp_fib_mc_plots.py

MC / trajectories version — seed handling fixed for modern QuTiP (don't pass `seed=` into mcsolve).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
import qutip as qt
import time
import math
import random

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
            if m.nnz > 0:
                c_ops_q.append(qt.Qobj(m, dims=[[d],[d]]))
    if gamma_plus > 0:
        for p in plus_sparse:
            if p.nnz > 0:
                c_ops_q.append(qt.Qobj(p, dims=[[d],[d]]))
    return minus_sparse, plus_sparse, c_ops_q

def build_number_ops_reduced(L, states):
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
# Master evolution (MC) and plotting
# ---------------------------
def run_fib_mc_and_plot(L=8, Omega=1.0, gamma_minus=0.2, gamma_plus=0.0,
                        t_final=10.0, n_steps=201, periodic=False, start_with_one=True,
                        n_traj=400, batch_size=100, progress=True, seed=None):
    """
    n_traj: total number of trajectories to average
    batch_size: run mcsolve in chunks of this many trajectories (useful to limit memory / get progress)
    progress can be: True/False/"tqdm"/"enhanced" etc. It is passed into options={"progress_bar": progress}.
    seed can be None or an integer. If integer, different sub-seeds are used for each batch by calling
    numpy.random.seed(...) and random.seed(...) before mcsolve.
    """
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

    # wrap H into Qobj and prepare c_ops list
    H = qt.Qobj(H_sparse, dims=[[d],[d]])
    c_ops = c_ops_q  # list of qutip Qobj collapse ops

    # build n_i and Z_i as Qobj
    n_qops = [qt.Qobj(m, dims=[[d],[d]]) for m in n_sparse_list]
    I = qt.qeye(d)
    z_qops = [2 * n - I for n in n_qops]

    # Build pair operators n_i n_j for i<j
    pair_indices = []
    pair_nn_qops = []
    for i in range(L):
        for j in range(i+1, L):
            pair_indices.append((i, j))
            pair_nn_qops.append(n_qops[i] * n_qops[j])

    # initial Néel state in reduced basis (ket)
    neel_int = neel_bitstring(L, start_with_one=start_with_one)
    if neel_int not in index:
        raise ValueError("Néel bitstring not in reduced basis (check L and pattern).")
    psi0_red = qt.basis(d, index[neel_int])

    # Prepare e_ops for mcsolve:
    # order: [P0, Z_total, n_i (L), pair_nn ...]
    P0 = psi0_red * psi0_red.dag()
    Z_total = sum(z_qops) / float(L)
    e_ops = [P0, Z_total]
    e_ops.extend(n_qops)
    e_ops.extend(pair_nn_qops)

    # safety warning if too many expectations requested
    max_eops_warn = 2000
    if len(e_ops) > max_eops_warn:
        print(f"Warning: you're requesting {len(e_ops)} expectation ops. This may be slow / memory-heavy.")

    # time list
    tlist = np.linspace(0.0, t_final, n_steps)

    # --- run trajectories in batches and average expectations ---
    if n_traj <= 0:
        raise ValueError("n_traj must be positive.")
    batch_size = int(min(batch_size, n_traj))
    n_batches = math.ceil(n_traj / batch_size)

    if progress:
        print(f"Running mcsolve: total trajectories {n_traj} in {n_batches} batch(es) of up to {batch_size}")

    # accumulator for expectations: shape (n_eops, n_steps)
    n_eops = len(e_ops)
    expect_acc = np.zeros((n_eops, len(tlist)), dtype=np.complex128)
    traj_done = 0
    last_result = None

    tstart = time.time()
    for b in range(n_batches):
        this_batch = batch_size if (b < n_batches - 1) else (n_traj - traj_done)
        if progress:
            print(f"  Batch {b+1}/{n_batches}: running {this_batch} trajectories ...", end="", flush=True)

        # prepare options dict for the new API
        options_dict = {"progress_bar": progress}

        # manage seed per-batch by seeding numpy + python random BEFORE calling mcsolve
        batch_seed = None
        if isinstance(seed, int):
            batch_seed = int(seed + b)
            np.random.seed(batch_seed)
            random.seed(batch_seed)
        elif seed is None:
            # don't set seed (random)
            pass
        else:
            # if user passed another type (e.g., a sequence), don't override
            pass

        # mcsolve returns averaged expectations across trajectories in this batch
        # NOTE: do NOT pass 'seed=' into mcsolve (modern QuTiP rejects it).
        res = qt.mcsolve(H, psi0_red, tlist, c_ops, e_ops=e_ops,
                         ntraj=this_batch, options=options_dict)
        last_result = res
        expect_acc += np.array(res.expect)
        traj_done += this_batch

        if progress:
            print(" done")

    tend = time.time()
    # final average
    expect_avg = np.real_if_close(expect_acc / float(n_traj))
    if progress:
        print(f"mcsolve total time: {tend-tstart:.3f} s, averaged over {n_traj} trajectories")

    # Parse expectations (same ordering as before)
    idx = 0
    fidelity = expect_avg[idx]; idx += 1
    mean_mag = expect_avg[idx]; idx += 1

    n_expect = expect_avg[idx: idx + L]; idx += L

    n_pairs = len(pair_nn_qops)
    nn_expect_flat = expect_avg[idx: idx + n_pairs]; idx += n_pairs

    # reshape nn_expect into (L, L, n_steps) symmetric matrix
    nn_expect = np.zeros((L, L, len(tlist)))
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

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(tlist, fidelity, lw=2)
    ax1.set_ylabel("Fidelity with initial Néel")
    ax1.set_title(f"PXP reduced (Fibonacci), L={L}, d={d}, gamma-={gamma_minus}, gamma+={gamma_plus}")
    ax1.grid(True)

    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
    ax2.plot(tlist, mean_mag, lw=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"Mean magnetization $\langle Z\rangle$")
    ax2.grid(True)

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

    # Return results and averaged expectations
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
        "mcsolve_result_last_batch": last_result,
        "expect_avg": expect_avg
    }

# ---------------------------
# Main guard
# ---------------------------
if __name__ == "__main__":
    # parameters (edit as needed)
    L = 20
    Omega = 10.0
    gamma_minus = 0.1   # decay rate
    gamma_plus  = 1.0   # pump rate
    t_final = 5.0
    n_steps = 100
    periodic = True
    start_with_one = True   # Néel = 1010...

    # MC parameters
    n_traj = 400
    batch_size = 100   # run 4 batches of 100 each (400 total)
    seed = None        # set integer for reproducibility (per-batch seeds derived from this)

    out = run_fib_mc_and_plot(L=L, Omega=Omega,
                              gamma_minus=gamma_minus, gamma_plus=gamma_plus,
                              t_final=t_final, n_steps=n_steps,
                              periodic=periodic, start_with_one=start_with_one,
                              n_traj=n_traj, batch_size=batch_size, progress=True, seed=seed)
