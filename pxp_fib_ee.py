#!/usr/bin/env python3
"""
pxp_fib_steady_entropies.py

Find steady state in Fibonacci reduced basis for different L,
compute global von Neumann entropy S(rho) and entanglement entropy
(entropy of left-half reduced density matrix) by embedding the reduced
density matrix into the full 2^L Hilbert space and partial-tracing.

Requirements: numpy, scipy, qutip
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import qutip as qt
import time
import sys
import os

# ---------------------------
# Utilities (same mapping as previous scripts)
# ---------------------------
def bit_at(x, i):
    return (x >> i) & 1

def flip_bit(x, i):
    return x ^ (1 << i)

def fibonacci_basis(L):
    states = []
    for s in range(1 << L):
        if (s & (s << 1)) == 0:
            states.append(s)
    index = {s: i for i, s in enumerate(states)}
    return states, index

def neel_bitstring(L, start_with_one=True):
    s = 0
    for i in range(L):
        bit = (i % 2) if start_with_one else ((i + 1) % 2)
        if bit:
            s |= (1 << i)
    return s

# ---------------------------
# Reduced PXP Hamiltonian & jump operators (Fibonacci basis)
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
                    # match many-body sigmax convention: off-diagonal = 1, so H element = Omega
                    rows.append(idx); cols.append(jdx); data.append(Omega)
                    rows.append(jdx); cols.append(idx); data.append(Omega)
    H = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
    return H

def build_plus_minus_ops_reduced_qobj(L, states, index, gamma_minus=0.0, gamma_plus=0.0, periodic=False):
    """
    Build Qobj collapse operators (reduced basis) and also return sparse lists if needed.
    minus: sqrt(gamma_minus) |0><1|_i
    plus:  sqrt(gamma_plus)  |1><0|_i only if resulting state is valid
    Return list of qutip.Qobj c_ops.
    """
    d = len(states)
    sqrt_minus = np.sqrt(gamma_minus)
    sqrt_plus  = np.sqrt(gamma_plus)
    c_ops_q = []

    # minus ops
    if gamma_minus > 0:
        for i in range(L):
            rows_m, cols_m, data_m = [], [], []
            for idx, s in enumerate(states):
                if bit_at(s, i) == 1:
                    s2 = s & ~(1 << i)
                    jdx = index.get(s2)
                    if jdx is not None:
                        rows_m.append(jdx); cols_m.append(idx); data_m.append(sqrt_minus)
            if data_m:
                mat = coo_matrix((data_m, (rows_m, cols_m)), shape=(d, d)).tocsr()
                c_ops_q.append(qt.Qobj(mat, dims=[[d],[d]]))

    # plus ops
    if gamma_plus > 0:
        for i in range(L):
            rows_p, cols_p, data_p = [], [], []
            for idx, s in enumerate(states):
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
            if data_p:
                mat = coo_matrix((data_p, (rows_p, cols_p)), shape=(d, d)).tocsr()
                c_ops_q.append(qt.Qobj(mat, dims=[[d],[d]]))

    return c_ops_q

# ---------------------------
# Embedding reduced density matrix into full space
# ---------------------------
def build_projection_matrix_full_to_reduced(L, states):
    """Return (2^L x d) numpy array Phi whose columns are full-space product-state vectors for reduced basis states.
       Column j is the full-space basis vector for the bitstring states[j].
       Ordering: site 0 is factor 0 in tensor (site0 ⊗ site1 ⊗ ...)."""
    d = len(states)
    full_dim = 2 ** L
    cols = []
    for s in states:
        factors = []
        for i in range(L):
            if bit_at(s, i) == 1:
                factors.append(qt.basis(2, 1))
            else:
                factors.append(qt.basis(2, 0))
        vec = qt.tensor(factors).full().reshape((full_dim,))
        cols.append(vec)
    Phi = np.column_stack(cols)  # shape (2^L, d)
    return Phi

# ---------------------------
# Main routine: steady state and entropies for list of L
# ---------------------------
def compute_steady_entropies(L_list,
                             Omega=1.0,
                             gamma_minus=0.2,
                             gamma_plus=0.0,
                             periodic=False,
                             start_with_one=True,
                             L_max_embed=14,
                             savefile=None):
    """
    For each L in L_list compute steady state rho_ss (reduced), global entropy S(rho_ss),
    and entanglement entropy S_A for left-half subsystem (by embedding to full space).
    L_max_embed: maximum L for which embedding to full 2^L will be attempted (safety).
    """
    results = []
    for L in L_list:
        print(f"\n\n=== L = {L} ===")
        states, index = fibonacci_basis(L)
        d = len(states)
        full_dim = 2 ** L
        print(f"reduced dimension d = {d}, full 2^L = {full_dim}")

        # Build reduced H and c_ops
        H_sparse = build_pxp_hamiltonian_reduced(L, states, index, Omega=Omega, periodic=periodic)
        H = qt.Qobj(H_sparse, dims=[[d],[d]])
        c_ops = build_plus_minus_ops_reduced_qobj(L, states, index,
                                                  gamma_minus=gamma_minus, gamma_plus=gamma_plus,
                                                  periodic=periodic)

        # Compute steady state:
        print("Computing steady state (qutip.steadystate)...", end=' ', flush=True)
        t0 = time.time()
        rho_ss = None
        # Try direct method first; fallback to 'eigen' if direct fails
        try:
            rho_ss = qt.steadystate(H, c_ops, method='direct')
        except Exception as e:
            print(f"\ndirect steadystate failed: {e}\nTrying 'eigen' method...", end=' ', flush=True)
            try:
                rho_ss = qt.steadystate(H, c_ops, method='eigen')
            except Exception as e2:
                print(f"\n'eigen' steadystate failed: {e2}\nTrying Liouvillian nullspace eigen (sparse) ...", end=' ', flush=True)
                # fallback: build Liouvillian and find zero eigenvector via sparse solver
                L_super = qt.liouvillian(H, c_ops)
                # Use scipy.sparse.linalg.eigs on the superoperator matrix (dense conversion may be heavy)
                # Convert to csr for eigen solver
                try:
                    import scipy.sparse.linalg as spla
                    Lmat = L_super.data.tocsr()
                    # find eigenvector with eigenvalue closest to 0
                    vals, vecs = spla.eigs(Lmat, k=1, sigma=0.0)
                    v = vecs[:, 0]
                    rho_vec = v.reshape((-1, 1))
                    # Convert from vectorized form to density matrix
                    rho_ss = qt.operator_to_vector(L_super).dag()  # not used — keep minimal fallback
                    raise RuntimeError("Sparse fallback is not fully automated here; please try smaller L or ensure qutip.steadystate works.")
                except Exception as e3:
                    raise RuntimeError("Failed to compute steady state with all fallback methods. Error: " + str(e3))
        t1 = time.time()
        print(f"done in {t1-t0:.3f} s")

        # Ensure rho_ss is a density matrix Qobj
        if not isinstance(rho_ss, qt.Qobj):
            raise RuntimeError("steadystate did not return a Qobj density matrix.")

        # global von Neumann entropy
        S_global = qt.entropy_vn(rho_ss, base=2)  # base-2 (bits); change base if you prefer nats
        purity = float((rho_ss * rho_ss).tr())

        # # Prepare embedding to full space for entanglement entropy if allowed
        # if L <= L_max_embed:
        #     print("Embedding reduced rho_ss into full 2^L space and computing partial trace...", end=' ', flush=True)
        #     Phi = build_projection_matrix_full_to_reduced(L, states)  # shape (2^L, d)
        #     # convert rho_ss to dense numpy
        #     rho_red_mat = rho_ss.full()  # (d,d) array
        #     # build full density matrix: rho_full = Phi * rho_red * Phi^dagger
        #     rho_full = Phi @ rho_red_mat @ Phi.conj().T  # shape (2^L, 2^L) dense
        #     # convert to Qobj with tensor dims so ptrace works
        #     dims = [[2]*L, [2]*L]
        #     rho_full_q = qt.Qobj(rho_full, dims=dims)
        #     # choose left half as subsystem A
        #     LA = L // 2
        #     subsys_A = list(range(LA))
        #     # compute reduced rho_A
        #     rho_A = qt.ptrace(rho_full_q, subsys_A)
        #     # entanglement entropy (von Neumann) of rho_A
        #     S_ent = qt.entropy_vn(rho_A, base=2)
        #     print(" done.")
        # else:
        #     S_ent = np.nan
        #     print(f"Skipping embedding for L={L} (L_max_embed={L_max_embed}). Set L_max_embed higher if you want to embed.")
        S_ent = np.nan
        # store results
        results.append({
            "L": L,
            "d": d,
            "full_dim": full_dim,
            "rho_ss": rho_ss,
            "S_global": float(S_global),
            "S_entanglement_left_half": float(S_ent) if not np.isnan(S_ent) else np.nan,
            "purity": float(purity)
        })

        # print summary line
        print(f"L={L:2d}  d={d:5d}  S_global={S_global:.6f} bits  S_ent(left half)={S_ent if not np.isnan(S_ent) else 'skipped'}  purity={purity:.6e}")

    # Optionally save
    if savefile:
        print(f"\nSaving results to {savefile} (numpy .npz)...")
        # prepare arrays
        Ls = np.array([r["L"] for r in results])
        ds = np.array([r["d"] for r in results])
        S_globals = np.array([r["S_global"] for r in results])
        S_ents = np.array([r["S_entanglement_left_half"] for r in results])
        purities = np.array([r["purity"] for r in results])
        np.savez_compressed(savefile, Ls=Ls, ds=ds, S_globals=S_globals, S_ents=S_ents, purities=purities)
        print("Saved.")

    return results

# ---------------------------
# If run as script: example usage
# ---------------------------
if __name__ == "__main__":
    # user parameters: tune these
    L_list = [4, 6, 8, 10]   # chain lengths to analyze; keep small if embedding enabled
    Omega = 1.0
    gamma_minus = 0.2
    gamma_plus  = 0.0
    periodic = True
    start_with_one = True

    # do not try to embed for L > this (safety)
    L_max_embed = 14

    # optional: filename to save numeric results
    savefile = "pxp_fib_steady_entropies.npz"

    t_start = time.time()
    results = compute_steady_entropies(L_list, Omega=Omega, gamma_minus=gamma_minus, gamma_plus=gamma_plus,
                                       periodic=periodic, start_with_one=start_with_one,
                                       L_max_embed=L_max_embed, savefile=savefile)
    t_end = time.time()
    print(f"\nAll done in {t_end - t_start:.2f} s.")
    # print table
    print("\nSummary:")
    print(" L   d     S_global(bits)   S_ent_left_half(bits)    purity")
    for r in results:
        print(f"{r['L']:2d} {r['d']:5d}   {r['S_global']:12.6f}     {r['S_entanglement_left_half'] if not np.isnan(r['S_entanglement_left_half']) else 'skipped':>8}   {r['purity']: .6e}")
