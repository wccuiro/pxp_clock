#!/usr/bin/env python3
"""
pxp_fib_steady_scars_analysis.py

Compute steady state in Fibonacci reduced basis and analyze whether steady state
is composed of PXP scars.

- Build reduced PXP Hamiltonian H (Fibonacci basis)
- Build Lindblad collapse ops (gamma_minus, gamma_plus)
- Compute steady state rho_ss via qutip.steadystate
- Diagonalize H to get eigenstates |E_alpha>
- Identify 'scar candidates' as eigenstates with largest overlap with |Neel>
- Compute:
    * F_alpha = <E_alpha | rho_ss | E_alpha> (fidelity w/ each scar eigenstate)
    * w_scar = sum F_alpha over selected scar eigenstates (weight in scar subspace)
    * purity and von Neumann entropy of rho_ss
    * optionally: entanglement entropy of left half (uses reduced-basis partial trace function)
- Plot:
    * p(E) = diag(rho_ss in energy basis) vs E
    * bar of top fidelities / scar weights

Author: adapted to your Fibonacci-reduced conventions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import qutip as qt
import time

# --------------------------
# Basic bit / Fibonacci helpers (same conventions)
# --------------------------
def bit_at(x, i): return (x >> i) & 1
def flip_bit(x, i): return x ^ (1 << i)

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

# --------------------------
# Build reduced PXP H and collapse ops (reduced basis)
# --------------------------
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
                    # choose convention: H_{ij} = Omega  (matches many-body sigmax with off-diagonal 1)
                    rows.append(idx); cols.append(jdx); data.append(Omega)
                    rows.append(jdx); cols.append(idx); data.append(Omega)
    H = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
    return H

def build_plus_minus_ops_reduced_qobj(L, states, index, gamma_minus=0.0, gamma_plus=0.0, periodic=False):
    d = len(states)
    sqrt_minus = np.sqrt(gamma_minus)
    sqrt_plus  = np.sqrt(gamma_plus)
    c_ops = []
    # minus (decay 1->0)
    if gamma_minus > 0:
        for i in range(L):
            rows, cols, data = [], [], []
            for idx, s in enumerate(states):
                if bit_at(s, i) == 1:
                    s2 = s & ~(1 << i)
                    jdx = index.get(s2)
                    if jdx is not None:
                        rows.append(jdx); cols.append(idx); data.append(sqrt_minus)
            if len(data):
                mat = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
                c_ops.append(qt.Qobj(mat, dims=[[d],[d]]))
    # plus (pump 0->1 only if allowed)
    if gamma_plus > 0:
        for i in range(L):
            rows, cols, data = [], [], []
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
                            rows.append(jdx); cols.append(idx); data.append(sqrt_plus)
            if len(data):
                mat = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
                c_ops.append(qt.Qobj(mat, dims=[[d],[d]]))
    return c_ops

# --------------------------
# partial trace in reduced basis (left half) - direct method described earlier
# --------------------------
def reduced_rho_left(L, states, index, rho_reduced, L_A=None, return_qobj=True):
    if L_A is None:
        L_A = L // 2
    L_B = L - L_A

    if isinstance(rho_reduced, qt.Qobj):
        rho_mat = np.array(rho_reduced.full(), dtype=complex)
    else:
        rho_mat = np.array(rho_reduced, dtype=complex)

    d = len(states)
    if rho_mat.shape != (d,d):
        raise ValueError("rho size mismatch")

    left_states, left_index = fibonacci_basis(L_A)
    right_states, right_index = fibonacci_basis(L_B)
    dA = len(left_states); dB = len(right_states)

    # mapping global -> (a,b)
    global_idx_mat = -np.ones((dA, dB), dtype=int)
    for gidx, s in enumerate(states):
        left_part = s & ((1 << L_A) - 1)
        right_part = s >> L_A
        if left_part in left_index and right_part in right_index:
            global_idx_mat[left_index[left_part], right_index[right_part]] = gidx

    rhoA = np.zeros((dA, dA), dtype=complex)
    for b in range(dB):
        cols = global_idx_mat[:, b]
        valid_mask = cols >= 0
        if not np.any(valid_mask):
            continue
        inds = cols[valid_mask].astype(int)
        sub = rho_mat[np.ix_(inds, inds)]
        rows_a = np.nonzero(valid_mask)[0]
        for k, a in enumerate(rows_a):
            for l, a2 in enumerate(rows_a):
                rhoA[a, a2] += sub[k, l]

    rhoA = 0.5 * (rhoA + rhoA.conj().T)
    if return_qobj:
        return qt.Qobj(rhoA, dims=[[dA],[dA]])
    else:
        return rhoA

# --------------------------
# Analysis function
# --------------------------
def analyze_steady_for_scars(L=8, Omega=1.0, gamma_minus=0.0, gamma_plus=0.0,
                            periodic=False, start_with_one=True,
                            n_scar_candidates=6, energy_eigmethod='dense',
                            plot_results=True):
    """
    Build H (reduced), compute steady state rho_ss, diagonalize H,
    select scar candidates (largest overlap with Neel), compute scar weights.
    """
    # Build basis and operators
    states, index = fibonacci_basis(L)
    d = len(states)
    print(f"L={L}, d={d}")

    H_sparse = build_pxp_hamiltonian_reduced(L, states, index, Omega=Omega, periodic=periodic)
    H_q = qt.Qobj(H_sparse, dims=[[d],[d]])

    c_ops = build_plus_minus_ops_reduced_qobj(L, states, index, gamma_minus=gamma_minus, gamma_plus=gamma_plus, periodic=periodic)

    # compute steady state
    print("Computing steady state...", end=' ', flush=True)
    t0 = time.time()
    try:
        rho_ss = qt.steadystate(H_q, c_ops, method='direct')
    except Exception as e:
        print("\n direct steadystate failed:", e)
        rho_ss = qt.steadystate(H_q, c_ops, method='eigen')
    t1 = time.time()
    print(f"done in {t1-t0:.3f} s")

    # thermodynamic measures
    S_global = float(qt.entropy_vn(rho_ss, base=2))
    purity = float((rho_ss * rho_ss).tr())

    # diagonalize H to get eigenstates and energies
    print("Diagonalizing H...", end=' ', flush=True)
    t0 = time.time()
    if energy_eigmethod == 'dense' or d <= 1200:
        H_dense = H_q.full()
        E_vals, E_vecs = np.linalg.eigh(H_dense)
        # E_vecs columns are eigenvectors in reduced basis
    else:
        # sparse method: get many eigenpairs using scipy.sparse.linalg.eigs
        from scipy.sparse.linalg import eigs
        Hsp = H_sparse.tocsc()
        neigs = min(d-2, max(20, n_scar_candidates*5))
        vals, vecs = eigs(Hsp, k=neigs, which='SR')  # may return complex
        E_vals = np.real(vals)
        E_vecs = vecs
    t1 = time.time()
    print(f"done in {t1-t0:.3f} s; got {len(E_vals)} eigenvalues")

    # construct qutip kets for eigenstates
    eig_kets = [qt.Qobj(E_vecs[:, i], dims=[[d],[1]]) for i in range(E_vecs.shape[1])]

    # construct Neel product state in reduced basis (basis index)
    neel_int = neel_bitstring(L, start_with_one=start_with_one)
    if neel_int not in index:
        raise RuntimeError("Neel state not in reduced basis for this L/pattern.")
    psi_neel = qt.basis(d, index[neel_int])

    # compute overlaps between eigenstates and Neel
    overlaps = np.array([abs((ket.dag() * psi_neel)[0,0]) for ket in eig_kets])
    # sort eigenstates by overlap (descending)
    order = np.argsort(-overlaps)
    scar_indices = order[:n_scar_candidates]

    # compute diagonal of rho_ss in energy basis: p_alpha = <E_alpha | rho_ss | E_alpha>
    # first convert eigenvectors into a matrix U with columns eigenvectors (complex)
    U = np.array([vec.full().reshape(d,) for vec in eig_kets]).T  # shape (d, n_eigs)
    # if we computed only a subset of eigenstates (sparse), warn user
    n_eig_used = U.shape[1]
    if n_eig_used < d:
        print(f"Note: only {n_eig_used}/{d} eigenstates used for projection (sparse diag). p(E) will be partial.")

    # compute p = diag(U^\dagger * rho * U)
    rho_mat = np.array(rho_ss.full(), dtype=complex)
    # project rho into eigenbasis subspace: R = U^\dagger * rho * U  (n_eig_used x n_eig_used)
    Rproj = U.conj().T @ (rho_mat @ U)
    p_alpha = np.real(np.diag(Rproj))

    # compute fidelities F_alpha = <E_alpha| rho |E_alpha> (same as p_alpha)
    F_alpha = p_alpha

    # compute total weight on chosen scar candidates
    w_scar = np.sum(F_alpha[scar_indices])

    # compute coherence inside scar subspace: extract submatrix
    R_scar = Rproj[np.ix_(scar_indices, scar_indices)]
    # norm of off-diagonals (coherence measure)
    offdiag_norm = np.linalg.norm(R_scar - np.diag(np.diag(R_scar)))
    diag_norm = np.linalg.norm(np.diag(R_scar))
    # normalized purity of scar-subspace reduced density (trace may be <1)
    trace_scar = np.real(np.trace(R_scar))
    purity_scar = np.real(np.trace(R_scar @ R_scar)) if trace_scar > 0 else 0.0

    # optionally compute entanglement of left half (using direct partial trace in reduced basis)
    rhoA = reduced_rho_left(L, states, index, rho_ss, L_A=L//2, return_qobj=True)
    S_ent_left = float(qt.entropy_vn(rhoA, base=2))

    # Print summary
    print("\nSUMMARY:")
    print(f"Global entropy S(rho_ss) = {S_global:.6f} bits; purity = {purity:.6e}")
    print(f"Entanglement entropy (left half) S_A = {S_ent_left:.6f} bits")
    print(f"Weight on scar subspace (top {n_scar_candidates} eigenstates by overlap) w_scar = {w_scar:.6e}")
    print(f"trace on scar subspace = {trace_scar:.6e}; purity (scar subspace) = {purity_scar:.6e}")
    print(f"off-diagonal norm in scar subspace = {offdiag_norm:.6e}; diag-norm = {diag_norm:.6e}")

    # Print top candidate info
    print("\nTop candidate scar eigenstates (by overlap with Neel):")
    print(" rank |  index  |  E      |  overlap_with_Neel  |  F_alpha (rho diag)")
    for r, idx_e in enumerate(scar_indices):
        e = E_vals[idx_e]
        ov = overlaps[idx_e]
        f = F_alpha[idx_e]
        print(f"{r+1:4d}   {idx_e:6d}   {e:8.4f}     {ov:12.6e}     {f:12.6e}")

    # Plotting
    if plot_results:
        plt.figure(figsize=(10,4))
        # energy histogram weighted by p_alpha
        eps = 1e-16
        Es = E_vals
        plt.subplot(1,2,1)
        plt.scatter(Es, p_alpha, c='C0', s=40, label="p(E) diag weight")
        plt.xlabel("Energy")
        plt.ylabel("p(E) = <E|rho_ss|E>")
        plt.title("Energy-resolved steady state diagonal")
        # mark scar eigenstates
        plt.scatter(Es[scar_indices], p_alpha[scar_indices], c='C3', s=80, marker='*', label='scar candidates')
        plt.legend()

        # bar of top fidelities (top 12)
        topk = min(12, len(F_alpha))
        order2 = np.argsort(-F_alpha)[:topk]
        plt.subplot(1,2,2)
        plt.bar(np.arange(topk), F_alpha[order2])
        labels = [f"{i}" for i in order2]
        plt.xticks(np.arange(topk), labels, rotation=45)
        plt.ylabel("F_alpha = <E|rho|E>")
        plt.title(f"Top {topk} eigenstate weights in steady state")
        plt.tight_layout()
        plt.show()

    return {
        "L": L,
        "d": d,
        "rho_ss": rho_ss,
        "S_global": S_global,
        "S_ent_left": S_ent_left,
        "purity": purity,
        "E_vals": E_vals,
        "F_alpha": F_alpha,
        "scar_indices": scar_indices,
        "w_scar": w_scar,
        "R_scar": R_scar
    }

# --------------------------
# Example usage as script
# --------------------------
if __name__ == "__main__":
    L = 8
    Omega = 5.0
    gamma_minus = 1.0
    gamma_plus  = 1.0
    periodic = True
    start_with_one = True

    out = analyze_steady_for_scars(L=L, Omega=Omega, gamma_minus=gamma_minus, gamma_plus=gamma_plus,
                                   periodic=periodic, start_with_one=start_with_one,
                                   n_scar_candidates=6, energy_eigmethod='dense', plot_results=True)
