"""
Minimal PXP steady-state entropy vs chain length (QuTiP).
Requirements:
    pip install qutip numpy matplotlib
Warning:
    Hilbert dimension = 2**N grows quickly. Use small N (<= 8 typically).
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    qeye, sigmax, sigmaz, sigmap, sigmam, tensor, basis,
    steadystate, entropy_vn
)

# --- helpers ---
def P_ground():
    """Projector onto |0> for a single qubit: P = |0><0|"""
    return basis(2, 0) * basis(2, 0).dag()

def three_site_POP(L, i, Op_middle, proj=None):
    """Create a three-site operator with projectors on sites i and i+2"""
    if proj is None:
        proj = P_ground()
    
    if i < 0 or i+2 >= L:
        raise IndexError("three_site_POP: invalid i for given L")
    
    ops = [qeye(2)] * L
    ops[i] = proj
    ops[i+1] = Op_middle
    ops[i+2] = proj
    return tensor(ops)

def single_site_op(L, i, Op):
    """Create a single-site operator at position i"""
    ops = [qeye(2)] * L
    ops[i] = Op
    return tensor(ops)

# --- PXP Hamiltonian ---
def pxp_hamiltonian(N, Omega):
    """H = Omega * sum_i P_i X_{i+1} P_{i+2} with open boundaries."""
    # CORRECT: initialize H with matching multipartite dims
    H = 0 * tensor([qeye(2)] * N)
    
    for i in range(0, N-2):
        H += Omega * three_site_POP(N, i, sigmax())
    
    return H

# --- jump operators ---
def jump_L(N, j):
    """
    L_j = sigma_j^+ sigma_j^- - sigma_{j+1}^+ sigma_j^- + sigma_j^+ sigma_{j+1}^- - sigma_{j+1}^+ sigma_{j+1}^-
    """
    if not (0 <= j < N - 2):
        raise ValueError(f"j must be in [0, {N-3}]")
    
    sp = sigmap()
    sm = sigmam()
    
    # sigma_j^+ sigma_j^-  -> (sp*sm) at site j
    t1 = single_site_op(N, j, sp * sm)
    # sigma_{j+1}^+ sigma_j^-
    ops = [qeye(2)] * N
    ops[j+1] = sp
    ops[j] = sm
    t2 = tensor(ops)
    # sigma_j^+ sigma_{j+1}^-
    ops = [qeye(2)] * N
    ops[j] = sp
    ops[j+1] = sm
    t3 = tensor(ops)
    # sigma_{j+1}^+ sigma_{j+1}^- -> (sp*sm) at site j+1
    t4 = single_site_op(N, j+1, sp * sm)
    
    return t1 - t2 + t3 - t4

def build_collapse_ops(N, gamma):
    """Return list of collapse operators for PXP model with dephasing."""
    c_ops = []
    
    if gamma > 0:
        # Original jump operators (may cause singularity)
        for j in range(N - 2):
            c_ops.append(np.sqrt(gamma) * jump_L(N, j))
    
    # Add small uniform dephasing to break degeneracies and ensure unique steady state
    dephasing_rate = 0.01
    for i in range(N):
        c_ops.append(np.sqrt(dephasing_rate) * single_site_op(N, i, sigmaz()))
    
    return c_ops

# --- compute steady state and entropy ---
def steadystate_entropy(N, Omega, gamma):
    """
    Compute steady state rho_ss and von Neumann entropy (base 2).
    """
    H = pxp_hamiltonian(N, Omega)
    c_ops = build_collapse_ops(N, gamma)
    
    if len(c_ops) == 0:
        # Closed system -> use ground state projector as "steady" pure state
        evals, evecs = H.eigenstates()
        psi0 = evecs[0]
        rho_ss = psi0 * psi0.dag()
    else:
        # Try multiple methods to find steady state
        methods = ['eigen', 'direct', 'iterative']
        rho_ss = None
        
        for method in methods:
            try:
                rho_ss = steadystate(H, c_ops, method=method, sparse=True, use_precond=True)
                break
            except Exception as e:
                if method == methods[-1]:  # Last method failed
                    raise RuntimeError(f"All methods failed. Last error: {str(e)}")
                continue
        
        # Ensure proper normalization
        trace_val = rho_ss.tr()
        if abs(trace_val - 1.0) > 1e-10:
            rho_ss = rho_ss / trace_val
    
    S = entropy_vn(rho_ss, base=2)  # von Neumann entropy in bits
    return rho_ss, S

# --- sweep over chain lengths and plot ---
def entropy_vs_lengths(L_list, Omega=1.0, gamma=0.1):
    """
    Compute steady-state entropy for each N in L_list and plot.
    Returns list of entropies.
    """
    entropies = []
    successful_Ls = []
    
    for N in L_list:
        print(f"Computing N={N} (Hilbert dim {2**N}) ... ", end="", flush=True)
        try:
            rho_ss, S = steadystate_entropy(N, Omega, gamma)
            print(f"S = {S:.6f} bits")
            entropies.append(S)
            successful_Ls.append(N)
        except Exception as e:
            print(f"FAILED - {str(e)[:50]}...")
            continue
    
    if len(successful_Ls) == 0:
        print("\nNo successful calculations.")
        return []
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(successful_Ls, entropies, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Chain length N")
    plt.ylabel("Steady-state von Neumann entropy (bits)")
    plt.title(f"PXP steady-state entropy (Ω={Omega}, γ={gamma})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return entropies

# --- example usage ---
if __name__ == "__main__":
    # Choose small Ns to keep runtime manageable.
    L_list = [4, 5, 6, 7, 8]   # Start from 4 since PXP needs at least 3 sites for non-trivial terms
    Omega = 1.0
    gamma = 0.1   # use gamma>0 for nontrivial Lindblad steady state
    
    print("Computing PXP steady-state entropy for different chain lengths...")
    ent = entropy_vs_lengths(L_list, Omega=Omega, gamma=gamma)
    
    if ent:
        print("\nFinal results:")
        successful_Ls = [L for L in L_list if len(ent) > L_list.index(L)]
        for i, L in enumerate(L_list[:len(ent)]):
            print(f"N={L}: entropy = {ent[i]:.6f} bits")