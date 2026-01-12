# pxp_lindblad_spectrum.py - FIXED VERSION
import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    tensor, qeye, sigmax, sigmap, sigmam, basis, liouvillian, steadystate
)
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals

# ---------- model builders (same as your previous scripts) ----------
def projector_down():
    """P = |0><0|"""
    return basis(2, 0) * basis(2, 0).dag()

def manybody(op_list):
    return tensor(op_list)

def build_pxp_hamiltonian(L):
    P = projector_down()
    X = sigmax()
    H = 0 * manybody([qeye(2) for _ in range(L)])
    for i in range(L):
        ops = [qeye(2) for _ in range(L)]
        ops[i] = P
        ops[(i + 1) % L] = X
        ops[(i + 2) % L] = P
        H += manybody(ops)
    return H

def build_collapse_ops(L, gamma=1.0):
    P = projector_down()
    sp = sigmap()
    sm = sigmam()
    c_ops = []
    sqrt_g = np.sqrt(gamma)
    for i in range(L):
        ops1 = [qeye(2) for _ in range(L)]
        ops2 = [qeye(2) for _ in range(L)]
        ops1[i] = P
        ops1[(i + 1) % L] = sp
        ops1[(i + 2) % L] = P
        ops2[i] = P
        ops2[(i + 1) % L] = sm
        ops2[(i + 2) % L] = P
        c_ops.append(sqrt_g * manybody(ops1))
        c_ops.append(sqrt_g * manybody(ops2))
    return c_ops

# ---------- FIXED spectrum routine ----------
def compute_lindblad_spectrum(H, c_ops, n_eigs=10, dense_threshold=1600):
    """
    Build Liouvillian and compute spectrum.
    - If superoperator dimension D^2 <= dense_threshold: compute full (dense) spectrum.
    - Otherwise compute 'n_eigs' eigenvalues with largest real part (slowest decays) using sparse eigs.
    Returns: (eigenvalues, L_super (Qobj))
    """
    # Ensure c_ops is a list
    if not isinstance(c_ops, list):
        c_ops = [c_ops] if c_ops is not None else []
    
    # build Qutip Liouvillian Qobj (superoperator)
    Lq = liouvillian(H, c_ops)   # Qobj in superoperator form
    
    # FIXED: Use shape[0] since Liouvillian is square matrix
    D2 = Lq.shape[0]  # shape is (D^2, D^2) 
    D = int(np.sqrt(D2))  # Hilbert space dimension
    
    print(f"Hilbert dim D = {D}, superoperator dim = {D2} ({D2}x{D2})")

    if D2 <= dense_threshold:
        # compute full dense spectrum (may be costly but returns all eigenvalues)
        print("Converting Liouvillian to dense array and computing full spectrum (dense).")
        Lmat = Lq.full()        # dense numpy ndarray
        w = eigvals(Lmat)       # full dense eigenvalues (complex)
        return w, Lq
    else:
        # sparse eigenproblem: compute a few eigenvalues with largest real part
        print(f"Using sparse eigs to compute {n_eigs} eigenvalues with largest real part.")
        Lsp = Lq.data          # scipy sparse matrix (CSR/CSC)
        
        # FIXED: Make sure we don't ask for more eigenvalues than possible
        max_possible = D2 - 2  # eigs requires k < n-1 for some methods
        n_eigs_actual = min(n_eigs, max_possible)
        
        if n_eigs_actual <= 0:
            raise ValueError(f"Cannot compute eigenvalues: system too small (D^2={D2})")
            
        print(f"Computing {n_eigs_actual} eigenvalues (requested {n_eigs})")
        
        # which='LR' -> eigenvalues with largest real part (closest to 0 for Lindbladian steady-state)
        try:
            w, v = eigs(Lsp, k=n_eigs_actual, which='LR')
            return w, Lq
        except Exception as e:
            print(f"Sparse eigs failed: {e}")
            print("Falling back to dense computation...")
            Lmat = Lq.full()
            w = eigvals(Lmat)
            return w, Lq

# ---------- plotting helper ----------
def plot_spectrum(eigvals_complex, title="Liouvillian spectrum", fname=None):
    re = np.where(np.abs(eigvals_complex.real) < 10e-10, 0, eigvals_complex.real)
    # re = eigvals_complex.real
    im = np.where(np.abs(eigvals_complex.imag) < 10e-10, 0, eigvals_complex.imag)
    plt.figure(figsize=(6,5))
    plt.scatter(re, im, s=40, edgecolor='k')
    plt.axvline(0, color='gray', lw=0.6)
    plt.axhline(0, color='gray', lw=0.6)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title(title)
    plt.grid(True, lw=0.2)
    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
    plt.show()

# ---------- example usage ----------
if __name__ == "__main__":
    # parameters
    L = 4           # Start with L=3 to test
    gamma = 0.1
    omega = 0.0
    
    print(f"Testing with L={L}")
    print(f"Expected Hilbert dimension: 2^{L} = {2**L}")
    print(f"Expected superoperator dimension: {2**L}^2 = {(2**L)**2}")

    # build model
    H = omega * build_pxp_hamiltonian(L)
    c_ops = build_collapse_ops(L, gamma=gamma)
    
    print(f"Built Hamiltonian: {H.shape}")
    print(f"Number of collapse operators: {len(c_ops)}")

    # compute spectrum: for typical laptop set dense_threshold to e.g. 2000 or less
    # superoperator dimension is D^2 = (2^L)^2 = 4^L
    dense_threshold = 100   # Set low for testing - will use dense for L=3
    n_eigs = 6              # Reduced for small system

    try:
        eigvals_L, Lq = compute_lindblad_spectrum(H, c_ops, n_eigs=n_eigs, dense_threshold=dense_threshold)
        
        print(f"\nSuccessfully computed {len(eigvals_L)} eigenvalues")
        print("Computed eigenvalues (sample):")
        
        # sort by real part descending (largest real part first, near 0 are slow modes / steady)
        idxs = np.argsort(-eigvals_L.real)
        for ii in idxs[:min(10, len(eigvals_L))]:
            print(f"  {ii:3d}:  {eigvals_L[ii].real:+.8f} {eigvals_L[ii].imag:+.8f}j")

        plot_spectrum(eigvals_L, title=f"Liouvillian spectrum (L={L}, gamma={gamma}, omega={omega})", fname=f"lindblad_spectrum_L{L}.png")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # # Test with larger L
    # if L == 3:  # Only test L=4 if L=3 worked
    #     print("\n" + "="*50)
    #     print("Testing L=4...")
    #     try:
    #         L = 4
    #         H = build_pxp_hamiltonian(L)
    #         c_ops = build_collapse_ops(L, gamma=gamma)
    #         eigvals_L, Lq = compute_lindblad_spectrum(H, c_ops, n_eigs=12, dense_threshold=300)
    #         print(f"L=4 also works! Got {len(eigvals_L)} eigenvalues")
    #     except Exception as e:
    #         print(f"L=4 failed: {e}")