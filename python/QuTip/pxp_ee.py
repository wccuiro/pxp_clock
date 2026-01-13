import time
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

L_min = 4
L_max = 6
Omega = 1.0
gamma_p = 0.3
gamma_m = 0.7

proj0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
id2 = qt.qeye(2)
sx = qt.sigmax()
sz = qt.sigmaz()
sp = qt.sigmap()
sm = qt.sigmam()
P = proj0  # use P = |0><0|

def three_site_POP(L, i, Op_middle, proj=P):
    """Create a three-site operator with projectors on sites i and i+2"""
    if i < 0 or i+2 >= L:
        raise IndexError("three_site_POP: invalid i for given L")
    ops = [id2] * L
    ops[i] = proj
    ops[i+1] = Op_middle
    ops[i+2] = proj
    return qt.tensor(ops)

def single_site_op(L, i, Op):
    """Create a single-site operator at position i"""
    ops = [id2] * L
    ops[i] = Op
    return qt.tensor(ops)

Ls = []
Svals = []

for L in range(L_min, L_max+1):
    dim = 2**L
    print(f"L = {L} (dim = {dim}) ... ", end="", flush=True)
    
    # CORRECT: initialize H with matching multipartite dims
    H = 0 * qt.tensor([id2] * L)
    
    for i in range(0, L-2):
        H += Omega * three_site_POP(L, i, sx)
    
    c_ops = []
    for i in range(0, L-2):
        c_ops.append(np.sqrt(gamma_p) * three_site_POP(L, i, sp))
        c_ops.append(np.sqrt(gamma_m) * three_site_POP(L, i, sm))
    
    # Add small uniform dephasing to break degeneracies and ensure unique steady state
    dephasing_rate = 0.01
    for i in range(L):
        c_ops.append(np.sqrt(dephasing_rate) * single_site_op(L, i, sz))
    
    # Try multiple methods to find steady state
    methods = ['eigen', 'direct', 'iterative']
    rho_ss = None
    
    t0 = time.time()
    
    for method in methods:
        try:
            rho_ss = qt.steadystate(H, c_ops, method=method, sparse=True, use_precond=True)
            break
        except Exception as e:
            if method == methods[-1]:  # Last method failed
                print(f"All methods failed. Error: {str(e)[:50]}...")
                break
            continue
    
    if rho_ss is None:
        print("FAILED - skipping")
        continue
    
    t_elapsed = time.time() - t0
    
    # Ensure proper normalization
    trace_val = rho_ss.tr()
    if abs(trace_val - 1.0) > 1e-10:
        rho_ss = rho_ss / trace_val
    
    # Compute entropy of the full steady state
    S = qt.entropy_vn(rho_ss, base=2)
    
    Ls.append(L)
    Svals.append(float(S))
    
    print(f"S = {S:.6f} bits (t = {t_elapsed:.2f}s)")

if len(Ls) == 0:
    print("\nNo successful calculations. The model may have fundamental issues.")
    print("Suggestions:")
    print("1. Increase dephasing_rate (currently 0.01)")
    print("2. Check if gamma_p and gamma_m values make physical sense")
    print("3. Verify the three-site PXP model construction")
else:
    plt.figure(figsize=(6,4))
    plt.plot(Ls, Svals, marker='o', linestyle='-')
    plt.xlabel('Chain length L')
    plt.ylabel('Von Neumann entropy (bits) of full steady state')
    plt.title(r'Steady-state $S_{\mathrm{vN}}(\mathrm{full\ state})$ vs $L$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults:")
    for L, S in zip(Ls, Svals):
        print(f"L={L}: S={S:.6f} bits")