# pxp_fidelity_neel.py  (drop-in replacement / minimal example)
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, basis, qeye, sigmax, sigmap, sigmam, mesolve

def neel_state(L, pattern=0):
    """
    Return Qobj ket for the Neel (Z2) product state.
      pattern=0 -> |1 0 1 0 ...>
      pattern=1 -> |0 1 0 1 ...>
    Note: for PBC choose even L if you want the pattern to satisfy the
    blockade wrapping constraint.
    """
    sites = []
    for i in range(L):
        bit = (1 if ((i % 2 == 0) and pattern == 0) or ((i % 2 == 1) and pattern == 1) else 0)
        # simpler: pattern=0 => 1 on even i; pattern=1 => 1 on odd i
        if pattern == 0:
            bit = 1 if (i % 2 == 0) else 0
        else:
            bit = 1 if (i % 2 == 1) else 0
        sites.append(basis(2, 1) if bit == 1 else basis(2, 0))
    return tensor(sites)

# --- rest of model (same as your previous script) ---
def projector_down():
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

def build_collapse_ops(L, gammap=1.0, gammam=1.0):
    P = projector_down()
    sp = sigmap()
    sm = sigmam()
    c_ops = []
    sqrt_gp = np.sqrt(gammap)
    sqrt_gm = np.sqrt(gammam)
    for i in range(L):
        ops1 = [qeye(2) for _ in range(L)]
        ops2 = [qeye(2) for _ in range(L)]
        ops1[i] = P
        ops1[(i + 1) % L] = sp
        ops1[(i + 2) % L] = P
        ops2[i] = P
        ops2[(i + 1) % L] = sm
        ops2[(i + 2) % L] = P
        c_ops.append(sqrt_gp * manybody(ops1))
        c_ops.append(sqrt_gm * manybody(ops2))
    return c_ops

# --- parameters ---
L = 8               # choose even L for PBC + Neel (recommended)
gammap = 0.0
gammam = 0.0
omega = 10.0   # not used, just for reference
tmax = 15.0
nsteps = 10000
tlist = np.linspace(0.0, tmax, nsteps)

# build model
H = omega * build_pxp_hamiltonian(L)
c_ops = build_collapse_ops(L, gammap=gammap, gammam=gammam)

# choose neel initial state
psi0 = neel_state(L, pattern=0)   # pattern=0 -> |1010...>, pattern=1 -> |0101...>

# projector onto initial pure state -> expectation is fidelity with psi0
P0 = psi0 * psi0.dag()

# time evolve and compute fidelity(t)
result = mesolve(H, psi0, tlist, c_ops, [P0])
fidelity_t = np.real_if_close(result.expect[0])

# plot
plt.figure(figsize=(6,4))
# plt.plot(tlist, 0.05 + (1 - 0.05) * np.exp(-tlist*gammap/omega))
plt.plot(tlist, fidelity_t, lw=2)
plt.xlabel("Time")
plt.ylabel(r"Fidelity $F(t)=\langle\psi_{Neel}|\rho(t)|\psi_{Neel}\rangle$")
plt.title(f"PXP Lindblad: L={L}, gammap={gammap}, gammam={gammam}, Neel pattern=0")
plt.grid(True)
plt.tight_layout()
plt.show()
