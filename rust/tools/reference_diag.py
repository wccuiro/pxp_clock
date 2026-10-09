"""
Reference diagonalization of the PXP Lindbladian (independent of the Rust code).

    python3 tools/reference_diag.py L gp gm omega out.json [alpha] [plus=0|1]

With alpha it is the partial projection model of src/bin/lindblad_alpha (full 2^L basis,
see pxp_reference.py); without it the constrained model. With plus=0 or plus=1 it is the staggered
model of src/bin/lindblad_staggered (sigma^+ only on the sites j = plus mod 2, sigma^- on the others).

For the sectors Q = 0 and Q = L/2 (staggered: the sector Q = 0 of the translation by two sites) it writes, per eigenmode k of the dense sector matrix:
  lambda_k
  c_k   Neel expansion coefficient on the unit-norm right eigenvector r_k  (rho0 = sum_k c_k r_k)
  o_k   Tr(r_k^dag rho0)
  w_k   c_k Tr(n r_k)            ->  <n>(t) = sum_k w_k e^{lambda_k t}
  oee_k operator entanglement entropy of r_k^dag r_k across the cut [0, L/2) | [L/2, L)
and the steady state (<n>, <nn>, spectrum of rho_ss with the occupation of each eigenvector),
if it is unique.

c_k and o_k carry the arbitrary phase of r_k; |c|, |o|, c conj(o) and w do not.
Everything is dense and exact; practical up to L = 10 (constrained) and L = 6 (alpha;
L = 8 was measured at 4.8 GB and 11-17 min).
"""
import sys
import json
import time

import numpy as np

import pxp_reference as ref

TOL_ZERO_EIG = 1e-8

#############################################################################
###################### OPERATOR ENTANGLEMENT ################################
#############################################################################

def fibonacci_sub_basis(L):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      states.append(i)
  return states

def schmidt_indices(L, states, constrained=True):
  """Row/column of the A|B Schmidt matrix for every element S[b1, b2] (A = lower L/2 bits)."""
  L_A = L // 2
  sub_basis = fibonacci_sub_basis if constrained else ref.full_basis
  index_A = {s: i for i, s in enumerate(sub_basis(L_A))}
  index_B = {s: i for i, s in enumerate(sub_basis(L - L_A))}
  low = np.array([index_A[s & ((1 << L_A) - 1)] for s in states])
  up = np.array([index_B[s >> L_A] for s in states])
  d_A, d_B = len(index_A), len(index_B)
  rows = low[:, None] * d_A + low[None, :]
  cols = up[:, None] * d_B + up[None, :]
  return rows, cols, d_A, d_B

def operator_entanglement(r, rows, cols, d_A, d_B):
  S = r.conj().T @ r
  M = np.zeros((d_A * d_A, d_B * d_B), dtype=complex)
  M[rows, cols] = S
  sv = np.linalg.svd(M, compute_uv=False)
  total = np.sum(sv**2)
  if total < 1e-12:
    return 0.0
  p = sv**2 / total
  p = p[p > 1e-12]
  return float(-np.sum(p * np.log(p)))

#############################################################################
###################### MAIN PROGRAM #########################################
#############################################################################

def main():
  plus_site, step = ref.plus_site_argument(sys.argv)
  if len(sys.argv) not in (6, 7):
    sys.exit("usage: reference_diag.py L gp gm omega out.json [alpha] [plus=0|1]")
  L = int(sys.argv[1])
  gamma_plus, gamma_minus, omega = (float(x) for x in sys.argv[2:5])
  out_file = sys.argv[5]
  alpha = float(sys.argv[6]) if len(sys.argv) == 7 else None

  t0 = time.time()
  states, index, Lind = ref.build_model(L, gamma_plus, gamma_minus, omega, alpha, plus_site)
  N = len(states)
  f = ref.trace_vectors(L, states, index)
  rows, cols, d_A, d_B = schmidt_indices(L, states, constrained=(alpha is None))
  model = "constrained" if alpha is None else f"alpha = {alpha}"
  if plus_site is not None:
    model += f", staggered (sigma+ on sites j = {plus_site} mod 2)"
  print(f"L = {L}, {model}: {N} configurations, operator space {N*N}  (built in {time.time()-t0:.1f}s)")

  result = {"L": L, "gp": gamma_plus, "gm": gamma_minus, "omega": omega, "alpha": alpha, "plus_site": plus_site,
            "sectors": {}, "steady": None}

  for Q in ref.sectors(L, step):
    t0 = time.time()
    B = ref.sector_basis(L, states, index, Q, step)
    L_Q, residual = ref.reduce_to_sector(Lind, B)
    f_Q = {name: B.T @ vec for name, vec in f.items()}

    eigvals, V = np.linalg.eig(L_Q)          # columns of V have unit norm
    order = np.lexsort((-eigvals.imag, -eigvals.real))
    eigvals, V = eigvals[order], V[:, order]

    c = np.linalg.solve(V, f_Q['rho0'])
    o = V.conj().T @ f_Q['rho0']
    tr_r = f_Q['tr'] @ V
    w = c * (f_Q['n'] @ V)

    oee = np.zeros(len(eigvals))
    for k in range(len(eigvals)):
      r = (B @ V[:, k]).reshape((N, N), order='C')
      oee[k] = operator_entanglement(r, rows, cols, d_A, d_B)

    if Q == 0:
      k0 = np.argmin(np.abs(eigvals))
      zero_modes = int(np.sum(np.abs(eigvals) < TOL_ZERO_EIG))
      if zero_modes > 1:
        print(f"  Q=0: {zero_modes} zero modes, the steady state is not unique and is not written")
      elif np.abs(eigvals[k0]) < TOL_ZERO_EIG and np.abs(tr_r[k0]) > 1e-10:
        rho_ss = (B @ V[:, k0]).reshape((N, N), order='C') / tr_r[k0]
        rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
        p, U = np.linalg.eigh(rho_ss)
        occ = np.real(np.einsum('ak,a,ak->k', U.conj(), np.diag(f['n'].reshape(N, N)), U))
        desc = np.argsort(-p)
        result["steady"] = {
          "n": float(np.real(f_Q['n'] @ V[:, k0] / tr_r[k0])),
          "nn": float(np.real(f_Q['nn'] @ V[:, k0] / tr_r[k0])),
          "spectrum_p": (p[desc] / np.sum(p)).tolist(),
          "spectrum_n": occ[desc].tolist(),
        }

    result["sectors"][str(Q)] = {
      "dim": len(eigvals),
      "invariance_residual": float(residual),
      "lambda_re": eigvals.real.tolist(), "lambda_im": eigvals.imag.tolist(),
      "c_re": c.real.tolist(), "c_im": c.imag.tolist(),
      "o_re": o.real.tolist(), "o_im": o.imag.tolist(),
      "w_re": w.real.tolist(), "w_im": w.imag.tolist(),
      "oee": oee.tolist(),
    }
    print(f"  Q={Q}: dim {len(eigvals)}  sector residual {residual:.1e}  "
          f"sum c Tr(r) = {np.sum(c * tr_r).real:.12f}  sum w = {np.sum(w).real:.12f}  ({time.time()-t0:.1f}s)")

  if result["steady"] is not None:
    print(f"  steady state: <n> = {result['steady']['n']:.10f}  <nn> = {result['steady']['nn']:.10f}")

  with open(out_file, 'w') as file:
    json.dump(result, file)
  print(f"written {out_file}")

if __name__ == "__main__":
  main()
