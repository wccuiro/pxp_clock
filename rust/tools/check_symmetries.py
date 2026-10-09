"""
Check which symmetry reductions of the Rust code hold for a model, without using the Rust code.

    python3 tools/check_symmetries.py L gp gm omega [alpha] [plus=0|1]   constrained model, partial projection with alpha,
                                                                         or staggered jumps (sigma^+ on the sites j = plus mod 2)
    python3 tools/check_symmetries.py sizes L [full|staggered]           block sizes only (no matrices), up to L

With parameters it builds the full Lindbladian (no symmetry) and prints
  - the residuals of the symmetries used by src/common.rs:
      translation   max|L B - B (B^T L B)| for the sectors Q = 0 and Q = L/2
                    (staggered: the sector Q = 0 of the translation by two sites; the residual of the
                    translation by one site, which is lost, is printed for information)
      R             max|L R - R L|          reflection j -> -j mod L
      S             max|L S - S L|          rho -> C rho^T C, C = prod_j Z_j
      Hermiticity   max|L K - K conj(L)|    rho -> rho^dag
  - the sizes of the four blocks (sigma, tau) of each sector, from the projectors and from
    fixed-point counting, and the weight of the Neel state and of the trace in each block.

Exit status 1 if a residual is above 1e-12 or the two block sizes disagree.
"""
import sys

import numpy as np
import scipy.sparse as sp

import pxp_reference as ref

TOL = 1e-12
BLOCKS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

def label(block):
  return "(" + ",".join("+" if x > 0 else "-" for x in block) + ")"

def print_sizes(L_max, constrained, step=1):
  print(("constrained basis" if constrained else "full 2^L basis") + (", translation by two sites (staggered)" if step == 2 else ""))
  print(f"{'L':>3} {'Q':>3} {'sector':>10} " + " ".join(f"{label(b):>10}" for b in BLOCKS))
  for L in range(4, L_max + 1, 2):
    sizes = ref.block_sizes(L, constrained, step)
    for Q in ref.sectors(L, step):
      print(f"{L:>3} {Q:>3} {sum(sizes[Q].values()):>10} " + " ".join(f"{sizes[Q][b]:>10}" for b in BLOCKS))

def main():
  if len(sys.argv) in (3, 4) and sys.argv[1] == "sizes":
    staggered = len(sys.argv) == 4 and sys.argv[3] == "staggered"
    print_sizes(int(sys.argv[2]), constrained=(len(sys.argv) == 3 or staggered), step=(2 if staggered else 1))
    return
  plus_site, step = ref.plus_site_argument(sys.argv)
  if len(sys.argv) not in (5, 6):
    sys.exit("usage: check_symmetries.py L gp gm omega [alpha] [plus=0|1]   |   check_symmetries.py sizes L [full|staggered]")
  L = int(sys.argv[1])
  gamma_plus, gamma_minus, omega = (float(x) for x in sys.argv[2:5])
  alpha = float(sys.argv[5]) if len(sys.argv) == 6 else None

  states, index, Lind = ref.build_model(L, gamma_plus, gamma_minus, omega, alpha, plus_site)
  N = len(states)
  f = ref.trace_vectors(L, states, index)
  R, S, K = ref.symmetry_superoperators(L, states, index)
  model = "constrained" if alpha is None else f"partial projection alpha={alpha}"
  if plus_site is not None:
    model += f", staggered (sigma+ on sites j = {plus_site} mod 2)"
  print(f"L = {L}, gp={gamma_plus} gm={gamma_minus} omega={omega}, {model}: {N} configurations, operator space {N*N}")

  maxabs = lambda M: abs(M).max() if M.nnz else 0.0
  residuals = {
    "R": maxabs(Lind @ R - R @ Lind),
    "S": maxabs(Lind @ S - S @ Lind),
    "Hermiticity": maxabs(Lind @ K - K @ Lind.conj()),
  }
  counted = ref.block_sizes(L, constrained=(alpha is None), step=step)
  failed = False
  if step == 2:
    T1 = np.array([index[ref.translation(L, s)] for s in states])
    a, b = np.divmod(np.arange(N * N), N)
    T1 = sp.csr_matrix((np.ones(N * N), (T1[a] * N + T1[b], a * N + b)), shape=(N * N, N * N))
    print(f"  translation by one site: residual {maxabs(Lind @ T1 - T1 @ Lind):.1e} (not a symmetry of the staggered model)")

  print(f"\n{'Q':>3} {'block':>6} {'size':>8} {'counted':>8} {'Neel weight':>12} {'trace weight':>13}")
  for Q in ref.sectors(L, step):
    B = ref.sector_basis(L, states, index, Q, step)
    _, residuals[f"translation Q={Q}"] = ref.reduce_to_sector(Lind, B)
    blocks, residual = ref.symmetry_blocks(B, R, S)
    residuals[f"R, S keep sector Q={Q}"] = residual
    rho0, tr = B.T @ f['rho0'], B.T @ f['tr']
    for block in BLOCKS:
      P = blocks[block]
      size = int(round(P.diagonal().sum()))
      failed = failed or size != counted[Q][block]
      print(f"{Q:>3} {label(block):>6} {size:>8} {counted[Q][block]:>8} {np.sum((P @ rho0)**2):>12.6f} {np.sum((P @ tr)**2) / N:>13.6f}")

  print()
  for name, value in residuals.items():
    ok = value <= TOL
    failed = failed or not ok
    print(f"  {name:26s} residual {value:.1e}   {'ok' if ok else 'FAIL'}")
  sys.exit(1 if failed else 0)

if __name__ == "__main__":
  main()
