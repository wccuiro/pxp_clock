"""
Independent Python reference for the PXP Lindbladian, shared by the tools in this folder.

    L[rho] = -i Omega [H, rho] + sum_{g in {+,-}} gamma_g sum_j ( A_j rho A_j^dag - 1/2 {A_j^dag A_j, rho} )

The basis, Hamiltonian, dissipator and Lindbladian are the builders of
python/pxp_lindblad_full.py: full periodic constrained basis, no symmetry reduction,
row-major vectorization vec(rho)[a*N + b] = rho[a, b]. The only change is scipy.sparse
instead of dense np.kron, so that L = 10 fits in memory.

The translation sectors Q = 0 and Q = L/2 (the two that contain the Neel state) are built
from the orbits of pairs (a, b) under (a, b) -> (T a, T b). This is a different construction
from the |n,k><m,k-Q| basis of the Rust code, and the invariance of the sector under L is
checked numerically (reduce_to_sector returns the residual).

Partial projection (alpha) model, as in src/bin/lindblad_alpha: the Hamiltonian keeps the strict
blockade, H = sum_j P_{j-1} X_j P_{j+1}, but the jumps use the soft projector

    A^+-_j = P^alpha_{j-1} sigma^+-_j P^alpha_{j+1},   P^alpha = [(1+alpha)|0><0| + (1-alpha)|1><1|] / (1+|alpha|)

so adjacent excitations are created for alpha < 1 and the full 2^L basis is needed.
alpha = 1 gives P^alpha = |0><0| (the constrained model, embedded in the full basis).
All builders take alpha = None for the constrained model.

Staggered model, as in src/bin/lindblad_staggered (constrained basis): sigma^+ jumps only on the
sites j = plus_site (mod 2), sigma^- jumps only on the other sublattice (plus_site = None: both on
every site). The Neel state occupies the sublattice 0. Only the translation by two sites is left,
so the sector builders take step = 2 and there is one sector with the Neel state, Q = 0.
The tools read it from a last argument plus=0 or plus=1 (see plus_site_argument).
"""
import numpy as np
import scipy.sparse as sp

#############################################################################
###################### GENERATION OF THE BASIS ##############################
#############################################################################

def fibonacci_basis(L):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      if 2**0 & i and 2**(L-1) & i:
        continue
      else:
        states.append(i)
  return states

def full_basis(L):
  return list(range(1 << L))

def generation_basis(L, constrained=True):
  rep_states = fibonacci_basis(L) if constrained else full_basis(L)
  rep_index = {s: i for i,  s in enumerate(rep_states)}
  return rep_states, rep_index

#############################################################################
###################### GENERATION OF THE HAMILTONIAN ########################
#############################################################################

def Hamiltonian(L, states, index, omega):

  H = sp.lil_matrix((len(states), len(states)))

  for state in states:
    for i in range(L):
      if (state >> ((i-1)%L)) & 1 == 0 and (state >> ((i+1)%L)) & 1 == 0:
          state_p = state ^ 2**i
          H [ index[state], index[state_p] ] += omega

  return H.tocsr()

#############################################################################
###################### GENERATION OF THE DISSIPATOR #########################
#############################################################################

def soft_projector(L, state, site, alpha):
  """Diagonal element of P^alpha at `site`."""
  c_norm = 1.0 / (1.0 + abs(alpha))
  if state & (1 << (site % L)):
    return c_norm * (1.0 - alpha)
  return c_norm * (1.0 + alpha)

def dissipation(L, states, index, gamma_plus, gamma_minus, alpha=None, plus_site=None):
  N = len(states)
  I = sp.identity(N, format='csr')
  D_minus = sp.csr_matrix((N**2, N**2))
  D_plus = sp.csr_matrix((N**2, N**2))

  for i in range(L):

    L_minus_i = sp.lil_matrix((N, N))
    L_plus_i = sp.lil_matrix((N, N))

    for state in states:
      if alpha is None:
        factor = 1.0 if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0 else 0.0
      else:
        factor = soft_projector(L, state, i-1, alpha) * soft_projector(L, state, i+1, alpha)
      if factor != 0.0:
        state_p = state ^ (1 << i)
        if state & 1<<i:
          if plus_site is None or i % 2 != plus_site:
            L_minus_i [ index[state_p], index[state]] += factor
        else:
          if plus_site is None or i % 2 == plus_site:
            L_plus_i [ index[state_p], index[state]] += factor

    L_minus_i = L_minus_i.tocsr()
    L_plus_i = L_plus_i.tocsr()

    D_minus += sp.kron(L_minus_i, L_minus_i.conj()) - 0.5 * sp.kron(L_minus_i.conj().T @ L_minus_i, I) - 0.5 * sp.kron(I, (L_minus_i.conj().T @ L_minus_i).T)
    D_plus += sp.kron(L_plus_i, L_plus_i.conj()) - 0.5 * sp.kron(L_plus_i.conj().T @ L_plus_i, I) - 0.5 * sp.kron(I, (L_plus_i.conj().T @ L_plus_i).T)

  D = gamma_minus * D_minus + gamma_plus * D_plus

  return D

#############################################################################
###################### GENERATION OF THE LINDBLADIAN ########################
#############################################################################

def lindblad_evolution(H, D):
  I = sp.identity(H.shape[0], format='csr')
  L = -1j * (sp.kron(H, I) - sp.kron(I, H.T)) + D
  return L.tocsr()

#############################################################################
###################### TRANSLATION SECTORS ##################################
#############################################################################

def translation(L, state, step=1):
  return ((state << step) | (state >> (L - step))) & ((1 << L) - 1)

def sector_basis(L, states, index, Q, step=1):
  """
  Orthonormal basis of the operators with U rho U^dag = e^{-i 2 pi Q / nt} rho, U = T^step the
  translation by `step` sites and nt = L / step, for Q = 0 or nt/2:

      (1/sqrt(p)) sum_{d<p} s^d |U^d a><U^d b|,   s = +1 (Q = 0), -1 (Q = nt/2),

  one vector per orbit of pairs (a, b) with period p (only even p for Q = nt/2).
  Returns B (N^2 x dim, real, B^T B = 1).
  """
  nt = L // step
  if L % 2 or L % step or not (Q == 0 or 2 * Q == nt):
    raise ValueError("only Q = 0 and Q = nt/2 (even L, nt = L/step) are implemented")

  N = len(states)
  T = [index[translation(L, s, step)] for s in states]
  seen = np.zeros(N * N, dtype=bool)
  rows, cols, vals = [], [], []
  dim = 0

  for a in range(N):
    for b in range(N):
      if seen[a * N + b]:
        continue
      orbit = []
      x, y = a, b
      while not seen[x * N + y]:
        seen[x * N + y] = True
        orbit.append(x * N + y)
        x, y = T[x], T[y]
      p = len(orbit)
      if (Q * p) % nt != 0:
        continue
      for d, idx in enumerate(orbit):
        rows.append(idx)
        cols.append(dim)
        vals.append((-1 if (Q != 0 and d % 2) else 1) / np.sqrt(p))
      dim += 1

  return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, dim))

def reduce_to_sector(Lind, B):
  """Dense sector matrix B^T L B and the invariance residual max|L B - B (B^T L B)|."""
  X = (Lind @ B).tocsr()
  L_Q = B.T @ X
  residual = X - B @ L_Q
  return L_Q.toarray(), (abs(residual).max() if residual.nnz else 0.0)

#############################################################################
###################### NEEL STATE AND OBSERVABLES ###########################
#############################################################################

def neel_state(L):
  return sum(1 << i for i in range(0, L, 2))

def trace_vectors(L, states, index):
  """
  Row vectors f with Tr(O rho) = f . vec(rho) for
    'tr'   : O = 1
    'n'    : O = (1/L) sum_j n_j
    'nn'   : O = (1/L) sum_j n_{j-1} n_{j+1}
    'rho0' : O = |Neel><Neel|   (this is also vec(rho0))
  """
  N = len(states)
  f = {name: np.zeros(N * N) for name in ('tr', 'n', 'nn', 'rho0')}

  for state in states:
    d = index[state] * (N + 1)
    f['tr'][d] = 1
    for i in range(L):
      if state & (1 << i):
        f['n'][d] += 1 / L
      if state & (1 << ((i-1)%L)) and state & (1 << ((i+1)%L)):
        f['nn'][d] += 1 / L

  f['rho0'][index[neel_state(L)] * (N + 1)] = 1

  return f

#############################################################################
###################### FULL MODEL ###########################################
#############################################################################

def build_model(L, gamma_plus, gamma_minus, omega, alpha=None, plus_site=None):
  states, index = generation_basis(L, constrained=(alpha is None))
  H = Hamiltonian(L, states, index, omega)
  D = dissipation(L, states, index, gamma_plus, gamma_minus, alpha, plus_site)
  Lind = lindblad_evolution(H, D)
  return states, index, Lind

def sectors(L, step=1):
  """Sectors that contain the Neel state: Q = 0 and L/2 of T, or Q = 0 of the translation by two sites."""
  return [0, L // 2] if step == 1 else [0]

def plus_site_argument(argv):
  """Removes a last argument plus=0 / plus=1 from argv. Returns (plus_site or None, translation step)."""
  if len(argv) > 1 and argv[-1].startswith("plus="):
    plus_site = int(argv.pop()[len("plus="):])
    if plus_site not in (0, 1):
      raise SystemExit("plus= must be 0 or 1")
    return plus_site, 2
  return None, 1

#############################################################################
###################### REFLECTION, S AND HERMITICITY ########################
#############################################################################

def reflection(L, state):
  """j -> -j mod L"""
  out = 0
  for j in range(L):
    if state & (1 << j):
      out |= 1 << ((L - j) % L)
  return out

def symmetry_superoperators(L, states, index):
  """
  Sparse N^2 x N^2 matrices acting on vec(rho):
    R : |a><b| -> |Ra><Rb|                  reflection j -> -j mod L
    S : |a><b| -> (-1)^{|a|+|b|} |b><a|     rho -> C rho^T C,  C = prod_j Z_j
    K : |a><b| -> |b><a|                    rho -> rho^dag is K followed by complex conjugation
  """
  N = len(states)
  a, b = np.divmod(np.arange(N * N), N)
  refl = np.array([index[reflection(L, s)] for s in states])
  parity = np.array([bin(s).count("1") % 2 for s in states])
  shape = (N * N, N * N)
  R = sp.csr_matrix((np.ones(N * N), (refl[a] * N + refl[b], a * N + b)), shape=shape)
  S = sp.csr_matrix((1.0 - 2.0 * ((parity[a] + parity[b]) % 2), (b * N + a, a * N + b)), shape=shape)
  K = sp.csr_matrix((np.ones(N * N), (b * N + a, a * N + b)), shape=shape)
  return R, S, K

def symmetry_blocks(B, R, S):
  """
  Projectors of the sector with basis B onto the four blocks (sigma, tau), the characters of
  {1, R, S, RS}:  P = (1 + sigma R)(1 + tau S) / 4, as sparse matrices in the sector basis.
  Also returns max|R B - B (B^T R B)| and the same for S (the sector must be invariant).
  """
  one = sp.identity(B.shape[1], format='csr')
  R_Q = (B.T @ R @ B).tocsr()
  S_Q = (B.T @ S @ B).tocsr()
  residual = max(abs(R @ B - B @ R_Q).max(), abs(S @ B - B @ S_Q).max())
  blocks = {}
  for sigma in (1, -1):
    for tau in (1, -1):
      blocks[(sigma, tau)] = (0.25 * (one + sigma * R_Q) @ (one + tau * S_Q)).tocsr()
  return blocks, residual

def block_sizes(L, constrained=True, step=1):
  """
  Sizes of the blocks (sigma, tau) of the sectors of sectors(L, step) without building anything,
  from the number F(g) of configurations fixed by g (U = T^step, nt = L / step):

      dim = 1/(4 nt) sum_{d<nt} s_Q^d [ F(U^d)^2 + sigma F(U^d R)^2 + tau F(U^{2d}) + sigma tau N ],

  s_Q = +1 (Q = 0), -1 (Q = nt/2). Returns {Q: {(sigma, tau): dim}}.
  """
  mask = (1 << L) - 1
  x = np.arange(1 << L, dtype=np.int64)
  if constrained:
    x = x[((x & (x >> 1)) == 0) & ~(((x & 1) == 1) & (((x >> (L - 1)) & 1) == 1))]
  N = len(x)
  rot = lambda y, d: y if d % L == 0 else ((y << (d % L)) | (y >> (L - d % L))) & mask
  refl = np.zeros_like(x)
  for j in range(L):
    refl |= ((x >> j) & 1) << ((L - j) % L)
  fixed = lambda y: int(np.count_nonzero(y == x))

  nt = L // step
  sizes = {}
  for Q in sectors(L, step):
    total = {(sigma, tau): 0 for sigma in (1, -1) for tau in (1, -1)}
    for d in range(nt):
      s_Q = -1 if (Q != 0 and d % 2) else 1
      f_T, f_TR, f_T2 = fixed(rot(x, step * d)), fixed(rot(refl, step * d)), fixed(rot(x, 2 * step * d))
      for (sigma, tau) in total:
        total[(sigma, tau)] += s_Q * (f_T**2 + sigma * f_TR**2 + tau * f_T2 + sigma * tau * N)
    assert all(v % (4 * nt) == 0 for v in total.values())
    sizes[Q] = {key: v // (4 * nt) for key, v in total.items()}
  return sizes
