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

def generation_basis(L):
  rep_states = fibonacci_basis(L)
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

def dissipation(L, states, index, gamma_plus, gamma_minus):
  N = len(states)
  I = sp.identity(N, format='csr')
  D_minus = sp.csr_matrix((N**2, N**2))
  D_plus = sp.csr_matrix((N**2, N**2))

  for i in range(L):

    L_minus_i = sp.lil_matrix((N, N))
    L_plus_i = sp.lil_matrix((N, N))

    for state in states:
      if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
        state_p = state ^ (1 << i)
        if state & 1<<i:
          L_minus_i [ index[state_p], index[state]] += 1
        else:
          L_plus_i [ index[state_p], index[state]] += 1

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

def translation(L, state):
  return ((state << 1) | (state >> (L - 1))) & ((1 << L) - 1)

def sector_basis(L, states, index, Q):
  """
  Orthonormal basis of the operators with T rho T^dag = e^{-i 2 pi Q / L} rho, for Q = 0 or L/2:

      (1/sqrt(p)) sum_{d<p} s^d |T^d a><T^d b|,   s = +1 (Q = 0), -1 (Q = L/2),

  one vector per orbit of pairs (a, b) with period p (only even p for Q = L/2).
  Returns B (N^2 x dim, real, B^T B = 1).
  """
  if Q not in (0, L // 2) or L % 2:
    raise ValueError("only Q = 0 and Q = L/2 (even L) are implemented")

  N = len(states)
  T = [index[translation(L, s)] for s in states]
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
      if (Q * p) % L != 0:
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

def build_model(L, gamma_plus, gamma_minus, omega):
  states, index = generation_basis(L)
  H = Hamiltonian(L, states, index, omega)
  D = dissipation(L, states, index, gamma_plus, gamma_minus)
  Lind = lindblad_evolution(H, D)
  return states, index, Lind

def sectors(L):
  return [0, L // 2]
