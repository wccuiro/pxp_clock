import math
import numpy as np

import matplotlib.pyplot as plt

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

def translationally_invariant_basis(L):
  states= fibonacci_basis(L)
  rep_states = []
  rep_index = {}

  basis = set(states)

  while len(basis) > 0:

    state = basis.copy().pop()

    shifted_states = {((state << i) | (state >> (L - i))) & ((1 << L) - 1) for i in range(L)}

    min_state = min(shifted_states)

    basis = set(basis) ^ set(shifted_states)

    rep_states.append(min_state)

    rep_index[min_state] = [((min_state << i) | (min_state >> (L - i))) & ((1 << L) - 1) for i in range(L)]

  return rep_states, rep_index

def generation_basis(L, t_inv=False):
  if t_inv:
    rep_states, rep_index = translationally_invariant_basis(L)
    for i, s in enumerate(rep_states):
      rep_index[s].append(i)
  else:
    rep_states = fibonacci_basis(L)
    rep_index = {s: i for i,  s in enumerate(rep_states)}
  return rep_states, rep_index

#############################################################################
############################ JUMP OPERATORS #################################
#############################################################################

threshold = 1e-10
def normalization_factor(L, states, k=0):
  norm2 = 0
  for j in range(L):
    for i in range(L):
      if states[j] == states[i]:
        norm2 += np.exp(1j*2*np.pi*k*(j-i)/L)
  norm2 = norm2 if np.abs(norm2) > threshold else 0
  return np.sqrt(norm2)

def dissipators_spin_k_kp(L, rep_basis, full_basis, spin, t_inv=False, k=0, kp=0):

  if t_inv:
    L_plus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_minus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_dagger_L_minus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_dagger_L_plus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)

    for init in rep_basis:
      for target in rep_basis:
        norm_init = normalization_factor(L, full_basis[init], k)
        norm_target = normalization_factor(L, full_basis[target], kp)
        # print(norm_init)

        if norm_init > threshold and norm_target > threshold:
          for i in range(L):
            for t in range(L):

              if ((full_basis[init][i] >> ((spin-1)%L)) & 1) == 0 and ((full_basis[init][i] >> ((spin+1)%L)) & 1) == 0:
                state_p = full_basis[init][i] ^ (1 << spin)

                if state_p & 1<<spin and state_p == full_basis[target][t]:
                  # print(full_basis[init][i],full_basis[target][t])
                  L_plus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*(k*i-kp*t)/L) / ( norm_init * norm_target)

                if not (state_p & 1<<spin) and state_p == full_basis[target][t]:
                  L_minus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*(k*i-kp*t)/L) / ( norm_init * norm_target)

                if full_basis[init][i] & 1<<spin and full_basis[init][i] == full_basis[target][t]:
                  L_dagger_L_minus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*(k*i-kp*t)/L) / ( norm_init * norm_target)

                if not(full_basis[init][i] & 1<<spin) and full_basis[init][i] == full_basis[target][t]:
                  L_dagger_L_plus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*(k*i-kp*t)/L) / ( norm_init * norm_target)



  else:
    L_plus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_minus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
  
    for state in rep_basis:
      if ((state >> ((spin-1)%L)) & 1) == 0 and ((state >> ((spin+1)%L)) & 1) == 0:
        state_p = state ^ (1 << spin)
        d = 0
        if state & 1<<spin:
          L_minus_spin [ full_basis[state_p], full_basis[state]] += 1
          # print("gamma_minus")
        else:
          L_plus_spin [ full_basis[state_p], full_basis[state]] += 1
          # print("gamma_plus")
    
    L_dagger_L_minus_spin = L_minus_spin.conj().T @ L_minus_spin
    L_dagger_L_plus_spin = L_plus_spin.conj().T @ L_plus_spin


  return L_plus_spin, L_minus_spin, L_dagger_L_minus_spin, L_dagger_L_plus_spin 

#############################################################################
###################### GENERATION OF THE DISSIPATOR #########################
#############################################################################

def W_matrix(L, states, index, gamma_plus, gamma_minus, t_inv=False, nQ=0):
  W_minus = np.zeros((len(states)**2, len(states)**2), dtype=complex)
  W_plus = np.zeros((len(states)**2, len(states)**2), dtype=complex)

  if t_inv:
    Q = 2 * nQ * np.pi / L
    L_plus_i = np.zeros((len(states),len(states)), dtype=complex)
    L_minus_i = np.zeros((len(states),len(states)), dtype=complex)
    L_dagger_L_minus_spin = np.zeros((len(states),len(states)), dtype=complex)
    L_dagger_L_plus_spin = np.zeros((len(states),len(states)), dtype=complex)

    L_plus_i_Q = np.zeros((len(states),len(states)), dtype=complex)
    L_minus_i_Q = np.zeros((len(states),len(states)), dtype=complex)
    L_dagger_L_minus_spin_Q = np.zeros((len(states),len(states)), dtype=complex)
    L_dagger_L_plus_spin_Q = np.zeros((len(states),len(states)), dtype=complex)

    for i in range(L):

      for nk in range(L):
          for nkp in range(L):
            k = 2 * nk * np.pi / L
            kp = 2 * nkp * np.pi / L
            L_plus_i_k_kp, L_minus_i_k_kp, L_dagger_L_minus_spin_k_kp, L_dagger_L_plus_spin_k_kp = dissipators_spin_k_kp(L, states, index, i, t_inv, k=k, kp=kp)
            L_plus_i_k_kp_Q, L_minus_i_k_kp_Q, L_dagger_L_minus_spin_k_kp_Q, L_dagger_L_plus_spin_k_kp_Q = dissipators_spin_k_kp(L, states, index, i, t_inv, k=k-Q, kp=kp-Q)
            L_plus_i += L_plus_i_k_kp
            L_plus_i_Q += L_minus_i_k_kp_Q
            
            L_minus_i += L_minus_i_k_kp
            L_minus_i_Q += L_plus_i_k_kp_Q
            
            L_dagger_L_minus_spin += L_dagger_L_minus_spin_k_kp
            L_dagger_L_minus_spin_Q += L_dagger_L_minus_spin_k_kp_Q
            
            L_dagger_L_plus_spin += L_dagger_L_plus_spin_k_kp
            L_dagger_L_plus_spin_Q += L_dagger_L_plus_spin_k_kp_Q

      W_minus += np.kron(L_minus_i, L_minus_i_Q.T) - 0.5 * np.kron(L_dagger_L_minus_spin, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_dagger_L_minus_spin_Q).T)
      W_plus += np.kron(L_plus_i, L_plus_i_Q.T) - 0.5 * np.kron(L_dagger_L_plus_spin, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_dagger_L_plus_spin_Q).T)

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W

  else:
    for i in range(L):

      L_plus_i, L_minus_i, L_dagger_L_minus_spin, L_dagger_L_plus_spin = dissipators_spin_k_kp(L, states, index, i, t_inv, k=k)

      W_minus += np.kron(L_minus_i, L_minus_i.conj()) - 0.5 * np.kron(L_dagger_L_minus_spin,np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), L_dagger_L_minus_spin.T)
      W_plus += np.kron(L_plus_i, L_plus_i.conj()) - 0.5 * np.kron(L_dagger_L_plus_spin,np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), L_dagger_L_plus_spin.T)

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W


#############################################################################
#############################################################################
#############################################################################

L = 6
T_INV = True
Q_sector = 0
basis = generation_basis(L, t_inv=T_INV)

gamma_plus = 1.0
gamma_minus = 1.5
z =  gamma_plus / gamma_minus


W = W_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, t_inv=T_INV, nQ=Q_sector)

print("Building finished. Dimension of the dissipator:", W.shape)

eig_vals, eig_vecs = np.linalg.eig(W)
print("Eigenvalues of the dissipator:")
for i in eig_vals:
  print(i.real)
  # print(i.imag)