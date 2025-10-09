import numpy as np

import matplotlib.pyplot as plt

#############################################################################
###################### GENERATION OF THE BASIS ##############################
#############################################################################

def fibonacci_basis(L, pbc=False):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      if pbc and 2**0 & i and 2**(L-1) & i:
        continue
      else:
        states.append(i)
  return states

def translationally_invariant_basis(L):
  states= fibonacci_basis(L, pbc=True)
  rep_states = []
  rep_index = {}

  basis = set(states)

  while len(basis) > 0:

    state = basis.copy().pop()

    shifted_states = {((state << i) | (state >> (L - i))) & ((1 << L) - 1) for i in range(L)}

    min_state = min(shifted_states)

    basis = set(basis) ^ set(shifted_states)

    rep_states.append(min_state)

    rep_index[min_state] = [len(shifted_states)]

  return rep_states, rep_index

def generation_basis(L, pbc=False):
  if pbc:
    rep_states, rep_index = translationally_invariant_basis(L)
    for i, s in enumerate(rep_states):
      rep_index[s].append(i)
      rep_index[s] = rep_index[s][::-1]
  else:
    rep_states = fibonacci_basis(L, pbc=False)
    rep_index = {s: i for i,  s in enumerate(rep_states)}
  return rep_states, rep_index

#############################################################################
###################### GENERATION OF THE BASIS ##############################
#############################################################################

def fibonacci_basis(L, pbc=False):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      if pbc and 2**0 & i and 2**(L-1) & i:
        continue
      else:
        states.append(i)
  return states

#############################################################################
###################### GENERATION OF THE DISSIPATOR #########################
#############################################################################

def W_matrix(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
  W_minus = np.zeros((len(states), len(states)), dtype=complex)
  W_plus = np.zeros((len(states), len(states)), dtype=complex)

  if pbc:

    for i in range(L):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
      L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
          state_i_p = state ^ (1 << i)
          state_p = state_i_p
          d = 0
          for j in range(L):
            state_ii_p = ((state_i_p << j) | (state_i_p >> (L - j))) & ((1 << L) - 1)

            if state_ii_p < state_p:
              state_p = state_ii_p
              d = j
          if state & 1<<i:
            L_minus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_plus")

      W_minus += np.multiply(L_minus_i, L_minus_i.conj()) - np.diag(np.diag(L_minus_i.conj().T @ L_minus_i))
      W_plus += np.multiply(L_plus_i, L_plus_i.conj()) - np.diag(np.diag(L_plus_i.conj().T @ L_plus_i))

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W

  else:
    for i in range(1,L-1):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
      L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> (i-1)) & 1) == 0 and ((state >> (i+1)) & 1) == 0:
          state_p = state ^ (1 << i)
          d = 0
          if state & 1<<i:
            L_minus_i [ index[state_p], index[state]] += 1
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p], index[state]] += 1
            # print("gamma_plus")

      W_minus += np.multiply(L_minus_i, L_minus_i.conj()) - np.diag(np.diag(L_minus_i.conj().T @ L_minus_i))
      W_plus += np.multiply(L_plus_i, L_plus_i.conj()) - np.diag(np.diag(L_plus_i.conj().T @ L_plus_i))

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W


#############################################################################
########################## AVG OCCUPATION ###################################
#############################################################################

def avg_occupation (L, states, index, pbc=False, k=0):
  n_avg = np.zeros((len(states),len(states)))

  if pbc:
    for state in states:
      for i in range(L):
        if state & (1 << i):
          n_avg [ index[state][0], index[state][0]] += 1
        # else:
        #   n_avg [ index[state][0], index[state][0]] -= 1

  else:
    for state in states:
      for i in range(L):
        if state & (1 << i):
          n_avg [ index[state], index[state]] += 1
        # else:
        #   n_avg [ index[state], index[state]] -= 1

  return n_avg/L

#############################################################################
########################### MAGNETIZATION ###################################
#############################################################################

def magnetization (L, states, index, pbc=False, k=0):
  S_z = np.zeros((len(states),len(states)))

  if pbc:
    for state in states:
      for i in range(L):
        if state & (1 << i):
          S_z [ index[state][0], index[state][0]] += 1
        else:
          S_z [ index[state][0], index[state][0]] -= 1

  else:
    for state in states:
      for i in range(L):
        if state & (1 << i):
          S_z [ index[state], index[state]] += 1
        else:
          S_z [ index[state], index[state]] -= 1

  return S_z/L

#############################################################################
#############################################################################
#############################################################################

L = 20
PBC = True
k_sector = 0
basis = generation_basis(L, pbc=PBC)

gamma_plus = 0.5
gamma_minus = 0.01
z =  gamma_plus / gamma_minus

W = W_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)

eig_vals, eig_vecs = np.linalg.eig(W)

for i in range(len(eig_vals)):
  if np.isclose(eig_vals[i], 0):
    pn_ss = 0.5 * ( eig_vecs[:,i] / np.sum(eig_vecs[:,i]) + (eig_vecs[:,i] / np.sum(eig_vecs[:,i])).conj() )
    pn_ss = pn_ss.real
    print("Control sum:", np.sum(pn_ss.imag))
    # print("Eigenvalue:", eig_vals[i])
    # print("Eigenvector:", eig_vecs[:,i])

n_avg = avg_occupation(L, basis[0], basis[1], pbc=PBC, k=k_sector)

print("Z =", z)

mpa_val = 2 * z / ( 1 + 4 * z + np.sqrt( 1 + 4 * z ))

print(np.sum(pn_ss))

exp_val_n_avg = np.trace( np.diag(pn_ss) @ n_avg )
print("Steady state occupation:", exp_val_n_avg)
print("Product Matrix value:", mpa_val)

print("Relative error:", np.abs(exp_val_n_avg - mpa_val) / mpa_val)
