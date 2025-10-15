import math
import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

#############################################################################
###################### GENERATION OF THE BASIS ##############################
#############################################################################

def fibonacci_basis(L, pbc=False):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      if 2**0 & i and 2**(L-1) & i:
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
############################## EXP VAL EEE OPERATOR #########################
#############################################################################

def EEE_EDE_matrix(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
  EEE_matrix = np.zeros((len(states), len(states)), dtype=complex)
  EDE_matrix = np.zeros((len(states), len(states)), dtype=complex)

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

      EEE_matrix += L_plus_i.conj().T @ L_plus_i
      EDE_matrix += L_minus_i.conj().T @ L_minus_i

    return EEE_matrix/L, EDE_matrix/L

  else:
    for i in range(L):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
      L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> (i-1)%L) & 1) == 0 and ((state >> (i+1)%L) & 1) == 0:
          state_p = state ^ (1 << i)
          d = 0
          if state & 1<<i:
            L_minus_i [ index[state_p], index[state]] += 1
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p], index[state]] += 1
            # print("gamma_plus")

      EEE_matrix += L_plus_i.conj().T @ L_plus_i
      EDE_matrix += L_minus_i.conj().T @ L_minus_i

    return EEE_matrix/L, EDE_matrix/L


#############################################################################
########################## AVG OCCUPATION ###################################
#############################################################################

def avg_occupation (L, states, index, pbc=False, k=0):
  n_avg = np.zeros((len(states),len(states)),dtype=complex)

  if pbc:

    for i in range(L):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)

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

      n_avg += L_minus_i.conj().T @ L_minus_i

    return n_avg/L

  else:
    for i in range(L):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
          state_p = state ^ (1 << i)
          d = 0
          if state & 1<<i:
            L_minus_i [ index[state_p], index[state]] += 1
            # print("gamma_minus")

      n_avg += L_minus_i.conj().T @ L_minus_i

    return n_avg/L

#############################################################################
##################### CORRELATION <Nj-1 Nj+1> ###############################
#############################################################################

def correlation (L, states, index, pbc=False, k=0):
  n_n = np.zeros((len(states),len(states)),dtype=complex)

  n_i_e = np.eye(len(states))
  n_i_o = np.eye(len(states))
  old_e = 0
  old_o = 1
  if pbc:

    for i in range(L+2):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
          state_i_p = state ^ (1 << i%L)
          state_p = state_i_p
          d = 0
          for j in range(L):
            state_ii_p = ((state_i_p << j) | (state_i_p >> (L - j))) & ((1 << L) - 1)

            if state_ii_p < state_p:
              state_p = state_ii_p
              d = j
          if state & 1<<(i%L):
            L_minus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")

      n_i = L_minus_i.conj().T @ L_minus_i
      
      if i%2 == 0:
        if i != 0:
          n_n += n_i_e @ n_i
          # print(old_e%L,i%L)
        n_i_e = n_i
        old_e = i
      else:
        if i != 1:
          n_n += n_i_o @ n_i
          # print(old_o%L,i%L)
        n_i_o = n_i
        old_o = i

    return n_n/L

  else:
    for i in range(L):

      for state in states:
        if ((state >> ((i-1)%L)) & 1) != 0 and ((state >> ((i+1)%L)) & 1) != 0:
          state_p = state ^ (1 << i%L)
          if not state & 1<<(i%L):
            n_n [ index[state], index[state]] += 1
            # print("gamma_minus")

    return n_n/L

#############################################################################
##################### CORRELATION <EEEEE> ###############################
#############################################################################

def correlation_2 (L, states, index, pbc=False, k=0):
  n_n = np.zeros((len(states),len(states)),dtype=complex)

  n_i_e = np.eye(len(states))
  n_i_o = np.eye(len(states))
  old_e = 0
  old_o = 1
  if pbc:

    for i in range(L+2):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
          state_i_p = state ^ (1 << i%L)
          state_p = state_i_p
          d = 0
          for j in range(L):
            state_ii_p = ((state_i_p << j) | (state_i_p >> (L - j))) & ((1 << L) - 1)

            if state_ii_p < state_p:
              state_p = state_ii_p
              d = j
          if not state & 1<<(i%L):
            L_minus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")

      n_i = L_minus_i.conj().T @ L_minus_i
      
      if i%2 == 0:
        if i != 0:
          n_n += n_i_e @ n_i
          # print(old_e%L,i%L)
        n_i_e = n_i
        old_e = i
      else:
        if i != 1:
          n_n += n_i_o @ n_i
          # print(old_o%L,i%L)
        n_i_o = n_i
        old_o = i

    return n_n/L

  else:
    for i in range(1,L+1):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
          state_p = state ^ (1 << i%L)
          if state & 1<<(i%L):
            L_minus_i [ index[state_p], index[state]] += 1
            # print("gamma_minus")

      n_i = L_minus_i.conj().T @ L_minus_i
      
      if i%2 == 0:
        if i != 0:
          n_n += n_i_e @ n_i
          # print(old_e%L,i%L)
        n_i_e = n_i
        old_e = i
      else:
        if i != 1:
          n_n += n_i_o @ n_i
          # print(old_o%L,i%L)
        n_i_o = n_i
        old_o = i

    return n_n/L
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
######################## ANALITYCAL NUMBER PARTICLES ########################
#############################################################################

def analytical_occupation(L, gamma_plus, gamma_minus):
  z = gamma_plus / gamma_minus
  num = 0
  denom = 0
  num_limit = math.ceil((L-3)/2)
  denom_limit = math.floor(L/2)
  limit_loop = max(num_limit, denom_limit)+1
  for n in range(limit_loop):
    if n <= num_limit:
      num += np.power(z,n+1) * math.comb(L-n-2,n)
    if n <= denom_limit:
      denom += np.power(z,n) * math.comb(L-n,n) * L / (L-n)
  
  return num/denom

#############################################################################
############################# ANALITYCAL EEE ################################
#############################################################################

def analytical_EEE(L, gamma_plus, gamma_minus):
  z = gamma_plus / gamma_minus
  num = 0
  denom = 0
  num_limit = math.ceil((L-3)/2)
  denom_limit = math.floor(L/2)
  limit_loop = max(num_limit, denom_limit)+1
  for n in range(limit_loop):
    if n <= num_limit:
      num += np.power(z,n) * math.comb(L-n-2,n)
    if n <= denom_limit:
      denom += np.power(z,n) * math.comb(L-n,n) * L / (L-n)
  
  return num/denom

#############################################################################
############################# ANALITYCAL EDEDE ################################
#############################################################################

def analytical_EDEDE(L, gamma_plus, gamma_minus):
  z = gamma_plus / gamma_minus
  num = 0
  denom = 0
  num_limit = math.ceil((L-5)/2)
  denom_limit = math.floor(L/2)
  limit_loop = max(num_limit, denom_limit)+1
  for n in range(limit_loop):
    if n <= num_limit:
      num += np.power(z,n+2) * math.comb(L-n-4,n)
    if n <= denom_limit:
      denom += np.power(z,n) * math.comb(L-n,n) * L / (L-n)
  
  return num/denom

#############################################################################
#############################################################################
#############################################################################


L = 10
PBC = True
k_sector = 0
basis = generation_basis(L, pbc=PBC)

# for i in basis[0]:
#   print(f"{i:0{L}b}")

gamma_plus = 1.0
gamma_minus = 1.5
z =  gamma_plus / gamma_minus

W = W_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)



# # Convert the dense matrix to a CSR sparse matrix
# W_sparse = csr_matrix(W)

# # You can now use the sparse solver
# eigenvalues, eigenvectors = eigs(W_sparse, k=1, which='SR', sigma=1e-9)

eig_vals, eig_vecs = np.linalg.eig(W)

ss = 0
non_ss = 0
pn_ss_set = []
for i in range(len(eig_vals)):
  if np.isclose(eig_vals[i], 0):
    pn_ss = 0.5 * ( eig_vecs[:,i] / np.sum(eig_vecs[:,i]) + (eig_vecs[:,i] / np.sum(eig_vecs[:,i])).conj() )
    pn_ss = pn_ss.real
    pn_ss_set.append(pn_ss/np.sum(pn_ss))
    print("Control sum:", np.sum(pn_ss.imag))
    ss += 1
  else:
    non_ss += 1
    # print("Eigenvalue:", eig_vals[i])
    # print("Eigenvector:", eig_vecs[:,i])

pn_ss_set = np.array(pn_ss_set).reshape(-1, len(basis[0]))

print("-------------------------------")
print("(steady states) + (non steady states) = base")
print(ss,"+" , non_ss, "=", ss+non_ss, "=", len(basis[0]))
print("-------------------------------")

n_avg = avg_occupation(L, basis[0], basis[1], pbc=PBC, k=k_sector)
n_n = correlation(L, basis[0], basis[1], pbc=PBC, k=k_sector)
n_n_2 = correlation_2(L, basis[0], basis[1], pbc=PBC, k=k_sector)

EEE_matrix, EDE_matrix = EEE_EDE_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)

for i in pn_ss_set:
  print("Control sum 2:", np.sum(i))

acc_EDE = 0
acc_EDEDE = 0

for pn_ss in pn_ss_set:
  print("-------------------------------")

  exp_val_n_avg = np.trace( np.diag(pn_ss) @ n_avg )
  exp_val_n_n = np.trace( np.diag(pn_ss) @ n_n )
  exp_val_n_n_2 = np.trace( np.diag(pn_ss) @ n_n_2 )


  exp_val_EDE = np.trace( np.diag(pn_ss) @ EDE_matrix )
  exp_val_EEE = np.trace( np.diag(pn_ss) @ EEE_matrix )
  
  acc_EDE += exp_val_n_avg.real
  print("Occupation <n>:", exp_val_n_avg, exp_val_EDE)

  print("gamma_minus * <EDE> =", gamma_minus * exp_val_EDE)
  print("gamma_plus * <EEE>  =", gamma_plus * exp_val_EEE)

  print("Correlation <EDEDE>:", exp_val_n_n)
  
  acc_EDEDE += exp_val_n_n.real
  
  print("Correlation <EEEEE> * z**2:", exp_val_n_n_2 * z**2)


  print("Difference:", gamma_minus * exp_val_EDE - gamma_plus * exp_val_EEE)
  print("Sum:" , exp_val_EDE + exp_val_EEE)

  mpa_val = 2 * z / ( 1 + 4 * z + np.sqrt( 1 + 4 * z ))

  an_val = analytical_occupation(L, gamma_plus, gamma_minus)
  an_val_E = analytical_EEE(L, gamma_plus, gamma_minus)
  an_val_n_n = analytical_EDEDE(L, gamma_plus, gamma_minus)


  print("Steady state occupation:", exp_val_n_avg, exp_val_EDE)

  print("Steady state correlation:", exp_val_n_n, an_val_n_n)
  print("Steady state correlation analytical:", (1 + 3*z)/z * an_val - 1, an_val_n_n)
  print("Steady state correlation expected:", (1 + 3*z)/z * exp_val_n_avg - 1)


  left_side = exp_val_n_avg
  right_side = z * (1 - 3 * exp_val_n_avg + exp_val_n_n)

  print(left_side, right_side)
  print("Difference:", left_side - right_side)


  # print(np.sum(pn_ss))

  print("Steady state occupation:", exp_val_n_avg)
  print("Analytical occupation:", an_val, exp_val_EDE)
  print("Analytical EEE:", an_val_E, exp_val_EEE)
  print("Product Matrix value:", mpa_val)

  print("Relative error:", np.abs(exp_val_n_avg - mpa_val) / mpa_val)

print("-------------------------------")
print("Accumulated EDE:", acc_EDE/3)
print("Accumulated EDEDE:", acc_EDEDE/3)
print("Steady state correlation accumulated:", (1 + 3*z)/z * acc_EDE/3-1, acc_EDEDE/3)
