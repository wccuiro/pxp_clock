import math
import numpy as np

from scipy.linalg import expm

import matplotlib.pyplot as plt

#############################################################################
########################## BASIS OF SUBSYSTEM ###############################
#############################################################################

def fibonacci_sub_basis(L):
  states = []
  for i in range(1 << L):
    if i & (i >> 1) == 0:
      states.append(i)
  return states

def generation_sub_basis(L):
  rep_states = fibonacci_sub_basis(L)
  rep_index = {s: i for i,  s in enumerate(rep_states)}
  
  return rep_states, rep_index


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

def dissipators_spin(rep_basis, full_basis, spin, t_inv=False, k=0):

  if t_inv:
    L_plus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_minus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_dagger_L_minus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)
    L_dagger_L_plus_spin = np.zeros((len(rep_basis),len(rep_basis)), dtype=complex)

    for init in rep_basis:
      for target in rep_basis:
        norm_init = normalization_factor(L, full_basis[init], k)
        norm_target = normalization_factor(L, full_basis[target], k)
        # print(norm_init)

        if norm_init > threshold and norm_target > threshold:
          for i in range(L):
            for t in range(L):

              if ((full_basis[init][i] >> ((spin-1)%L)) & 1) == 0 and ((full_basis[init][i] >> ((spin+1)%L)) & 1) == 0:
                state_p = full_basis[init][i] ^ (1 << spin)

                if state_p & 1<<spin and state_p == full_basis[target][t]:
                  # print(full_basis[init][i],full_basis[target][t])
                  L_plus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*k*(i-t)/L) / ( norm_init * norm_target)

                if not (state_p & 1<<spin) and state_p == full_basis[target][t]:
                  L_minus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*k*(i-t)/L) / ( norm_init * norm_target)

                if full_basis[init][i] & 1<<spin and full_basis[init][i] == full_basis[target][t]:
                  L_dagger_L_minus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*k*(i-t)/L) / ( norm_init * norm_target)

                if not(full_basis[init][i] & 1<<spin) and full_basis[init][i] == full_basis[target][t]:
                  L_dagger_L_plus_spin[full_basis[target][-1],full_basis[init][-1]] += np.exp(1j*2*np.pi*k*(i-t)/L) / ( norm_init * norm_target)

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

def W_matrix(L, states, index, gamma_plus, gamma_minus, t_inv=False, k=0):
  W_minus = np.zeros((len(states), len(states)), dtype=complex)
  W_plus = np.zeros((len(states), len(states)), dtype=complex)

  if t_inv:

    for i in range(L):

      L_plus_i, L_minus_i, L_dagger_L_minus_spin, L_dagger_L_plus_spin = dissipators_spin(states, index, i, t_inv, k=k)

      W_minus += np.multiply(L_minus_i, L_minus_i.conj()) - np.diag(np.diag(L_dagger_L_minus_spin))
      W_plus += np.multiply(L_plus_i, L_plus_i.conj()) - np.diag(np.diag(L_dagger_L_plus_spin))

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W

  else:
    for i in range(L):

      L_plus_i, L_minus_i,_,_ = dissipators_spin(states, index, i, t_inv, k=k)

      W_minus += np.multiply(L_minus_i, L_minus_i.conj()) - np.diag(np.diag(L_minus_i.conj().T @ L_minus_i))
      W_plus += np.multiply(L_plus_i, L_plus_i.conj()) - np.diag(np.diag(L_plus_i.conj().T @ L_plus_i))

    W = gamma_minus * W_minus + gamma_plus * W_plus

    return W

#############################################################################
############################## EXP VAL EEE OPERATOR #########################
#############################################################################

def EEE_EDE_matrix(L, states, index, gamma_plus, gamma_minus, t_inv=False, k=0):
  EEE_matrix = np.zeros((len(states), len(states)), dtype=complex)
  EDE_matrix = np.zeros((len(states), len(states)), dtype=complex)

  if t_inv:

    for i in range(L):

      L_plus_i, L_minus_i, L_dagger_L_minus_spin, L_dagger_L_plus_spin = dissipators_spin(states, index, i, t_inv, k=k)

      EEE_matrix += L_dagger_L_plus_spin
      EDE_matrix += L_dagger_L_minus_spin

    return EEE_matrix/L, EDE_matrix/L

  else:
    for i in range(L):

      L_plus_i, L_minus_i,_,_ = dissipators_spin(states, index, i, t_inv, k=k)

      EEE_matrix += L_plus_i.conj().T @ L_plus_i
      EDE_matrix += L_minus_i.conj().T @ L_minus_i

    return EEE_matrix/L, EDE_matrix/L


#############################################################################
########################## AVG OCCUPATION ###################################
#############################################################################

def avg_occupation (L, states, index, t_inv=False, k=0):
  n_avg = np.zeros((len(states),len(states)),dtype=complex)

  if t_inv:

    for i in range(L):

      _, _, L_dagger_L_minus_spin, _ = dissipators_spin(states, index, i, t_inv, k=k)

      n_avg += L_dagger_L_minus_spin

    return n_avg/L

  else:
    for i in range(L):

      _, L_minus_i,_,_ = dissipators_spin(states, index, i, t_inv, k=k)

      n_avg += L_minus_i.conj().T @ L_minus_i

    return n_avg/L

#############################################################################
##################### CORRELATION <Nj-1 Nj+1> ###############################
#############################################################################

def correlation (L, states, index, t_inv=False, k=0):
  n_n = np.zeros((len(states),len(states)),dtype=complex)

  if t_inv:

    for i in range(L):

      _, _, L_dagger_L_minus_spin_ib,_ = dissipators_spin(states, index, (i-1)%L, t_inv, k=k)
      _, _, L_dagger_L_minus_spin_ia,_ = dissipators_spin(states, index, (i+1)%L, t_inv, k=k)

      n_n += L_dagger_L_minus_spin_ib @ L_dagger_L_minus_spin_ia
      

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

def correlation_2 (L, states, index, t_inv=False, k=0):
  n_n = np.zeros((len(states),len(states)),dtype=complex)

  if t_inv:

    for i in range(L):

      _, _,_,L_dagger_L_plus_spin_ib = dissipators_spin(states, index, (i-1)%L, t_inv, k=k)
      _, _,_,L_dagger_L_plus_spin_ia = dissipators_spin(states, index, (i+1)%L, t_inv, k=k)

      n_n += L_dagger_L_plus_spin_ib @ L_dagger_L_plus_spin_ia

    return n_n/L

  else:
    for i in range(L):

      L_plus_ib, _,_,_ = dissipators_spin(states, index, (i-1)%L, t_inv, k=k)
      L_plus_ia, _,_,_ = dissipators_spin(states, index, (i+1)%L, t_inv, k=k)

      n_n += L_plus_ib.conj().T @ L_plus_ib @ L_plus_ia.conj().T @ L_plus_ia

    return n_n/L

#############################################################################
########################### MAGNETIZATION ###################################
#############################################################################

def magnetization (L, states, index, t_inv=False, k=0):
  S_z = np.zeros((len(states),len(states)))

  if t_inv:
    for i in range(L):
      L_plus_i, L_minus_i,_,_ = dissipators_spin(states, index, i, t_inv, k=k)
      S_z += L_minus_i.conj().T @ L_minus_i - L_plus_i.conj().T @ L_plus_i

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
############################# ANALITYCAL EDEDE ##############################
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
############################# ENTANGLEMENT ENTROPY ##########################
#############################################################################

def partition(n,k):
  left = n>>k
  right = n - (left<<k)
  return left, right

def ent_entropy(L, qn, basis):
  reduced_basis = generation_sub_basis(int(L/2))
  reduced_vector_left = np.zeros(len(reduced_basis[0]))
  reduced_vector_right = np.zeros(len(reduced_basis[0]))
  for i in range(len(basis)):
    reduced_state = partition(basis[i], int(L/2))
    reduced_vector_left[reduced_basis[1][reduced_state[0]]] += qn[i].real
    reduced_vector_right[reduced_basis[1][reduced_state[1]]] += qn[i].real
  return reduced_vector_left, reduced_vector_right

L = 10
T_INV = False
k_sector = 0
basis = generation_basis(L, t_inv=T_INV)

# for i in basis[0]:
#   print(f"{i:0{L}b}")

gamma_plus = 1.0
gamma_minus = 1.5
z =  gamma_plus / gamma_minus


W = W_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, t_inv=T_INV, k=k_sector)

initial_state = np.random.random(len(basis[0]))
initial_state = initial_state / np.sum(initial_state)

final_time = 6.
dt = 0.01
steps = int(final_time / dt)
time = np.linspace(0, final_time, steps)

exp_val_n_avg = []
entropy = []

ent_entropy_left = []
ent_entropy_right = []

for i in time:
  state = expm(W*dt) @ initial_state
  initial_state = state / np.sum(state)
  
  left, right = ent_entropy(L,initial_state,basis[0])
  
  ent_entropy_left.append(-np.sum(np.log(left.real)))
  ent_entropy_right.append(-np.sum(np.log(right.real)))

  pn_ss = np.diag(initial_state)

  entropy.append(-np.sum(np.log(initial_state.real)))
  
  n_avg = avg_occupation(L, basis[0], basis[1], t_inv=T_INV, k=k_sector)

  exp_val_n_avg.append(np.trace( pn_ss @ n_avg ).real)

plt.plot(time, exp_val_n_avg, label='Avg Occupation')
plt.xlabel('Time')
plt.ylabel('<n>')
plt.grid()
plt.show()
plt.close()

plt.plot(time, entropy)
plt.xlabel('Time')
plt.ylabel('S')
plt.grid()
plt.show()
plt.close()

plt.plot(time, ent_entropy_left, label='Left L/2')
plt.plot(time, ent_entropy_right, label='Right L/2')
plt.xlabel('Time')
plt.ylabel(r'$S_{ent}$')
plt.legend()
plt.grid()
plt.show()
plt.close()

# eig_vals, eig_vecs = np.linalg.eig(W)