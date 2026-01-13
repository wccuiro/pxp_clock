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
##################### TRACE IN REPRESENTATIVE BASIS #########################
#############################################################################

def main(gamma_plus, gamma_minus):
  global L
  L = 10
  T_INV = False
  k_sector = 0
  basis = generation_basis(L, t_inv=T_INV)

  # for i in basis[0]:
  #   print(f"{i:0{L}b}")

  # gamma_plus = 1.0
  # gamma_minus = 1.5
  z =  gamma_plus / gamma_minus


  W = W_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, t_inv=T_INV, k=k_sector)

  # # You can now use the sparse solver
  # eigenvalues, eigenvectors = eigs(W_sparse, k=1, which='SR', sigma=1e-9)

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

  n_avg = avg_occupation(L, basis[0], basis[1], t_inv=T_INV, k=k_sector)
  n_n = correlation(L, basis[0], basis[1], t_inv=T_INV, k=k_sector)
  n_n_2 = correlation_2(L, basis[0], basis[1], t_inv=T_INV, k=k_sector)

  EEE_matrix, EDE_matrix = EEE_EDE_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, t_inv=T_INV, k=k_sector)

  for i in pn_ss_set:
    print("Control sum 2:", np.sum(i))

  acc_EDE = 0
  acc_EDEDE = 0

  for pn_ss in pn_ss_set:
    print("-------------------------------")
    print("---------- NUMERICAL ----------")
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

    print("-------------------------------")
    print("---------- ANALYTICAL ---------")
    print("-------------------------------")


    print("Steady state occupation:", an_val)
    print("Steady state correlation analytical:", (1 + 3*z)/z * an_val - 1, an_val_n_n)
    print("Steady state correlation expected:", (1 + 3*z)/z * an_val - 1)


    left_side = an_val
    right_side = z * (1 - 3 * an_val + an_val_n_n)

    print(left_side, right_side)
    print("Difference Identity:", left_side - right_side)


    # print(np.sum(pn_ss))
    print("-------------------------------")
    print("---------- COMPARING ----------")
    print("-------------------------------")


    print("Steady state occupation:", an_val, exp_val_EDE)
    print("Steady state EEE:", an_val_E, exp_val_EEE)

    print("-----------------------------------------")
    print("---------- THERMODYNAMIC VALUE ----------")
    print("-----------------------------------------")

    print("Product Matrix value:", mpa_val)

    print("Relative error with numerical:", np.abs(exp_val_n_avg - mpa_val) / mpa_val)
    print("Relative error with analytical:", np.abs(an_val - mpa_val) / mpa_val)

  # print("-------------------------------")
  # print("Accumulated EDE:", acc_EDE)
  # print("Accumulated EDEDE:", acc_EDEDE)
  # print("Steady state correlation accumulated:", (1 + 3*z)/z * acc_EDE-1, acc_EDEDE)

  return exp_val_n_avg, exp_val_n_n, an_val, an_val_n_n, mpa_val, (1 + 3*z)/z * mpa_val -1

g_vals =  np.linspace(1.5, 2, 2)

num_n, num_n_n, an_n, an_n_n, large_N_n, large_N_n_n = [], [], [], [], [], []

for g in g_vals:
  gamma_plus = g
  gamma_minus = 1.0
  results = main(gamma_plus, gamma_minus)
  num_n.append(results[0].real)
  num_n_n.append(results[1].real)
  an_n.append(results[2].real)
  an_n_n.append(results[3].real)
  large_N_n.append(results[4].real)
  large_N_n_n.append(results[5].real)

plt.plot(g_vals, num_n, '*-', label="Numerical <n>", color='blue')
plt.plot(g_vals, an_n, label="Analytical <n>", color='orange', linestyle='dashed')
plt.plot(g_vals, large_N_n, label="Thermodynamic <n>", color='green', linestyle='dotted')
plt.xlabel("g")
plt.ylabel("<n>")
plt.legend()
plt.title("Average Occupation vs g")
plt.grid()
plt.show()
plt.close()

plt.plot(g_vals, num_n_n, '*-', label="Numerical <EDEDE>", color='blue')
plt.plot(g_vals, an_n_n, label="Analytical <EDEDE>", color='orange', linestyle='dashed')
plt.plot(g_vals, large_N_n_n, label="Thermodynamic <EDEDE>", color='green', linestyle='dotted')
plt.xlabel("g")
plt.ylabel("<EDEDE>")
plt.legend()
plt.title("Correlation <EDEDE> vs g")
plt.grid()
plt.show()
plt.close()
