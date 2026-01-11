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

threshold = 1e-10
def normalization_factor(L, states, k=0):
  norm2 = 0
  for j in range(L):
    for i in range(L):
      if states[j] == states[i]:
        norm2 += np.exp(1j*2*np.pi*k*(j-i)/L)
  norm2 = norm2 if np.abs(norm2) > threshold else 0
  return np.sqrt(norm2).real

def basis_per_sector(L, basis):
  basis_sector = {}
  for k_sector in range(L):
    for state in basis[0]:
      norm = normalization_factor(L, basis[1][state], k=k_sector)
      if norm > 0:
        basis_sector.setdefault(k_sector,[]).append(state)
  return basis_sector

def basis_per_sector_ordered(basis_sector):
  
  states = []
  
  for k_sector in basis_sector.keys():
    # print(f"k={k_sector}:")
    for state_a in basis_sector[k_sector]:
      for state_b in basis_sector[k_sector]:
        states.append([state_a, state_b, k_sector])
  
  return states
# print(f"|{state_a},{k_sector}> x |{state_b},{k_sector}> ")

def inner_product(a_in, b_out, k_sector_in, k_sector_out, L, basis):
  norm_a = normalization_factor(L, a_in, k=k_sector_in)
  norm_b = normalization_factor(L, b_out, k=k_sector_out)
  
  if norm_a == 0 or norm_b == 0:
    return 0
  
  ip = 0
  for i in range(L):
    for j in range(L):
      if a_in[i] == b_out[j]:
        ip += np.exp(1j*2*np.pi*(k_sector_out*j-k_sector_in*i)/L)
  
  return ip/(norm_a*norm_b)

def Hamiltonian_Lindblad_T_inv(L, basis, basis_per_sector_ordered_list, dimension_Q_sector, Q_sector):
  H_cal = np.zeros((dimension_Q_sector, dimension_Q_sector), dtype=complex)
  
  ''' Run over all basis states  '''
  for i, (a_in, b_in, k_in) in enumerate(basis_per_sector_ordered_list):
    for j, (a_out, b_out, k_out) in enumerate(basis_per_sector_ordered_list):
      states_a_in = basis[1][a_in]
      states_b_in = basis[1][b_in]
      states_a_out = basis[1][a_out]
      states_b_out = basis[1][b_out]
      
      ''' Run over spin sites to apply Hamiltonian '''
      norm_a_in = normalization_factor(L, states_a_in, k=k_in)
      norm_b_in = normalization_factor(L, states_b_in, k=k_in)
      norm_a_out = normalization_factor(L, states_a_out, k=k_out)
      norm_b_out = normalization_factor(L, states_b_out, k=k_out)

      for site in range(L):
        states_a_h_in = np.array(states_a_in) ^ (1 << site)
        states_b_h_in = np.array(states_b_in) ^ (1 << site)

        ''' Run over elements of representative states '''
        a_prod = 0
        for ii in range(L):
          for jj in range(L):
            if states_a_h_in[ii] == states_a_out[jj] and states_a_h_in[ii] & (1 << ((site-1)%L)) == 0 and states_a_h_in[ii] & (1 << ((site+1)%L)) == 0:
              a_prod += np.exp(1j*2*np.pi*(k_out*jj-k_in*ii)/L)
        a_prod /= (norm_a_in*norm_a_out) 

        b_prod = 0
        for ii in range(L):
          for jj in range(L):
            if states_b_h_in[ii] == states_b_out[jj] and states_b_h_in[ii] & (1 << ((site-1)%L)) == 0 and states_b_h_in[ii] & (1 << ((site+1)%L)) == 0:
              b_prod += np.exp(1j*2*np.pi*( (k_out - Q_sector)*jj - (k_in - Q_sector)*ii)/L)
        b_prod /= (norm_b_in*norm_b_out)
        
        H_cal[j,i] += -1j * (a_prod * inner_product(states_b_in, states_b_out, k_in - Q_sector, k_out - Q_sector, L, basis) - inner_product(states_a_in, states_a_out, k_in, k_out, L, basis) * b_prod)

  return H_cal

def Dissipation_Lindblad_T_inv(L, basis, basis_per_sector_ordered_list, dimension_Q_sector, Q_sector):
  D_cal = np.zeros((dimension_Q_sector, dimension_Q_sector), dtype=complex)
  
  for i, (a_in, b_in, k_in) in enumerate(basis_per_sector_ordered_list):
    for j, (a_out, b_out, k_out) in enumerate(basis_per_sector_ordered_list):
      states_a_in = basis[1][a_in]
      states_b_in = basis[1][b_in]
      states_a_out = basis[1][a_out]
      states_b_out = basis[1][b_out]
      D_cal[j,i] = inner_product(states_a_in, states_a_out, k_in, k_out, L, basis) * inner_product(states_b_in, states_b_out, k_in - Q_sector, k_out - Q_sector, L, basis)

  return D_cal

L = 6
T_INV = True
Q_sector = 0

basis = generation_basis(L, t_inv=T_INV)

basis_sector = basis_per_sector(L, basis)

dimension_Q_sector = sum(len(states)**2 for states in basis_sector.values())

print(basis[0])

print(len(np.concatenate(list(basis_sector.values()))))

''' Listing basis test'''
basis_per_sector_ordered_list = basis_per_sector_ordered(basis_sector)

H_cal = Hamiltonian_Lindblad_T_inv(L, basis, basis_per_sector_ordered_list, dimension_Q_sector, Q_sector)

eigvlas, eigvecs = np.linalg.eig(H_cal)

print("Eigenvalues Hamiltonian Lindblad:")
print(eigvlas.real)
print(eigvlas.imag)

plt.plot(eigvlas.real, eigvlas.imag, 'o')
plt.show()

print(H_cal)

count = 0

A = np.zeros((dimension_Q_sector, dimension_Q_sector), dtype=object)

for i, (a_in, b_in, k_in) in enumerate(basis_per_sector_ordered_list):
  for j, (a_out, b_out, k_out) in enumerate(basis_per_sector_ordered_list):
    values = f"<{a_out},{b_out},{k_out}| |{a_in},{b_in},{k_in}>"

    A[j,i] = values
    count += 1

print(A)

print(f"Total states listed: {count**0.5}, {dimension_Q_sector}")

total = 0
for k_sector in range(L):
  i = 0
  for state in basis[0]:
    norm = normalization_factor(L, basis[1][state], k=k_sector)
    if norm > 0:
      i += 1
      # basis_sector.setdefault(k_sector,[]).append(state)
      # print(f"State: {state:0{L}b}, Norm: {norm}")
  # print(basis[0], basis_sector)
  total += i
  print(f"Total states in k={k_sector} sector: {i}")

print(f"Total states in all sectors: {total}")

T_INV = False

print(basis[1][1][:L], np.array(basis[1][1][:L]) ^ 2)

basis = generation_basis(L, t_inv=T_INV)

total = len(basis[0])
print(f"Total states without translational invariance: {total}")

