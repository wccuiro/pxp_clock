import numpy as np

import matplotlib.pyplot as plt

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


def transition_matrix(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
  W = np.zeros((len(states), len(states)), dtype=complex)
 
  if pbc:
    for state in states:
      for i in range(L):
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
            W [ index[state_p][0], index[state][0]] += gamma_minus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")
          else:
            W [ index[state_p][0], index[state][0]] += gamma_plus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_plus")          
          W [ index[state][0], index[state][0] ] -= gamma_plus if not (state & 1<<i) else gamma_minus
          # print("{:04b} --h-- {:04b} -- T -- {:04b}".format(state, state_i_p, state_p),i,d)
          # print("Index:", index[state][0], index[state][1], index[state_p][0], index[state_p][1])
        
        
  else:
    for state in states:
      for i in range(1,L-1):
        if (state >> (i-1)) & 1 == 0 and (state >> (i+1)) & 1 == 0:
          state_p = state ^ 2**i
          if state & 1<<i:
            W [ index[state_p], index[state]] += gamma_minus
          else:
            W [ index[state_p], index[state]] += gamma_plus
          W [ index[state], index[state] ] -= gamma_plus if not (state & 1<<i) else gamma_minus
            # print("{:04b} --h-- {:04b}".format(state, state_i_p),i)
  return W       

L = 20
gamma_plus = 10.0
gamma_minus = 1.0

basis = generation_basis(L, pbc=True)

# for i in basis[0]:
#   print("{:04b}".format(i))

W = transition_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=True)

eigenvalues, eigenvectors = np.linalg.eig(W)

neel_state = sum(1 << i for i in range(0, L, 2))
index_neel = basis[1][neel_state][0]
neel_proj = np.abs(eigenvectors[index_neel])**2


# dif = np.max(np.abs(W - W.T))
# print("Max difference from Hermitian:", dif)

plt.matshow(np.real(W), cmap='viridis')
plt.colorbar()
plt.title("Transition matrix |W|")
plt.show()

plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo')
plt.show()

plt.plot(np.real(eigenvalues), neel_proj, 'bo')
# plt.plot(scars_vals, np.abs(scars_vecs[index_neel])**2, 'ro')
plt.show()


print(W)
