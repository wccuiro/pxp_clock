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

def Hamiltonian(L, states, index, pbc=False, k=0):
  
  H = np.zeros((len(states), len(states)), dtype=complex)
  
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
          H [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
          # print("{:04b} --h-- {:04b} -- T -- {:04b}".format(state, state_i_p, state_p),i,d)
          # print("Index:", index[state][0], index[state][1], index[state_p][0], index[state_p][1])
  else:
    for state in states:
      for i in range(1,L-1):
        if (state >> (i-1)) & 1 == 0 and (state >> (i+1)) & 1 == 0:
            state_p = state ^ 2**i
            H [ index[state], index[state_p] ] += 1

            # print("{:04b} --h-- {:04b}".format(state, state_i_p),i)
  return H       

# print(fibonacci_basis)

L = 22

gamma_plus = 10.0
gamma_minus = 1.0

basis = generation_basis(L, pbc=True)

H = Hamiltonian(L, basis[0], basis[1], pbc=True)

eigenvalues_H, eigenvectors_H = np.linalg.eigh(H)

neel_state = sum(1 << i for i in range(0, L, 2))
index_neel = basis[1][neel_state][0]
neel_proj = np.abs(eigenvectors_H[index_neel])**2

scars_vals = np.zeros(L)
scars_vecs = np.zeros((len(basis[0]),L), dtype=complex)
j=0
for i, val in enumerate(neel_proj):
  if val >= np.sort(neel_proj)[-L]:
    scars_vals[j] = eigenvalues_H[i]
    scars_vecs[:,j] = eigenvectors_H[:,i]
    j+=1



# for i in basis[0]:
#   print("{:04b}".format(i))

W = transition_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=True)

eigenvalues_W, eigenvectors_W = np.linalg.eig(W)

# Filterign steady states
threshold = 1e-10
number_steady_states = np.sum(np.abs(eigenvalues_W) < threshold)
steady_states= np.zeros((number_steady_states,eigenvectors_W.shape[0]), dtype=complex)
index_steady = []
k=0
for i in range(len(eigenvalues_W)):
  if np.abs(eigenvalues_W[i]) < threshold:
    steady_states[k] = eigenvectors_W[:,i]
    index_steady.append(i)
    print("Steady state:", eigenvalues_W[i])    
    k+=1

print(steady_states.shape)
print(scars_vecs.shape)

steady_over_scar = np.abs((steady_states @ scars_vecs))**2

plt.matshow(steady_over_scar, cmap='viridis', vmin=0, vmax=1)
plt.colorbar()
plt.title("Steady states overlap with scar subspace")
plt.show()

# for i in steady_states:
#   print(i @ scars_vecs)

neel_state = sum(1 << i for i in range(0, L, 2))
index_neel = basis[1][neel_state][0]
neel_proj = np.abs(eigenvectors_W[index_neel])**2

eigenvalues_W_proj = np.zeros(len(eigenvalues_W))
for i in range(len(eigenvalues_W)):
  eigenvalues_W_proj[i] = np.sum(np.abs(eigenvectors_W[:,i] @ scars_vecs)**2)

# dif = np.max(np.abs(W - W.T))
# print("Max difference from Hermitian:", dif)

plt.plot(np.real(eigenvalues_W), eigenvalues_W_proj, 'bo')
for i in index_steady:
  plt.plot(np.real(eigenvalues_W[i]), eigenvalues_W_proj[i], 'r*', markersize=12)
plt.grid()
plt.title("Overlap with scar subspace")
plt.show()


plt.matshow(np.real(W), cmap='viridis')
plt.colorbar()
plt.title("Transition matrix |W|")
plt.show()


plt.plot(np.real(eigenvalues_W), np.imag(eigenvalues_W), 'bo')
plt.title("Eigenvalues of W")
plt.show()


plt.plot(np.real(eigenvalues_W), neel_proj, 'bo')
for i in index_steady:
  plt.plot(np.real(eigenvalues_W[i]), eigenvalues_W_proj[i], 'r*', markersize=12)
plt.title("Overlap with Neel state")
# plt.plot(scars_vals, np.abs(scars_vecs[index_neel])**2, 'ro')
plt.show()


# print(W)
