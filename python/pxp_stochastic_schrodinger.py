## Schrodinger Stochastic Equation

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


def Hamiltonian(L, states, index, omega, pbc=False, k=0):
  
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
          H [ index[state_p][0], index[state][0]] += omega * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
          # print("{:04b} --h-- {:04b} -- T -- {:04b}".format(state, state_i_p, state_p),i,d)
          # print("Index:", index[state][0], index[state][1], index[state_p][0], index[state_p][1])
  else:
    for state in states:
      for i in range(1,L-1):
        if (state >> (i-1)) & 1 == 0 and (state >> (i+1)) & 1 == 0:
            state_p = state ^ 2**i
            H [ index[state], index[state_p] ] += omega

            # print("{:04b} --h-- {:04b}".format(state, state_i_p),i)
  return H       

def Hamiltonian_NH(L, states, index, omega, pbc=False, k=0):
  
  H_nh = np.zeros((len(states), len(states)), dtype=complex)
  
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
            L_minus_i [ index[state_p][0], index[state][0]] += gamma_minus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p][0], index[state][0]] += gamma_plus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_plus")          

      H_nh_minus += - 0.5 * (L_minus_i.conj().T @ L_minus_i)
      H_nh_plus += - 0.5 * (L_plus_i.conj().T @ L_plus_i)
    
    H_nh = H_nh_minus + H_nh_plus
    
    return H_nh

  else:
    for i in range(1,L-1):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
      L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> (i-1)) & 1) == 0 and ((state >> (i+1)) & 1) == 0:
          state_p = state ^ (1 << i)
          d = 0
          if state & 1<<i:
            L_minus_i [ index[state_p], index[state]] += gamma_minus
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p], index[state]] += gamma_plus
            # print("gamma_plus")          

      H_nh_minus += - 0.5 * (L_minus_i.conj().T @ L_minus_i)
      H_nh_plus += - 0.5 * (L_plus_i.conj().T @ L_plus_i)
    
    H_nh = H_nh_minus + H_nh_plus
    
    return H_nh
  

def stochastic_evolution(H, states, index, gamma_plus, gamma_minus, dt):  
  pass

def dissipation(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
  D_minus = np.zeros((len(states)**2, len(states)**2), dtype=complex)
  D_plus = np.zeros((len(states)**2, len(states)**2), dtype=complex)

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
            L_minus_i [ index[state_p][0], index[state][0]] += gamma_minus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p][0], index[state][0]] += gamma_plus * np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_plus")          

      D_minus += np.kron(L_minus_i, L_minus_i.conj()) - 0.5 * np.kron(L_minus_i.conj().T @ L_minus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_minus_i.conj().T @ L_minus_i).T)
      D_plus += np.kron(L_plus_i, L_plus_i.conj()) - 0.5 * np.kron(L_plus_i.conj().T @ L_plus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_plus_i.conj().T @ L_plus_i).T)
    
    D = D_minus + D_plus
    
    return D

  else:
    for i in range(1,L-1):

      L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
      L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

      for state in states:
        if ((state >> (i-1)) & 1) == 0 and ((state >> (i+1)) & 1) == 0:
          state_p = state ^ (1 << i)
          d = 0
          if state & 1<<i:
            L_minus_i [ index[state_p], index[state]] += gamma_minus
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p], index[state]] += gamma_plus
            # print("gamma_plus")          

      D_minus += np.kron(L_minus_i, L_minus_i.conj()) - 0.5 * np.kron(L_minus_i.conj().T @ L_minus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_minus_i.conj().T @ L_minus_i).T)
      D_plus += np.kron(L_plus_i, L_plus_i.conj()) - 0.5 * np.kron(L_plus_i.conj().T @ L_plus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_plus_i.conj().T @ L_plus_i).T)
    
    D = D_minus + D_plus
    
    return D
  
def lindblad_evolution(H, D):
  I = np.eye(H.shape[0])
  L = -1j * (np.kron(H, I) - np.kron(I, H.T)) + D
  return L


L = 12
basis = generation_basis(L, pbc=True)

print("Basis size:", len(basis[0])**2)

gamma_plus = 1.0
gamma_minus = 1.0
omega = 0.0

H = Hamiltonian(L, basis[0], basis[1], omega, pbc=True)
D = dissipation(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=True)
Lind = lindblad_evolution(H, D)



