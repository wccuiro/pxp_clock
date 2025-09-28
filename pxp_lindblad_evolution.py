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
###################### GENERATION OF THE HAMILTONIAN ########################
#############################################################################

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

#############################################################################
###################### GENERATION OF THE DISSIPATOR #########################
#############################################################################

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
            L_minus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p][0], index[state][0]] += np.exp(-1j * 2 * np.pi * k * d / L) * np.sqrt(index[state][1] / index[state_p][1] )
            # print("gamma_plus")          

      D_minus += np.kron(L_minus_i, L_minus_i.conj()) - 0.5 * np.kron(L_minus_i.conj().T @ L_minus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_minus_i.conj().T @ L_minus_i).T)
      D_plus += np.kron(L_plus_i, L_plus_i.conj()) - 0.5 * np.kron(L_plus_i.conj().T @ L_plus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_plus_i.conj().T @ L_plus_i).T)
    
    D = gamma_minus * D_minus + gamma_plus * D_plus
    
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
            L_minus_i [ index[state_p], index[state]] += 1
            # print("gamma_minus")
          else:
            L_plus_i [ index[state_p], index[state]] += 1
            # print("gamma_plus")          

      D_minus += np.kron(L_minus_i, L_minus_i.conj()) - 0.5 * np.kron(L_minus_i.conj().T @ L_minus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_minus_i.conj().T @ L_minus_i).T)
      D_plus += np.kron(L_plus_i, L_plus_i.conj()) - 0.5 * np.kron(L_plus_i.conj().T @ L_plus_i, np.eye(len(states))) - 0.5 * np.kron(np.eye(len(states)), (L_plus_i.conj().T @ L_plus_i).T)
    
    D = gamma_minus * D_minus + gamma_plus * D_plus
    
    return D

#############################################################################
###################### GENERATION OF THE LINDBLADIAN ########################
#############################################################################

def lindblad_evolution(H, D):
  I = np.eye(H.shape[0])
  L = -1j * (np.kron(H, I) - np.kron(I, H.T)) + D
  return L

#############################################################################
###################### GENERATION OF THE TRANSITION MATRIX ##################
#############################################################################

def transition_matrix(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
  W = np.zeros((len(states), len(states)))

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
            W [ index[state_p][0], index[state][0]] += gamma_minus * index[state][1] / index[state_p][1] 
            # print("gamma_minus")
          else:
            W [ index[state_p][0], index[state][0]] += gamma_plus * index[state][1] / index[state_p][1]
            # print("gamma_plus")          
    # np.fill_diagonal(W, -np.sum(W, axis=0))
          W [ index[state][0], index[state][0] ] -= gamma_minus * index[state][1] / index[state_p][1] if state & 1<<i else gamma_plus * index[state][1] / index[state_p][1]
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

#############################################################################
#############################################################################
#############################################################################

def diag_indices(N):
    """Indices in column-stacking vec where |i><i| sits: i + i*N = i*(N+1)."""
    return np.array([i*(N+1) for i in range(N)], dtype=int)

def project_D_onto_diagonal_subspace(D):
    """
    Project superoperator D (shape N^2 x N^2) onto the diagonal operator subspace.

    Returns:
      D_diag_reduced : (N, N)  -- compressed matrix acting on diagonal operators
      idx            : (N,)    -- indices in vec-space corresponding to |i><i|
      D_diag_full    : (N^2, N^2) -- optional full projector result P_diag @ D @ P_diag
    """
    dim = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square.")
    N = int(np.sqrt(dim))
    if N * N != dim:
        raise ValueError("Dimension of D must be a perfect square (N^2).")

    idx = diag_indices(N)
    # full projector (sparse-looking diagonal matrix)
    Pdiag = np.zeros((dim, dim), dtype=D.dtype)
    Pdiag[idx, idx] = 1.0

    D_diag_full = Pdiag @ D @ Pdiag
    D_diag_reduced = D[np.ix_(idx, idx)]

    return D_diag_reduced, idx, D_diag_full

############################################################################
###################### MAIN PROGRAM ########################################
############################################################################


L = 8
PBC = True
k_sector = 0
basis = generation_basis(L, pbc=PBC)

print("Basis size:", len(basis[0])**2)

gamma_plus = 0.1
gamma_minus = 10.0
omega = 0.0

H = Hamiltonian(L, basis[0], basis[1], omega, pbc=PBC, k=k_sector)
D = dissipation(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)
Lind = lindblad_evolution(H, D)

print(np.min(np.abs(D)))

proj_D, idx, full_proj_D = project_D_onto_diagonal_subspace(D)

W = transition_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)

print(np.matrix(proj_D))
print("-----")
print(np.matrix(W))
print("Difference:")
print(np.max(np.abs(W-proj_D)))

eigenvalues_Lind, eigenvectors_Lind = np.linalg.eig(Lind)
eigenvalues_W, eigenvectors_W = np.linalg.eig(W)

# #############################################################################
# ###################### CLEANING EIGENVALUES #################################
# #############################################################################

threshold_eigval = 1e-10

for i in range(len(eigenvalues_Lind)):
  if np.abs(np.real(eigenvalues_Lind[i])) < threshold_eigval:
    eigenvalues_Lind[i] = 1j * np.imag(eigenvalues_Lind[i])
  if np.abs(np.imag(eigenvalues_Lind[i])) < threshold_eigval:
    eigenvalues_Lind[i] = np.real(eigenvalues_Lind[i])

for i in range(len(eigenvalues_W)):
  if np.abs(np.real(eigenvalues_W[i])) < threshold_eigval:
    eigenvalues_W[i] = 0

# # for i in range(len(eigenvalues_W)):
# #   eigenvalues_W[i] = eigenvalues_W[i] * basis[1][basis[0][i]][1]

plt.plot(np.real(eigenvalues_W), np.imag(eigenvalues_W), 'o')
plt.plot(np.real(eigenvalues_Lind), np.imag(eigenvalues_Lind), 'x')
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()
plt.close()

l=0
for i in range(len(eigenvalues_Lind)):
  rho = eigenvectors_Lind[:,i].reshape((H.shape[0], H.shape[0]), order='C')
  if np.trace(rho) < 1e-10:
    l+=1
print(l)

threshold = 1e-10

j=0
for i in range(eigenvalues_Lind.shape[0]):
  if np.abs(eigenvalues_Lind[i]) < threshold:
    vec_steady_state = eigenvectors_Lind[:,i]
    j+=1
  if j > 1:
    print("More than one steady state!")
    break

steady_state = vec_steady_state.reshape((H.shape[0], H.shape[0]), order='C')
steady_state = 0.5*(steady_state + steady_state.conj().T)
tr = np.trace(steady_state)
if np.abs(tr) < 1e-16:
    raise RuntimeError("Trace numerically zero; cannot normalize")
steady_state = steady_state / tr

print(steady_state.shape)

neel_state = sum(1 << i for i in range(0, L, 2))
index_neel = basis[1][neel_state][0]
print(steady_state[index_neel, index_neel])
print(index_neel)

print("Eigenvalues of Lindbladian:")
print(eigenvalues_Lind)

# print(Lind)

plt.matshow(np.real(Lind), cmap='viridis')
# plt.matshow(H, cmap='viridis')
plt.colorbar()

plt.show()
plt.close()

plt.plot(np.real(eigenvalues_Lind), np.imag(eigenvalues_Lind), 'o')
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()
plt.close()

plt.matshow(np.real(steady_state), cmap='viridis')
# plt.matshow(H, cmap='viridis')
plt.colorbar()

plt.show()
plt.close()

