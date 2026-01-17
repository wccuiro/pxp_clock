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

def generation_basis(L):
  rep_states = fibonacci_basis(L)
  rep_index = {s: i for i,  s in enumerate(rep_states)}
  return rep_states, rep_index

#############################################################################
###################### GENERATION OF THE HAMILTONIAN ########################
#############################################################################

def Hamiltonian(L, states, index, omega):
  
  H = np.zeros((len(states), len(states)), dtype=complex)
  
  for state in states:
    for i in range(L):
      if (state >> ((i-1)%L)) & 1 == 0 and (state >> ((i+1)%L)) & 1 == 0:
          state_p = state ^ 2**i
          H [ index[state], index[state_p] ] += omega

            # print("{:04b} --h-- {:04b}".format(state, state_i_p),i)
  return H       

#############################################################################
###################### GENERATION OF THE DISSIPATOR #########################
#############################################################################

def dissipation(L, states, index, gamma_plus, gamma_minus):
  D_minus = np.zeros((len(states)**2, len(states)**2), dtype=complex)
  D_plus = np.zeros((len(states)**2, len(states)**2), dtype=complex)

  for i in range(L):

    L_minus_i = np.zeros((len(states), len(states)), dtype=complex)
    L_plus_i = np.zeros((len(states), len(states)), dtype=complex)

    for state in states:
      if ((state >> ((i-1)%L)) & 1) == 0 and ((state >> ((i+1)%L)) & 1) == 0:
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

  for state in states:
    for i in range(1,L-1):
      if (state >> ((i-1)%L)) & 1 == 0 and (state >> ((i+1)%L)) & 1 == 0:
        state_p = state ^ 2**i
        if state & 1<<i:
          W [ index[state_p], index[state]] += gamma_minus
        else:
          W [ index[state_p], index[state]] += gamma_plus
        W [ index[state], index[state] ] -= gamma_plus if not (state & 1<<i) else gamma_minus
          # print("{:04b} --h-- {:04b}".format(state, state_i_p),i)
  return W       

#############################################################################
#################### PROJECTION OVER DIAGONAL ###############################
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

#############################################################################
################### DENSITY MATRICES NEEL ###################################
#############################################################################

def rho_neel_state (L, states, index, pbc=False, k=0):
  rho_neel = np.zeros((len(states),len(states)))
  neel_state = sum(1 << i for i in range(0, L, 2))
  if pbc:
    if k == 0 or k == L/2:
      rho_neel[index[neel_state][0], index[neel_state][0]] = 1/2      
  else:
    rho_neel[index[neel_state], index[neel_state]] = 1
  
  return rho_neel

def rho_neel_state_T (L, states, index, pbc=False, k=0):
  rho_neel = np.zeros((len(states),len(states)))
  neel_state = sum(1 << i for i in range(0, L, 2))
  neel_state_T = 2 * neel_state
  if pbc:
    if k == 0 or k == L/2:
      rho_neel[index[neel_state][0], index[neel_state][0]] = 1/2      
  else:
    rho_neel[index[neel_state_T], index[neel_state_T]] = 1
  
  return rho_neel

def rho_neel_state_sup (L, states, index, pbc=False, k=0):
  rho_neel = np.zeros((len(states),len(states)))
  neel_state = sum(1 << i for i in range(0, L, 2))
  neel_state_T = 2 * neel_state
  if pbc:
    if k == 0 or k == L/2:
      rho_neel[index[neel_state][0], index[neel_state][0]] = 1/4 * np.abs(1 + np.exp(2j * np.pi * k / L))**2      
  else:
    rho_neel[index[neel_state], index[neel_state]] = 1/2
    rho_neel[index[neel_state_T], index[neel_state_T]] = 1/2
    rho_neel[index[neel_state_T], index[neel_state]] = 1/2
    rho_neel[index[neel_state], index[neel_state_T]] = 1/2
  return rho_neel

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
######################### MAGNETIZATION IN TIME #############################
#############################################################################

def magnetization_in_time (sup_L, rho_init, S_z, t, dt):
  
  vec_rho_init =  rho_init.flatten(order='C')
  
  steps = int(t/dt)
  time_array = np.linspace(0, t, steps)
  
  magnetization = np.zeros(steps)
  
  I = np.eye(len(vec_rho_init))
  
  vec_rho = vec_rho_init
  for i in range(steps):
    vec_rho = ( I + dt * sup_L) @ vec_rho
    rho = vec_rho.reshape((rho_init.shape[0], rho_init.shape[0]), order='C')
    rho_norm = rho / np.trace(rho)
    magnetization [i] = np.real(np.trace(rho_norm @ S_z))
    vec_rho = rho_norm.flatten(order='C')
  
  return time_array, magnetization

#############################################################################
############################ HEAT CURRENTS ##################################
#############################################################################

def dissipators(L, states, index, gamma_plus, gamma_minus, pbc=False, k=0):
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
    
    return gamma_minus * D_minus, gamma_plus * D_plus

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
    
    return gamma_minus * D_minus, gamma_plus * D_plus


def heat_current(rho, sup_p_D, sup_m_D,H):
  
  vec_rho=  rho.flatten(order='C')
    
  vec_rho_p = sup_p_D @ vec_rho
  vec_rho_m = sup_m_D @ vec_rho

  rho_p = vec_rho_p.reshape((rho.shape[0], rho.shape[0]), order='C')
  rho_m = vec_rho_m.reshape((rho.shape[0], rho.shape[0]), order='C')

  current_p = np.real(np.trace(rho_p @ H))
  current_m = np.real(np.trace(rho_m @ H))
  
  return current_p, current_m


############################################################################
############################################################################
###################### MAIN PROGRAM ########################################
############################################################################
############################################################################

############################################################################
###################### SETTING MATRICES ####################################
############################################################################

L = 8
basis = generation_basis(L)

omega = 1.0
gamma_plus = 0.1
gamma_minus = 2.0

H = Hamiltonian(L, basis[0], basis[1], omega)
D = dissipation(L, basis[0], basis[1], gamma_plus, gamma_minus)
Lind = lindblad_evolution(H, D)

eigvals_Lind, eigvecs_Lind = np.linalg.eig(Lind)

# plt.plot(np.real(eigvals_Lind), np.imag(eigvals_Lind), 'o')
# plt.show()
for i in range(len(eigvals_Lind)):
  # print(str(eigvals_Lind[i].real)+','+str(eigvals_Lind[i].imag))
  if np.abs(eigvals_Lind[i]) < 1e-10:
    print("Steady state eigenvalue:", eigvals_Lind[i])
    steady_state = eigvecs_Lind[:,i].reshape((H.shape[0], H.shape[0]), order='C')
    steady_state = 0.5*(steady_state + steady_state.conj().T)
    steady_state = steady_state / np.trace(steady_state)
    print("Steady state trace: ", np.trace(steady_state))
    magnetization = np.real(np.trace(steady_state @ magnetization (L, basis[0], basis[1])))
    print("Steady state magnetization per site: ", magnetization)
    print("Steady state found at index ", i)

# W = transition_matrix(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)

# proj_D, idx, full_proj_D = project_D_onto_diagonal_subspace(D)

# ############################################################################
# ###################### SANITY CHECKS PRINTS ################################
# ############################################################################
# print("Basis H size:", len(basis[0]))
# print("Basis L = H x H size:", len(basis[0])**2)


# # print(np.min(np.abs(D)))

# # print(np.matrix(proj_D))
# # print("-----")
# # print(np.matrix(W))
# print("Max Difference between W and PLP:", np.max(np.abs(W-proj_D)))

# ##############################################################################
# ########################## EIGENVALUES CLEAN #################################
# ##############################################################################

# eigenvalues_proj_D, eigenvectors_proj_D = np.linalg.eig(proj_D)

# eigenvalues_Lind, eigenvectors_Lind = np.linalg.eig(Lind)
# eigenvalues_W, eigenvectors_W = np.linalg.eig(W)


# threshold_eigval = 1e-10

# for i in range(len(eigenvalues_Lind)):
#   if np.abs(np.real(eigenvalues_Lind[i])) < threshold_eigval:
#     eigenvalues_Lind[i] = 1j * np.imag(eigenvalues_Lind[i])
#   if np.abs(np.imag(eigenvalues_Lind[i])) < threshold_eigval:
#     eigenvalues_Lind[i] = np.real(eigenvalues_Lind[i])

# for i in range(len(eigenvalues_W)):
#   if np.abs(np.real(eigenvalues_W[i])) < threshold_eigval:
#     eigenvalues_W[i] = 0

# for i in range(len(eigenvalues_proj_D)):
#   if np.abs(np.real(eigenvalues_proj_D[i])) < threshold_eigval:
#     eigenvalues_proj_D[i] = 1j * np.imag(eigenvalues_proj_D[i])

# # # for i in range(len(eigenvalues_W)):
# # #   eigenvalues_W[i] = eigenvalues_W[i] * basis[1][basis[0][i]][1]

# plt.plot(np.real(eigenvalues_W), np.imag(eigenvalues_W), 'bo', markersize=7, label='W')
# plt.plot(np.real(eigenvalues_proj_D), np.imag(eigenvalues_proj_D), 'r*', markersize=5, label='PLP')
# plt.plot(np.real(eigenvalues_Lind), np.imag(eigenvalues_Lind), 'gx', label='L')
# plt.xlabel('Re')
# plt.ylabel('Im')
# plt.title('Eigenvalues of W, PLP, L')
# plt.legend(loc = "upper left")
# plt.grid(True)
# plt.show()
# plt.close()

# ##############################################################################
# ########################## EIGENVECTORS CHECK ################################
# ##############################################################################

# threshold_tr = 1e-7
# l=0
# for i in range(len(eigenvalues_Lind)):
#   rho = eigenvectors_Lind[:,i].reshape((H.shape[0], H.shape[0]), order='C')
#   rho = 0.5*(rho + rho.conj().T)
#   if np.abs(np.trace(rho)) < threshold_tr:
#     l+=1
# print("Number of traceless states with threshold", threshold_tr, ": ",l)

# j=0
# vec_steady_states = np.zeros((len(basis[0])**2,len(basis[0])**2-l), dtype=complex) 
# for i in range(eigenvalues_Lind.shape[0]):
#   if np.abs(eigenvalues_Lind[i]) == 0:
#     vec_steady_states[:,j] = eigenvectors_Lind[:,i]
#     j+=1

# print("Number of steady states with threshold", threshold_eigval, ": ",len(basis[0])**2-l)

# ##############################################################################
# ################### NORMALIZATION STEADY STATES ##############################
# ##############################################################################

# steady_states = np.zeros((len(basis[0])**2-l, len(basis[0]), len(basis[0])), dtype=complex)
# for i in range(len(basis[0])**2-l):
#   steady_state = vec_steady_states[:,i].reshape((H.shape[0], H.shape[0]), order='C')
#   steady_state = 0.5*(steady_state + steady_state.conj().T)
#   tr = np.trace(steady_state)
#   if np.abs(tr) < 1e-16:
#       raise RuntimeError("Trace numerically zero; cannot normalize")
#   steady_states[i] = steady_state / tr

# print(steady_states.shape[0], " steady states normalized")

# for steady_state in steady_states:
#   if PBC:  
#     neel_state = sum(1 << i for i in range(0, L, 2))
#     index_neel = basis[1][neel_state][0]
#     print("The overlap of Neel representative with the steady state is: ", np.abs(steady_state[index_neel, index_neel]))

#   else:
#     neel_state = sum(1 << i for i in range(0, L, 2))
#     neel_state_T = 2 * neel_state
#     index_neel = basis[1][neel_state]
#     index_neel_T = basis[1][neel_state_T]
#     print("The overlap of Neel state with the steady state is: ", np.abs(steady_state[index_neel, index_neel]))
#     print("The overlap of T-Neel state with the steady state is: ", np.abs(steady_state[index_neel_T, index_neel_T]))
#     print("The overlap with the coherence Neel and T-Neel is: ", steady_state[index_neel, index_neel_T], steady_state[index_neel_T, index_neel])

# # print("Eigenvalues of Lindbladian:")
# # print(eigenvalues_Lind)

# # print(Lind)

# # plt.matshow(np.real(Lind), cmap='viridis')
# # # plt.matshow(H, cmap='viridis')
# # plt.colorbar()

# # plt.show()
# # plt.close()

# plt.plot(np.real(eigenvalues_Lind), np.imag(eigenvalues_Lind), 'o')
# plt.xlabel('Re')
# plt.ylabel('Im')
# plt.title('Eigenvalues of Lindbladian')
# plt.grid(True)
# plt.show()
# plt.close()


# for steady_state in steady_states:
#   plt.matshow(np.abs(steady_state), cmap='viridis')
#   plt.title("Steady state")
#   # plt.matshow(H, cmap='viridis')
#   plt.colorbar()

#   plt.show()
#   plt.close()

# ##############################################################################
# ########################## HEAT CURRENTS IN TIME #############################
# ##############################################################################

# D_plus, D_minus = dissipators(L, basis[0], basis[1], gamma_plus, gamma_minus, pbc=PBC, k=k_sector)
# for steady_state in steady_states:
#   heat_current_p, heat_current_m = heat_current(steady_state, D_plus, D_minus, H)

#   print("Heat current plus in the steady state: ", heat_current_p)
#   print("Heat current plus in the steady state: ", heat_current_m)

# ##############################################################################
# ####################### MAGNETIZATION IN TIME ################################
# ##############################################################################

# S_z = magnetization(L, basis[0], basis[1], pbc=PBC, k=k_sector)

# rho_Neel = rho_neel_state(L, basis[0], basis[1], pbc=PBC, k=k_sector)
# rho_Neel_T = rho_neel_state_T(L, basis[0], basis[1], pbc=PBC, k=k_sector)
# rho_Neel_sup = rho_neel_state_sup(L, basis[0], basis[1], pbc=PBC, k=k_sector)

# t_final = 20
# dt = 0.1

# time_N, magnetization_N = magnetization_in_time (Lind, rho_Neel, S_z, t_final, dt)
# time_N_T, magnetization_N_T = magnetization_in_time (Lind, rho_Neel_T, S_z, t_final, dt)
# time_N_sup, magnetization_N_sup = magnetization_in_time (Lind, rho_Neel_sup, S_z, t_final, dt)

# plt.figure(figsize=(8,5))
# plt.plot(time_N, magnetization_N, label="Neel", color='blue')
# plt.plot(time_N_T, magnetization_N_T, label="Neel_T", color='red')
# plt.plot(time_N_sup, magnetization_N_sup, label="Neel + Neel_T", color='green')
# for steady_state in steady_states:
#   time_steady, magnetization_steady = magnetization_in_time (Lind, steady_state, S_z, t_final, dt)
#   plt.plot(time_steady, magnetization_steady, label="Steady state", color='orange')




# plt.xlabel("Time")
# plt.ylabel("Magnetization per site")
# plt.title("Magnetization dynamics")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.close()



