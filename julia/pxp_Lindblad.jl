using ITensors
using ITensorMPS # NOW REQUIRED: Contains MPS, MPO, dmrg, and tdvp

"""
Constructs the MPS representing the Identity operator |I>>.
This state enforces delta functions between Ket (2j-1) and Bra (2j) indices.
"""
function build_identity_mps(sites)
    # The identity state is a product of dimers: sum_s |s>_k |s>_b
    # We build it manually as a bond-dimension 1 MPS.
    M = MPS(sites)
    N_phys = div(length(sites), 2)
    
    for j in 1:N_phys
        s_ket = sites[2*j-1]
        s_bra = sites[2*j]
        
        # We need a dummy link index to connect Ket and Bra inside the pair
        # dim=2 because the physical index dim is 2
        l = Index(2, "Link,Pair$j")
        
        # Tensor for Ket site (2j-1): Delta(s_ket, l)
        T_ket = ITensor(s_ket, l)
        for val in 1:dim(s_ket)
             T_ket[s_ket=>val, l=>val] = 1.0
        end
        M[2*j-1] = T_ket
        
        # Tensor for Bra site (2j): Delta(l, s_bra)
        T_bra = ITensor(l, s_bra)
        for val in 1:dim(s_bra)
             T_bra[l=>val, s_bra=>val] = 1.0
        end
        M[2*j] = T_bra
    end
    return M
end

# -----------------------------------------------------------------------------
# 1. Model Definition (Liouvillian MPO)
# -----------------------------------------------------------------------------

# Define the Projector P = |dn><dn| = [0 0; 0 1]
ITensors.op(::OpName"P", ::SiteType"S=1/2") = [0 0; 0 1]
# Define ProjUp = |up><up| = [1 0; 0 0] (for measurement)
ITensors.op(::OpName"ProjUp", ::SiteType"S=1/2") = [1 0; 0 0]

function build_omegapxp_liouvillian(N::Int, Omega::Float64, 
                                    gamma_plus::Float64, gamma_minus::Float64)
  # Doubled Hilbert space for vectorized density matrix
  # Physical site j -> Ket (2j-1), Bra (2j)
  sites = siteinds("S=1/2", 2*N; conserve_qns=false)
  
  os = OpSum()

  # --- Hamiltonian Dynamics: -i(H_ket - H_bra) ---
  # H = Omega * sum_j P_{j-1} X_j P_{j+1}
  for j in 1:N
    j_left, j_right = j - 1, j + 1
    
    # 1. Ket term (-i H)
    # We must interleave operator names and indices: [Op, Idx, Op, Idx...]
    term_ket = Any[-1.0im * Omega]
    if j_left >= 1   push!(term_ket, "P", 2*j_left - 1) end
    push!(term_ket, "Sx", 2*j - 1)
    if j_right <= N  push!(term_ket, "P", 2*j_right - 1) end
    add!(os, term_ket...)

    # 2. Bra term (+i H^T) -> H is real symmetric, so H^T=H
    term_bra = Any[+1.0im * Omega]
    if j_left >= 1   push!(term_bra, "P", 2*j_left) end
    push!(term_bra, "Sx", 2*j)
    if j_right <= N  push!(term_bra, "P", 2*j_right) end
    add!(os, term_bra...)
  end

  # --- Dissipative Dynamics ---
  # Helper for Lindblad terms
  function add_dissipator!(os, gamma, op_name, j)
    if gamma == 0.0 return end
    j_left, j_right = j - 1, j + 1
    
    # 1. Jump Term: gamma * (L ⊗ L*)
    # L = P op P
    jump_ops = []
    if j_left >= 1   push!(jump_ops, ("P", j_left)) end
    push!(jump_ops, (op_name, j))
    if j_right <= N  push!(jump_ops, ("P", j_right)) end
    
    # Interleave args: [gamma, Op, KetIdx, Op, KetIdx..., Op, BraIdx...]
    # Wait: L ⊗ L* means L acts on Ket, L* acts on Bra.
    # We must list all Ket ops then all Bra ops? 
    # NO. OpSum simply takes a list of (Op, Site) pairs. The order of sites doesn't matter 
    # as long as the pairs are correct.
    
    term_args = Any[gamma] 
    # Add L on Ket
    for (op, k) in jump_ops
        push!(term_args, op, 2*k - 1) 
    end
    # Add L* on Bra
    for (op, k) in jump_ops
        push!(term_args, op, 2*k)     
    end
    add!(os, term_args...)

    # 2. Anti-commutator Terms: -0.5*gamma * {L†L, rho}
    dens_op = (op_name == "S+") ? "P" : "ProjUp"
    
    # Ket side: -0.5*gamma * L†L
    ac_ket = Any[-0.5 * gamma]
    if j_left >= 1   push!(ac_ket, "P", 2*j_left - 1) end
    push!(ac_ket, dens_op, 2*j - 1)
    if j_right <= N  push!(ac_ket, "P", 2*j_right - 1) end
    add!(os, ac_ket...)

    # Bra side: -0.5*gamma * (L†L)^T
    ac_bra = Any[-0.5 * gamma]
    if j_left >= 1   push!(ac_bra, "P", 2*j_left) end
    push!(ac_bra, dens_op, 2*j)
    if j_right <= N  push!(ac_bra, "P", 2*j_right) end
    add!(os, ac_bra...)
  end

  for j in 1:N
     add_dissipator!(os, gamma_plus, "S+", j)
     add_dissipator!(os, gamma_minus, "S-", j)
  end

  return MPO(os, sites)
end

# -----------------------------------------------------------------------------
# 2. Setup and Initialization
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Setup and Initialization
# -----------------------------------------------------------------------------

N = 10
Omega = 1.0
gamma_plus = 0.0
gamma_minus = 1.0
T_total = 5.0
dt = 0.1

# Build Liouvillian
L_mpo = build_omegapxp_liouvillian(N, Omega, gamma_plus, gamma_minus)
H_eff = im * L_mpo

# CORRECTED: Extract only the unprimed site indices from the MPO
# siteinds(MPO) gives [[s1, s1'], [s2, s2'], ...]. We want [s1, s2, ...].
sites = [firstind(L_mpo[j]; tags="Site", plev=0) for j in 1:length(L_mpo)]

# Create Initial State: All Atoms in Ground State |00...0>
# Maps to |Dn> in S=1/2 basis.
psi = MPS(sites, "Dn")

# -----------------------------------------------------------------------------
# 3. Measurement Function
# -----------------------------------------------------------------------------
function measure_rydberg_density(psi::MPS, N::Int)
    sites = siteinds(psi)
    
    # 1. Build the Identity MPS (Trace state)
    psi_trace = build_identity_mps(sites)
    
    # 2. Compute Norm = Tr(rho)
    # This contracts the full network, handling all links correctly.
    tr_rho = inner(psi_trace, psi)
    
    # 3. Compute Density
    # We want Tr( (Sum_j n_j) * rho )
    # This is equivalent to inner(psi_trace, MPO_of_N * psi)
    # OR: efficiently modify the trace state locally.
    
    total_n = 0.0
    
    # We iterate sites and modify the 'psi_trace' locally to measure n_j
    # n_j = ProjUp = |2><2| (in 1-based indexing for S=1/2)
    
    for j in 1:N
        # Copy the trace MPS to avoid corrupting it for other sites
        phi = copy(psi_trace)
        
        # Apply N_j to the Bra site of the Trace MPS? 
        # Logic: <I| n_j | rho> = <I| (n_j x I) |rho>
        # The Identity state implies ket=bra. n_j acts on ket.
        # Effectively, we just change the weight of the |up, up> component in the pair.
        
        # Access the tensors for pair j
        # T_ket connects physical ket to internal link 'l'
        # T_bra connects internal link 'l' to physical bra
        # We want to insert 'ProjUp' operator.
        # The Identity pair is: |d>|d> + |u>|u>.
        # We want: 0*|d>|d> + 1*|u>|u>. (Only count Up state)
        
        # We can achieve this by applying ProjUp to the Ket tensor of phi
        s_ket = sites[2*j-1]
        op_n = op("ProjUp", s_ket)
        
        # Update the tensor at 2j-1
        phi[2*j-1] = phi[2*j-1] * op_n
        # Note: This creates a prime index. Map it back to unprimed.
        phi[2*j-1] = replaceind(phi[2*j-1], s_ket', s_ket)
        
        # Compute overlap
        tr_n_rho = inner(phi, psi)
        
        total_n += (tr_n_rho / tr_rho)
    end
    
    return real(total_n / N)
end

# -----------------------------------------------------------------------------
# 4. Time Evolution (TDVP)
# -----------------------------------------------------------------------------
println("Starting evolution with ITensorMPS...")

# Correct: Explicitly build sites vector for initial state
sites_vec = [firstind(L_mpo[j]; tags="Site", plev=0) for j in 1:length(L_mpo)]
psi = MPS(sites_vec, "Dn")

t_curr = 0.0
steps = Int(T_total / dt)

for step in 1:steps
    global psi = tdvp(H_eff, dt, psi; 
                      cutoff=1e-10, 
                      maxdim=100,
                      nsweeps=1,
                      updater_kwargs=(; ishermitian=false, tol=1e-12, krylovdim=30))
    
    global t_curr += dt
    
    # Use the new robust measurement
    dens = measure_rydberg_density(psi, N)
    
    println("t = $(round(t_curr, digits=2)), <n> = $dens")
end