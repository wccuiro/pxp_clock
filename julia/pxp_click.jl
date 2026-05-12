using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using DelimitedFiles

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------

function trace_mps(rho::MPS, sites)
    result = ITensor(1.0)
    for j in 1:2:length(sites)
        delta_tensor = delta(sites[j], sites[j+1])
        result = result * rho[j] * delta_tensor * rho[j+1]
    end
    return scalar(result)
end

# -----------------------------------------------------------------------------
# 2. Vectorized Lindbladian Builder (PXP Model)
# -----------------------------------------------------------------------------

function build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    N = div(length(sites), 2)
    os = OpSum()
    
    k(j) = 2 * mod1(j, N) - 1
    b(j) = 2 * mod1(j, N)
    
    for j in 1:N
        # Hamiltonian: -i[H, ρ] 
        os += -2.0im * Omega, "ProjDn", k(j-1), "Sx", k(j), "ProjDn", k(j+1)
        os += +2.0im * Omega, "ProjDn", b(j-1), "Sx", b(j), "ProjDn", b(j+1)
        
        # Dissipator γ+
        if gamma_plus > 0
            os += gamma_plus, "ProjDn", k(j-1), "ProjDn", b(j-1), "S+", k(j), "S+", b(j), "ProjDn", k(j+1), "ProjDn", b(j+1)
            os += -0.5 * gamma_plus, "ProjDn", k(j-1), "ProjDn", k(j), "ProjDn", k(j+1)
            os += -0.5 * gamma_plus, "ProjDn", b(j-1), "ProjDn", b(j), "ProjDn", b(j+1)
        end
        
        # Dissipator γ-
        if gamma_minus > 0
            os += gamma_minus, "ProjDn", k(j-1), "ProjDn", b(j-1), "S-", k(j), "S-", b(j), "ProjDn", k(j+1), "ProjDn", b(j+1)
            os += -0.5 * gamma_minus, "ProjDn", k(j-1), "ProjUp", k(j), "ProjDn", k(j+1)
            os += -0.5 * gamma_minus, "ProjDn", b(j-1), "ProjUp", b(j), "ProjDn", b(j+1)
        end
    end
    return MPO(os, sites)
end

# -----------------------------------------------------------------------------
# 3. Warmup Optimization 
# -----------------------------------------------------------------------------

function warmup_initial_states(sites, L_mpo; n_samples=5)
    N = div(length(sites), 2)
    println("Running warmup to optimize initial states...")
    
    best_phys_val = Inf
    best_phys_state = nothing
    best_unphys_val = Inf
    best_unphys_state = nothing
    
    for i in 1:n_samples
        psi_arr = [rand(["Up", "Dn"]) for _ in 1:N]
        phi_arr = [rand(["Up", "Dn"]) for _ in 1:N]
        
        # Vectorized Traceful (Physical) state
        state_phys = String[]
        for j in 1:N
            push!(state_phys, psi_arr[j], psi_arr[j])
        end
        rho_phys = MPS(sites, state_phys)
        
        val_phys = real(inner(rho_phys', L_mpo, rho_phys))
        if val_phys < best_phys_val
            best_phys_val = val_phys
            best_phys_state = rho_phys
        end
        
        # Vectorized Traceless (Unphysical) state
        if psi_arr != phi_arr
            state_phi = String[]
            for j in 1:N
                push!(state_phi, phi_arr[j], phi_arr[j])
            end
            rho_phi = MPS(sites, state_phi)
            
            rho_unphys = +(rho_phys, -1.0 * rho_phi; cutoff=1e-12)
            
            val_unphys = real(inner(rho_unphys', L_mpo, rho_unphys))
            if val_unphys < best_unphys_val
                best_unphys_val = val_unphys
                best_unphys_state = rho_unphys
            end
        end
    end
    
    # TDVP requires strictly orthogonalized MPS
    orthogonalize!(best_phys_state, 1)
    orthogonalize!(best_unphys_state, 1)
    
    return best_phys_state, best_unphys_state
end

# -----------------------------------------------------------------------------
# 4. Time Evolution & Krylov Generation
# -----------------------------------------------------------------------------

function generate_krylov_sequence(rho_init, L_mpo, sites, z::ComplexF64, num_steps::Int, maxdim::Int, is_traceless::Bool)
    states = MPS[rho_init]
    rho = copy(rho_init)
    
    tau = 1im * z 
    
    for step in 1:num_steps
        # nsite=2 is strictly required to escape bond dimension 1 from product states
        rho = tdvp(L_mpo, tau, rho; 
                   nsweeps=1,
                   nsite=2, 
                   maxdim=maxdim, 
                   cutoff=1e-12,
                   updater_kwargs=(; ishermitian=false, tol=1e-12, krylovdim=30))
        
        if !is_traceless
            tr_val = trace_mps(rho, sites)
            if abs(tr_val) > 1e-14
                rho = rho / tr_val
            end
        else
            normalize!(rho)
        end
        push!(states, rho)
    end
    return states
end


function SVD_stabilized(M::AbstractMatrix)
    # Use Julia's LinearAlgebra svd object explicitly for safety
    F = svd(M)
    A = F.U
    L = F.S
    B = F.V 
    
    threshold = 1e-4
    
    # 1. Find the first index p where the singular value drops below the threshold
    p = length(L) + 1
    for i in 2:length(L)
        if L[i] / L[1] < threshold
            p = i
            break # Stop at the first drop
        end
    end

    # 2. Iteratively refine the smaller singular values
    while p <= length(L)
        # Correctly project M onto the subspace of the remaining singular vectors
        X = A[:, p:end]' * M * B[:, p:end]
        
        # Compute SVD of the subproblem
        Fp = svd(X)
        Ap = Fp.U
        Lp = Fp.S
        Bp = Fp.V

        # Find the next drop in the subproblem
        p1 = length(Lp) + 1
        for i in 2:length(Lp)
            if Lp[i] / Lp[1] < threshold
                p1 = i
                break
            end
        end

        # Correctly rotate the entire column vectors
        A[:, p:end] = A[:, p:end] * Ap
        B[:, p:end] = B[:, p:end] * Bp
        L[p:end] = Lp

        # Break to avoid infinite loops if no further threshold drop is found
        if p1 > length(Lp)
            break
        end

        # Map local submatrix index back to the global index
        p = p + p1 - 1
    end

    return A, L, B
end

function L_eff(krylov_states)
    K = length(krylov_states)
    M = zeros(ComplexF64, K, K)
    L_mat = zeros(ComplexF64, K, K)
    
    for i in 1:K
        for j in 1:K
            M[i, j] = inner(krylov_states[i], krylov_states[j])
            # The prime has been removed here to fix the double-prime bug
            L_mat[i, j] = inner(krylov_states[i]', L_mpo, krylov_states[j])
        end
    end

    M = (M + M') / 2.0


    return M, L_mat
end

# -----------------------------------------------------------------------------
# 5. CLIK-MPS Execution
# -----------------------------------------------------------------------------

function compute_clik_spectrum(N::Int, Omega::Float64, gamma_plus::Float64, gamma_minus::Float64; 
                               dt::Float64=0.05, alpha::Float64=0.05, num_steps::Int=20, maxdim::Int=200)
    
    println("--- Starting CLIK-MPS via TDVP ---")
    sites = siteinds("S=1/2", 2*N)
    L_mpo = build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    
    rho_phys, rho_unphys = warmup_initial_states(sites, L_mpo)
    
    println("Generating α=0 sequence (Steady State target)...")
    z_0 = dt * exp(-1im * 0.0) 
    krylov_0 = generate_krylov_sequence(rho_phys, L_mpo, sites, z_0, num_steps, maxdim, false)
    
    println("Generating α=$(alpha) sequence (Excited States target)...")
    z_alpha = dt * exp(-1im * alpha)
    krylov_alpha = generate_krylov_sequence(rho_unphys, L_mpo, sites, z_alpha, num_steps, maxdim, true)
    
    krylov_states = vcat(krylov_0, krylov_alpha)
    
    println("Constructing Gram Matrix...")
    K = length(krylov_states)
    M = zeros(ComplexF64, K, K)
    L_mat = zeros(ComplexF64, K, K)
    
    for i in 1:K
        for j in 1:K
            M[i, j] = inner(krylov_states[i], krylov_states[j])
            # The prime has been removed here to fix the double-prime bug
            L_mat[i, j] = inner(krylov_states[i]', L_mpo, krylov_states[j])
        end
    end
    
    # Enforce strictly Hermitian Gram matrix to prevent SVD numerical explosion
    M = (M + M') / 2.0
    
    println("Diagonalizing Effective Lindbladian via SVD...")
    U, S_vals, V = SVD_stabilized(M)

    S_inv_sqrt_vals = [s > 1e-13 ? 1.0 / sqrt(s) : 0.0 for s in S_vals]
    S_inv_sqrt = Diagonal(S_inv_sqrt_vals)

    X = S_inv_sqrt * U'

    L_eff = X' * L_mat * X

    spectra = eigen(L_eff)
    return spectra.values, spectra.vectors
end
# Execute
N = 8
Omega = 1.0
gamma_plus = 0.5
gamma_minus = 0.25

evals, evecs = compute_clik_spectrum(N, Omega, gamma_plus, gamma_minus; 
                                     dt=0.005, alpha=0.05, num_steps=40)

println("\nTop 5 Eigenvalues (Real Part closest to 0):")
sorted_idx = sortperm(real.(evals), rev=true)
for i in 1:min(5, length(sorted_idx))
    idx = sorted_idx[i]
    @printf("λ_%d = %.6f %+.6fim\n", i, real(evals[idx]), imag(evals[idx]))
end

csv_filename = "pxp_spectrum_julia.csv"
raw_data = hcat(real.(evals), imag.(evals))
writedlm(csv_filename, raw_data, ',')

println("Successfully saved raw spectrum data to '$csv_filename'")