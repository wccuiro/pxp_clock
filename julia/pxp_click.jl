using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

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
# 2. Vectorized Lindbladian Builder
# -----------------------------------------------------------------------------

function build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    N = div(length(sites), 2)
    os = OpSum()
    
    k(j) = 2 * mod1(j, N) - 1
    b(j) = 2 * mod1(j, N)
    
    for j in 1:N
        # Hamiltonian: -i[H, ρ] (Note: Sx = X/2, so coefficient is 2.0im)
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
# 3. Time Evolution & Krylov Generation
# -----------------------------------------------------------------------------

function generate_krylov_sequence(rho_init, L_mpo, sites, z::ComplexF64, num_steps::Int, maxdim::Int)
    states = MPS[rho_init]
    rho = copy(rho_init)
    
    # Map the target contour z to ITensor's internal Schrödinger time.
    # U = exp(-i * tau * L). We want U = exp(z * L). Therefore tau = 1im * z.
    tau = 1im * z 
    
    for step in 1:num_steps
        # Increased krylovdim and nsweeps to prevent NaN Arnoldi instabilities
        rho = tdvp(L_mpo, tau, rho; 
                   maxdim=maxdim, 
                   cutoff=1e-10,
                   nsweeps=2,
                   updater_kwargs=(; ishermitian=false, tol=1e-10, krylovdim=30))
        
        # Normalize trace to prevent underflow
        tr_val = trace_mps(rho, sites)
        if abs(tr_val) > 1e-14
            rho ./= tr_val
        end
        push!(states, rho)
    end
    return states
end

# -----------------------------------------------------------------------------
# 4. CLIK-MPS Execution
# -----------------------------------------------------------------------------

function compute_clik_spectrum(N::Int, Omega::Float64, gamma_plus::Float64, gamma_minus::Float64; 
                               dt::Float64=0.05, alpha::Float64=0.05, num_steps::Int=20, maxdim::Int=200)
    
    println("--- Starting CLIK-MPS via TDVP ---")
    sites = siteinds("S=1/2", 2*N)
    L_mpo = build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    
    # Start from an excited-like random state
    rho_init = random_mps(sites; linkdims=10)
    rho_init ./= trace_mps(rho_init, sites)
    
    # Sequence 1: Real-time evolution (α = 0)
    println("Generating α=0 sequence (Steady State target)...")
    z_0 = dt * exp(-1im * 0.0) 
    krylov_0 = generate_krylov_sequence(rho_init, L_mpo, sites, z_0, num_steps, maxdim)
    
    # Sequence 2: Complex-time evolution (α > 0)
    println("Generating α=$(alpha) sequence (Excited States target)...")
    z_alpha = dt * exp(-1im * alpha)
    krylov_alpha = generate_krylov_sequence(rho_init, L_mpo, sites, z_alpha, num_steps, maxdim)
    
    # Combine the subspaces: K_0 ⊕ K_α
    krylov_states = vcat(krylov_0, krylov_alpha)
    
    println("Constructing Gram Matrix...")
    K = length(krylov_states)
    M = zeros(ComplexF64, K, K)
    L_mat = zeros(ComplexF64, K, K)
    
    for i in 1:K
        for j in 1:K
            M[i, j] = inner(krylov_states[i], krylov_states[j])
            
            # Using psi' to fix the ITensor inner() match warning
            L_mat[i, j] = inner(krylov_states[i]', L_mpo, krylov_states[j])
        end
    end
    
    println("Diagonalizing Effective Lindbladian...")
    F = eigen(M)
    
    # Increased deflation threshold to 1e-11 to remove spurious degenerate steady states
    keep_idx = findall(x -> abs(x) > 1e-15, F.values)
    
    S_keep = F.values[keep_idx]
    U_keep = F.vectors[:, keep_idx]
    
    X = U_keep * Diagonal(1.0 ./ sqrt.(complex.(S_keep)))
    L_eff = X' * L_mat * X
    
    spectra = eigen(L_eff)
    return spectra.values, spectra.vectors
end

# Execute
N = 6
Omega = 1.0
gamma_plus = 0.2
gamma_minus = 0.001

evals, evecs = compute_clik_spectrum(N, Omega, gamma_plus, gamma_minus; 
                                     dt=0.2, alpha=0.05, num_steps=40)

println("\nTop 5 Eigenvalues (Real Part closest to 0):")
sorted_idx = sortperm(real.(evals), rev=true)
for i in 1:min(5, length(sorted_idx))
    idx = sorted_idx[i]
    @printf("λ_%d = %.6f %+.6fim\n", i, real(evals[idx]), imag(evals[idx]))
end