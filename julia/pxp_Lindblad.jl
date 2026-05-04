using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# 1. Helper Function: Trace Measurement
# -----------------------------------------------------------------------------

"""
    trace_mps(rho::MPS, sites)

Computes the trace of the vectorized density matrix by contracting 
physical (Ket) and auxiliary (Bra) sites.
"""
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

"""
    build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)

Constructs the exact vectorized Lindbladian L for the Omega-PXP model.
"""
function build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    Nvec = length(sites)
    N = div(Nvec, 2)
    os = OpSum()
    
    # Periodic boundary mappings for Ket (odd) and Bra (even)
    k(j) = 2 * mod1(j, N) - 1
    b(j) = 2 * mod1(j, N)
    
    for j in 1:N
        # --- Hamiltonian terms: -i[H, ρ] ---
        # -i H on Ket (X = 2 * Sx)
        os += -2.0im * Omega, "ProjDn", k(j-1), "Sx", k(j), "ProjDn", k(j+1)
        # +i H^T on Bra
        os += +2.0im * Omega, "ProjDn", b(j-1), "Sx", b(j), "ProjDn", b(j+1)
        
        # --- Dissipator Terms for γ+ (Pumping) ---
        if gamma_plus > 0
            # Jump: γ+ L+ ⊗ L+^* 
            os += gamma_plus, "ProjDn", k(j-1), "S+", k(j), "ProjDn", k(j+1), "ProjDn", b(j-1), "S+", b(j), "ProjDn", b(j+1)
            
            # Loss: -1/2 γ+ L+^† L+ ⊗ I (Note: S- S+ = ProjDn)
            os += -0.5 * gamma_plus, "ProjDn", k(j-1), "ProjDn", k(j), "ProjDn", k(j+1)
            os += -0.5 * gamma_plus, "ProjDn", b(j-1), "ProjDn", b(j), "ProjDn", b(j+1)
        end
        
        # --- Dissipator Terms for γ- (Decay) ---
        if gamma_minus > 0
            # Jump: γ- L- ⊗ L-^*
            os += gamma_minus, "ProjDn", k(j-1), "S-", k(j), "ProjDn", k(j+1), "ProjDn", b(j-1), "S-", b(j), "ProjDn", b(j+1)
            
            # Loss: -1/2 γ- L-^† L- ⊗ I (Note: S+ S- = ProjUp)
            os += -0.5 * gamma_minus, "ProjDn", k(j-1), "ProjUp", k(j), "ProjDn", k(j+1)
            os += -0.5 * gamma_minus, "ProjDn", b(j-1), "ProjUp", b(j), "ProjDn", b(j+1)
        end
    end
    
    return MPO(os, sites)
end

# -----------------------------------------------------------------------------
# 3. CLIK-MPS Algorithm
# -----------------------------------------------------------------------------

"""
    compute_clik_spectrum(N, Omega, gamma_plus, gamma_minus; kwargs...)

Executes the CLIK-MPS algorithm to find the low-lying eigenvalues of the Lindbladian.
"""
function compute_clik_spectrum(N::Int, Omega::Float64, gamma_plus::Float64, gamma_minus::Float64; 
                               dt::Float64=0.05, alpha::Float64=0.05, num_steps::Int=40, maxdim::Int=200)
    
    println("--- Starting CLIK-MPS Spectrum Computation ---")
    sites = siteinds("S=1/2", 2*N)
    
    println("Building Lindbladian...")
    L_mpo = build_pxp_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    
    # Initialize with a random MPS to ensure overlap with excited states
    rho_current = random_mps(sites; linkdims=10)
    
    # Normalize trace to 1 using your custom function
    tr_val = trace_mps(rho_current, sites)
    rho_current ./= tr_val
    
    krylov_states = MPS[rho_current]
    
    println("Executing complex-time evolution...")
    for step in 1:num_steps
        # Warmup step: use smaller dt initially for numerical stability
        current_dt = step <= 5 ? 0.005 : dt
        
        # Define complex contour z and map to ITensor's internal -i * tau = z
        z = current_dt * exp(-1im * alpha)
        tau = 1im * z 
        
        # Evolve using non-Hermitian TDVP configuration
        rho_current = tdvp(L_mpo, tau, rho_current; 
                           maxdim=maxdim, 
                           cutoff=1e-10,
                           nsweeps=2,
                           updater_kwargs=(; ishermitian=false, tol=1e-12, krylovdim=10))
        
        # Normalize to prevent underflow/overflow
        tr_val = trace_mps(rho_current, sites)
        if abs(tr_val) > 1e-14
            rho_current ./= tr_val
        end
        
        push!(krylov_states, rho_current)
    end
    
    println("Constructing Gram Matrix...")
    K = length(krylov_states)
    M = zeros(ComplexF64, K, K)
    L_mat = zeros(ComplexF64, K, K)
    
    for i in 1:K
        for j in 1:K
            M[i, j] = inner(krylov_states[i], krylov_states[j])
            
            # Project Lindbladian into Krylov space
            L_rho_j = contract(L_mpo, krylov_states[j]; maxdim=maxdim)
            L_mat[i, j] = inner(krylov_states[i], L_rho_j)
        end
    end
    
    println("Orthogonalizing and Diagonalizing...")
    # Diagonalize Gram Matrix M = U * S * U^†
    F = eigen(M)
    
    # Discard numerically insignificant singular values
    threshold = 1e-13
    keep_idx = findall(x -> abs(x) > threshold, F.values)
    
    S_keep = F.values[keep_idx]
    U_keep = F.vectors[:, keep_idx]
    
    # Construct orthonormal basis transformation X
    X = U_keep * Diagonal(1.0 ./ sqrt.(complex.(S_keep)))
    
    # Diagonalize Effective Lindbladian: L_eff = X^† * L_mat * X
    L_eff = X' * L_mat * X
    spectra = eigen(L_eff)
    
    println("--- Computation Complete ---")
    return spectra.values, spectra.vectors
end

# -----------------------------------------------------------------------------
# 4. Execution
# -----------------------------------------------------------------------------

N = 5
Omega = 1.0
gamma_plus = 0.2
gamma_minus = 0.001

evals, evecs = compute_clik_spectrum(N, Omega, gamma_plus, gamma_minus; 
                                     dt=0.05, alpha=0.05, num_steps=40)

println("\nTop 5 Eigenvalues (Real Part closest to 0):")
sorted_idx = sortperm(real.(evals), rev=true)
for i in 1:min(5, length(sorted_idx))
    idx = sorted_idx[i]
    @printf("λ_%d = %.6f %+.6fim\n", i, real(evals[idx]), imag(evals[idx]))
end