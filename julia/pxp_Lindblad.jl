using ITensors
using ITensorMPS
using Printf

# -----------------------------------------------------------------------------
# 1. Helper Functions: Trace and Measurement
# -----------------------------------------------------------------------------

function trace_mps(rho::MPS, sites)
    Nvec = length(sites)
    result = ITensor(1.0)
    for j in 1:2:Nvec
        delta_tensor = delta(sites[j], sites[j+1])
        result = result * rho[j] * delta_tensor * rho[j+1]
    end
    return scalar(result)
end

"""
    compute_average_occupation(rho, sites, trace_mps)

Computes 1/N * sum_j < n_j >.
It applies n_j (ProjUp) to the Ket site j, then contracts with the Trace MPS.
This calculates Tr( (1/N sum n_j) * rho ).
"""
function compute_average_occupation(rho, sites)
    Nvec = length(sites)
    N = div(Nvec, 2)
    total_occ = 0.0
    
    for j in 1:N
        k_idx = 2*j - 1
        
        # Create a copy of rho to modify
        rho_n = copy(rho)
        
        # Get the occupation operator as an ITensor
        n_op = op("ProjUp", sites[k_idx])
        
        # Apply by contracting with the j-th tensor
        rho_n[k_idx] = noprime(n_op * rho[k_idx])
        
        # Contract with trace MPS
        val = trace_mps(rho_n, sites)
        total_occ += real(val)
    end
    
    return total_occ / N
end

# -----------------------------------------------------------------------------
# 2. Lindbladian Builder
# -----------------------------------------------------------------------------

"""
    build_lindbladian(sites, Omega, gamma)

Constructs the vectorised Lindbladian H_eff = i * L for the OmegaPXP model.
Returns an MPO.
"""
function build_lindbladian(sites, Omega, gamma_plus, gamma_minus)
    Nvec = length(sites)
    N = div(Nvec, 2)
    os = OpSum()
    
    # Helper for vectorized indices: Ket (odd), Bra (even)
    k(j) = 2*j - 1
    b(j) = 2*j
    
    for j in 2:N-1

        ops_k = []
        push!(ops_k, "ProjDn", k(j-1))
        push!(ops_k, "Sx", k(j))
        push!(ops_k, "ProjDn", k(j+1))
        
        # Add to OpSum (splatting the operators)
        os += -1.0im * Omega, ops_k...

        # 2. Term +i (I x H^T) acting on Bras
        # H is real symmetric, so H^T = H.
        # Coeff: +i * Omega
        
        ops_b = []
        push!(ops_b, "ProjDn", b(j-1))
        push!(ops_b, "Sx", b(j))
        push!(ops_b, "ProjDn", b(j+1))
        
        os += +1.0im * Omega, ops_b...
        
        # --- Dissipator Terms ---
        # L = sqrt(gamma) * P_{j-1} * Sigma^+_j * P_{j+1}
        # Sigma^+ = "S+" (raises 0->1)
        
        if gamma_plus > 0
            # 3. Jump Term: + gamma * (L x L*)
            # L acts on Ket, L* acts on Bra. (S+ is real, so S+^* = S+)
            # Coeff: +gamma (Real in L, so +i*gamma in iL ???)
            # WAIT: We are building i*L directly or just L?
            # Standard: tdvp solves d/dt = -i K. 
            # Lindblad: d/dt = L.
            # So we want -i K = L  => K = i L.
            # Real terms in L become Imaginary in K.
            # L_jump = gamma (L x L_bar). In K: +i * gamma
            
            # Ops for Jump
            ops_jump = []
            push!(ops_jump, "ProjDn", k(j-1))
            push!(ops_jump, "ProjDn", b(j-1))
            push!(ops_jump, "S+", k(j))
            push!(ops_jump, "S+", b(j))
            push!(ops_jump, "ProjDn", k(j+1))
            push!(ops_jump, "ProjDn", b(j+1))
            
            os += gamma_plus, ops_jump...
            
            # 4. Decay Terms: -1/2 gamma {L^dag L, rho}
            # L^dag L = (P S- P) (P S+ P) = P S- S+ P = P * ProjDn * P
            # On Ket: -1/2 gamma (L^dag L x I) -> Coeff in K: -i * 0.5 * gamma
            # On Bra: -1/2 gamma (I x (L^dag L)^T) -> Coeff in K: -i * 0.5 * gamma
            
            # Op for L^dag L is P_{j-1} ProjDn_j P_{j+1}
            # Note: ProjDn is |0><0|. S- S+ = |0><1|1><0| = |0><0|. Correct.
            
            # Ket Decay
            ops_dk = []
            push!(ops_dk, "ProjDn", k(j-1))
            push!(ops_dk, "ProjDn", k(j))
            push!(ops_dk, "ProjDn", k(j+1))
            
            os += -0.5 * gamma_plus, ops_dk...
            
            # Bra Decay
            ops_db = []
            push!(ops_db, "ProjDn", b(j-1))
            push!(ops_db, "ProjDn", b(j))
            push!(ops_db, "ProjDn", b(j+1))
            
            os += -0.5 * gamma_plus, ops_db...
        end

        if gamma_minus > 0
            # 3. Jump Term: + gamma * (L x L*)
            # L acts on Ket, L* acts on Bra. (S- is real, so S-^* = S-)
            # Coeff: +gamma (Real in L, so +i*gamma in iL ???)
            # WAIT: We are building i*L directly or just L?
            # Standard: tdvp solves d/dt = -i K. 
            # Lindblad: d/dt = L.
            # So we want -i K = L  => K = i L.
            # Real terms in L become Imaginary in K.
            # L_jump = gamma (L x L_bar). In K: +i * gamma
            
            # Ops for Jump
            ops_jump = []
            push!(ops_jump, "ProjDn", k(j-1))
            push!(ops_jump, "ProjDn", b(j-1))
            push!(ops_jump, "S-", k(j))
            push!(ops_jump, "S-", b(j))
            push!(ops_jump, "ProjDn", k(j+1))
            push!(ops_jump, "ProjDn", b(j+1))
            
            os += gamma_minus, ops_jump...
            
            # 4. Decay Terms: -1/2 gamma {L^dag L, rho}
            # L^dag L = (P S- P) (P S+ P) = P S- S+ P = P * ProjDn * P
            # On Ket: -1/2 gamma (L^dag L x I) -> Coeff in K: -i * 0.5 * gamma
            # On Bra: -1/2 gamma (I x (L^dag L)^T) -> Coeff in K: -i * 0.5 * gamma
            
            # Op for L^dag L is P_{j-1} ProjDn_j P_{j+1}
            # Note: ProjDn is |0><0|. S- S+ = |0><1|1><0| = |0><0|. Correct.
            
            # Ket Decay
            ops_dk = []
            push!(ops_dk, "ProjDn", k(j-1))
            push!(ops_dk, "ProjUp", k(j))
            push!(ops_dk, "ProjDn", k(j+1))
            
            os += -0.5 * gamma_minus, ops_dk...
            
            # Bra Decay
            ops_db = []
            push!(ops_db, "ProjDn", b(j-1))
            push!(ops_db, "ProjUp", b(j))
            push!(ops_db, "ProjDn", b(j+1))
            
            os += -0.5 * gamma_minus, ops_db...
        end
    end
    
    return MPO(os, sites)
end

# -----------------------------------------------------------------------------
# 3. Time Evolution Function
# -----------------------------------------------------------------------------

"""
    run_evolution(rho_init, sites, H_solver, dt, t_total)

Evolves the state, normalizes, and measures occupation at each step.
Returns arrays of time points and average occupations.
"""
function run_evolution(rho_init, sites, H_solver, dt, t_total)
    rho = copy(rho_init)
    
    # # Pre-construct the trace operator
    # tr_mps = create_trace_mps(sites)
    
    times = Float64[]
    occupations = Float64[]
    
    # Time loop
    # We perform steps: Normalize -> Measure -> Evolve
    for t in 0:dt:t_total
        # 1. Normalize <1|rho> = 1
        z = trace_mps(rho, sites)
        # Avoid division by zero if dynamics are weird, though shouldn't happen
        if abs(z) > 1e-14
            rho /= z
        end
        
        # 2. Measure Average Occupation
        avg_occ = compute_average_occupation(rho, sites)
        
        # Store results
        push!(times, t)
        push!(occupations, avg_occ)
        
        println(@sprintf("Time: %.3f | Trace: %.5f | Imag Trace: %.5f | Avg Occ: %.5f", t, real(z), imag(z), avg_occ))
        
        # 3. Evolve one step
        # Note: We use ishermitian=false for Lindbladians
        rho = tdvp(H_solver, dt, rho;  
                   cutoff=1e-12,
                   nsweeps=2,
                   updater_kwargs=(; ishermitian=false, tol=1e-12, krylovdim=10))
    end
    
    return times, occupations
end

# -----------------------------------------------------------------------------
# 4. Main Function
# -----------------------------------------------------------------------------

function main()
    # A. Parameters
    N = 20                # Physical sites
    Omega = 1.0           # Rabi frequency
    gamma_plus = 0.2           # Pumping strength
    gamma_minus = 0.2          # Sraining strength

    dt = 0.1             # Time step
    t_total = 5.0         # Final time
    output_file = "occupation_dynamics.txt"

    println("--- Simulating Dissipative OmegaPXP Model ---")
    println("Sites: $N, Omega: $Omega, Gamma+: $gamma_plus, Gamma-: $gamma_minus")

    # B. Define Sites (Doubled space)
    # Odd indices = Ket, Even indices = Bra
    sites = siteinds("S=1/2", 2*N)

    # C. Build Lindbladian
    println("Building Lindbladian...")
    H_solver = build_lindbladian(sites, Omega, gamma_plus, gamma_minus)

    # D. Initial State: All Ground State |00...0>
    # In PXP, Ground = Down (|0>).
    # Vectorized |0><0| -> |0>_k |0>_b
    init_state_str = [isodd(i) ? "Dn" : "Dn" for i in 1:2*N]
    rho_init = MPS(sites, init_state_str)

    # E. Run Evolution
    println("Starting Evolution...")
    times, occs = run_evolution(rho_init, sites, H_solver, dt, t_total)

    # F. Save Results
    println("Saving results to $output_file...")
    open(output_file, "w") do io
        write(io, "Time\tAvgOccupation\n")
        for (t, o) in zip(times, occs)
            write(io, "$t\t$o\n")
        end
    end
    println("Done.")
end

# Run the simulation
main()