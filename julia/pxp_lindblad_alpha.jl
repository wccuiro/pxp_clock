# 1. Load the Distributed standard library on the master process
using Distributed

# Check if the user remembered to spawn workers
if nprocs() == 1
    println("WARNING: You are only running 1 process.")
    println("To fully exploit your cluster, cancel this and run: julia -p 114 pxp_simulation.jl")
else
    println("Successfully launched master process with $(nworkers()) workers.")
end

# 2. @everywhere block: Everything inside this block is compiled and loaded 
# onto ALL worker processes, not just the master node.
@everywhere begin
    using ITensors
    using ITensorMPS
    using LinearAlgebra
end

# 3. Configure the dense math threads and explicitly kill the nested Strided threads
@everywhere begin
    # Unleash BLAS for Dense Matrix Operations
    BLAS.set_num_threads(Threads.nthreads())

    # Explicitly disable Block-Sparse threading to prevent the warning
    ITensors.disable_threaded_blocksparse()

    # Disable Strided to prevent internal nested threading
    NDTensors.Strided.disable_threads()
    
    worker_id = myid()
    println("Worker $worker_id ready. BLAS Threads: $(Threads.nthreads())")
end

@everywhere begin

    function trace_mps(rho::MPS, sites)
        Nvec = length(sites)
        result = ITensor(1.0)
        for j in 1:2:Nvec
            delta_tensor = delta(sites[j], sites[j+1])
            result = result * rho[j] * delta_tensor * rho[j+1]
        end
        return scalar(result)
    end

    function compute_average_occupation(rho, sites)
        Nvec = length(sites)
        N = div(Nvec, 2)
        total_occ = 0.0
        
        for j in 1:N
            k_idx = 2*j - 1
            rho_n = copy(rho)
            n_op = op("ProjUp", sites[k_idx])
            rho_n[k_idx] = noprime(n_op * rho[k_idx])
            val = trace_mps(rho_n, sites)
            total_occ += real(val)
        end
        return total_occ / N
    end

    function hermitize_mps(rho::MPS, sites; cutoff=1e-10)
        rho_dag = copy(rho)
        for i in 1:length(rho_dag)
            rho_dag[i] = conj(rho_dag[i])
        end
        
        N = div(length(sites), 2)
        for j in 1:N
            k_idx = 2*j - 1
            b_idx = 2*j
            swap_gate = op("Swap", sites[k_idx], sites[b_idx])
            rho_dag = apply(swap_gate, rho_dag; cutoff=cutoff)
        end
        
        rho_herm = +(rho, rho_dag; cutoff=cutoff)
        rho_herm /= 2.0
        return rho_herm
    end

    function build_neel_state(sites)
        N = div(length(sites), 2)
        state_str = String[]
        for j in 1:N
            phys_state = isodd(j) ? "Up" : "Dn"
            push!(state_str, phys_state) 
            push!(state_str, phys_state) 
        end
        return MPS(sites, state_str)
    end

    function compute_neel_overlap(rho::MPS, rho_neel::MPS)
        return real(inner(rho_neel, rho))
    end

    # -------------------------------------------------------------------------
    # Lindbladian TEBD Gates
    # -------------------------------------------------------------------------

    function local_gate(sites, j, Omega, gamma_plus, gamma_minus, alpha, dt)
        k(i) = 2*i - 1
        b(i) = 2*i

        sub_sites = sites[[k(j-1), b(j-1), k(j), b(j), k(j+1), b(j+1)]]
        os = OpSum()

        # Coherent Hamiltonian Term (Retained as full projection per specific jump-operator instructions)
        os += -2im*Omega, "ProjDn",1, "Sx",3, "ProjDn",5
        os += +2im*Omega, "ProjDn",2, "Sx",4, "ProjDn",6

        # --- Partial Projection Coefficients ---
        # P(alpha) = c_up * P_up + 1.0 * P_dn
        c_up = (1.0 - alpha) / (1.0 + alpha)
        c_sq = c_up^2  # Since P(alpha) is not a true projector, P(alpha)^2 != P(alpha)

        ops_P = [("ProjUp", c_up), ("ProjDn", 1.0)]
        ops_Psq = [("ProjUp", c_sq), ("ProjDn", 1.0)]

        # --- Gamma Plus Jump Operators ---
        if gamma_plus > 0
            # L_+ rho L_+^\dagger term (indices 1, 2, 5, 6)
            for op1 in ops_P, op2 in ops_P, op5 in ops_P, op6 in ops_P
                coeff = gamma_plus * op1[2] * op2[2] * op5[2] * op6[2]
                os += coeff, op1[1],1, op2[1],2, "S+",3, "S+",4, op5[1],5, op6[1],6
            end
            # -0.5 L_+^\dagger L_+ rho term (indices 1, 3, 5). Note: S- S+ = ProjDn
            for op1 in ops_Psq, op5 in ops_Psq
                coeff = -0.5 * gamma_plus * op1[2] * op5[2]
                os += coeff, op1[1],1, "ProjDn",3, op5[1],5
            end
            # -0.5 rho L_+^\dagger L_+ term (indices 2, 4, 6)
            for op2 in ops_Psq, op6 in ops_Psq
                coeff = -0.5 * gamma_plus * op2[2] * op6[2]
                os += coeff, op2[1],2, "ProjDn",4, op6[1],6
            end
        end

        # --- Gamma Minus Jump Operators ---
        if gamma_minus > 0
            # L_- rho L_-^\dagger term (indices 1, 2, 5, 6)
            for op1 in ops_P, op2 in ops_P, op5 in ops_P, op6 in ops_P
                coeff = gamma_minus * op1[2] * op2[2] * op5[2] * op6[2]
                os += coeff, op1[1],1, op2[1],2, "S-",3, "S-",4, op5[1],5, op6[1],6
            end
            # -0.5 L_-^\dagger L_- rho term (indices 1, 3, 5). Note: S+ S- = ProjUp
            for op1 in ops_Psq, op5 in ops_Psq
                coeff = -0.5 * gamma_minus * op1[2] * op5[2]
                os += coeff, op1[1],1, "ProjUp",3, op5[1],5
            end
            # -0.5 rho L_-^\dagger L_- term (indices 2, 4, 6)
            for op2 in ops_Psq, op6 in ops_Psq
                coeff = -0.5 * gamma_minus * op2[2] * op6[2]
                os += coeff, op2[1],2, "ProjUp",4, op6[1],6
            end
        end

        L_local = MPO(os, sub_sites)
        gate = contract(L_local)
        gate = exp(dt * gate) 
        return gate
    end

    function tebd_step!(rho, sites, Omega, gamma_plus, gamma_minus, alpha, dt; cutoff=1e-10, maxdim=200)
        N = div(length(sites), 2)

        for j in 2:3:(N-1)
            gate = local_gate(sites, j, Omega, gamma_plus, gamma_minus, alpha, dt)
            rho = apply(gate, rho; cutoff=cutoff, maxdim=maxdim)
        end

        for j in 3:3:(N-1)
            gate = local_gate(sites, j, Omega, gamma_plus, gamma_minus, alpha, dt)
            rho = apply(gate, rho; cutoff=cutoff, maxdim=maxdim)
        end

        for j in 4:3:(N-1)
            gate = local_gate(sites, j, Omega, gamma_plus, gamma_minus, alpha, dt)
            rho = apply(gate, rho; cutoff=cutoff, maxdim=maxdim)
        end

        return rho
    end

    # -------------------------------------------------------------------------
    # Time Evolution Routine
    # -------------------------------------------------------------------------

    function run_evolution_TEBD(rho_init, rho_neel, sites, Omega, gamma_plus, gamma_minus, alpha, dt, t_total)
        rho = copy(rho_init)
        
        times = Float64[]
        occupations = Float64[]
        overlaps = Float64[]
        
        for t in 0:dt:t_total
            rho = hermitize_mps(rho, sites; cutoff=1e-10)

            z = trace_mps(rho, sites)
            if abs(real(z)) > 1e-14
                rho /= real(z)
            end
            
            avg_occ = compute_average_occupation(rho, sites)
            overlap = compute_neel_overlap(rho, rho_neel)
            
            push!(times, t)
            push!(occupations, avg_occ)
            push!(overlaps, overlap)
            
            rho = tebd_step!(rho, sites, Omega, gamma_plus, gamma_minus, alpha, dt; cutoff=1e-10, maxdim=200)
        end
        
        return times, occupations, overlaps
    end
end # End of @everywhere block


# =============================================================================
# Main Execution on Master Node
# =============================================================================

function main()
    # 1. Define your parameter sweep
    gamma_plus_range  = [0.001, 0.2]
    gamma_minus_range = [0.001, 0.2]
    alpha_range       = [0.0, 0.3, 0.6, 1.0] # Added alpha sweep
    
    param_sets = [
        (N=10, Omega=1.0, gp=gp, gm=gm, alpha=a, dt=0.1, t_total=25.0) 
        for gp in gamma_plus_range for gm in gamma_minus_range for a in alpha_range
    ]

    println("--- Starting Distributed Parameter Sweep ---")
    println("Total configurations to run: $(length(param_sets))")

    # 2. Distribute the work using pmap
    pmap(param_sets) do p
        worker_id = myid() 
        println("[Worker $worker_id] Starting: gp=$(p.gp), gm=$(p.gm), alpha=$(p.alpha)")

        # Updated output format to include alpha
        output_file = "data_N$(p.N)_dt$(p.dt)_gp$(p.gp)_gm$(p.gm)_a$(p.alpha).txt"

        sites = siteinds("S=1/2", 2 * p.N)
        rho_neel = build_neel_state(sites)
        rho_init = copy(rho_neel) 

        times, occs, overlaps = run_evolution_TEBD(
            rho_init, rho_neel, sites, p.Omega, p.gp, p.gm, p.alpha, p.dt, p.t_total
        )

        open(output_file, "w") do io
            write(io, "Time\tAvgOccupation\tNeelOverlap\n")
            for (t, o, f) in zip(times, occs, overlaps)
                write(io, "$t\t$o\t$f\n")
            end
        end
        
        println("[Worker $worker_id] Finished: gp=$(p.gp), gm=$(p.gm), alpha=$(p.alpha)")
    end
    
    println("--- All Simulations Complete! ---")
end

@time main()