using ITensors
using ITensorMPS
using Random
using Printf
using LinearAlgebra

# FIX: Prevent OpenBLAS from oversubscribing cores. 
# This stops Julia from trying to spawn hundreds of threads and locking up the CPU.
BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

function simulate_pxp_trajectories(;
  # --- 1. System Parameters ---
  L = 12,
  Omega = 1.0,
  gamma_plus  = 0.2,
  gamma_minus = 0.001,
  dt = 1e-1,
  total_time = 25.0,
  num_trajectories = 100,
  output_dir = "../data/trajectoriesTN"
  )

  output_dir = "../data/trajectoriesTN_L$(L)_gamma+$(gamma_plus)_gamma-$(gamma_minus)_dt$(dt)"

  mkpath(output_dir)
  steps = round(Int, total_time / dt)

  println("--- PXP Tensor Network Simulation ---")
  println("Parameters: L=$L, Omega=$Omega, Gamma+=$gamma_plus, Gamma-=$gamma_minus, dt=$dt")
  println("Starting simulation on ", Threads.nthreads(), " threads...")

  # Spin-1/2 mapping: "Up" = Occupied (1), "Dn" = Empty (0)
  s = siteinds("S=1/2", L)

  # ==========================================
  # Pre-Compile Operators (Solves 200% compilation overhead)
  # ==========================================
  
  # 1. TEBD Gates (3-site blocks for PBC)
  gates_1 = ITensor[]
  gates_2 = ITensor[]
  gates_3 = ITensor[]

  for i in 1:L
    prev_i = i == 1 ? L : i - 1
    next_i = i == L ? 1 : i + 1

    h_term = (2.0 * Omega) * op("ProjDn", s[prev_i]) * op("Sx", s[i]) * op("ProjDn", s[next_i]) +
             (-0.5im * gamma_plus) * op("ProjDn", s[prev_i]) * op("ProjDn", s[i]) * op("ProjDn", s[next_i]) +
             (-0.5im * gamma_minus) * op("ProjDn", s[prev_i]) * op("ProjUp", s[i]) * op("ProjDn", s[next_i])
             
    gate = exp(-1.0im * dt * h_term)

    if i % 3 == 1
      push!(gates_1, gate)
    elseif i % 3 == 2
      push!(gates_2, gate)
    else
      push!(gates_3, gate)
    end
  end

  # 2. Jump rate MPOs and Jump Operators
  rate_MPOs = MPO[]
  gammas    = Float64[]
  jump_ops  = ITensor[] 

  for i in 1:L
    prev_i = i == 1 ? L : i - 1
    next_i = i == L ? 1 : i + 1

    # L+ channel (empty->occupied: Dn->Up)
    os_plus = OpSum()
    os_plus += 1.0, "ProjDn", prev_i, "ProjDn", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_plus, s))
    push!(gammas, gamma_plus)
    push!(jump_ops, op("ProjDn", s[prev_i]) * op("S+", s[i]) * op("ProjDn", s[next_i]))

    # L- channel (occupied->empty: Up->Dn)
    os_minus = OpSum()
    os_minus += 1.0, "ProjDn", prev_i, "ProjUp", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_minus, s))
    push!(gammas, gamma_minus)
    push!(jump_ops, op("ProjDn", s[prev_i]) * op("S-", s[i]) * op("ProjDn", s[next_i]))
  end

  # 3. Next-Nearest Neighbor MPO
  os_nnn = OpSum()
  for j in 1:L
    j2 = j + 2 > L ? j + 2 - L : j + 2
    os_nnn += 1.0, "ProjUp", j, "ProjUp", j2
  end
  nnn_MPO = MPO(os_nnn, s)

  # ==========================================
  # Trajectory Loop
  # ==========================================
  
  # Thread-safe counter to monitor progress on the cluster
  completed_trajectories = Threads.Atomic{Int}(0)

  Threads.@threads for traj_id in 0:(num_trajectories - 1)
    rng = MersenneTwister(traj_id)
    thread_id = Threads.threadid()

    # Match Lindblad initial condition
    neel = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    psi = complex(MPS(s, neel))
    psi_init = deepcopy(psi)

    times      = Float64[]
    jump_types = Int[]
    n_vals     = Float64[]
    nnn_vals   = Float64[]
    fid_vals   = Float64[]

    current_time = 0.0
    p_accum      = 1.0
    r_threshold  = rand(rng)
    NO_JUMP      = -1   

    # Initial Event Record
    push!(times, current_time)
    push!(jump_types, 0) 
    n_arr = real.(expect(psi, "ProjUp"))
    push!(n_vals, sum(n_arr) / L)
    push!(nnn_vals, real(inner(psi', nnn_MPO, psi)) / L)
    push!(fid_vals, 1.0) # Fidelity with itself is 1

    for step in 1:steps

      # Calculate Probabilities using the pre-compiled MPOs
      probs = zeros(Float64, 2 * L)
      for k in 1:(2 * L)
        probs[k] = real(inner(psi', rate_MPOs[k], psi)) * gammas[k] * dt
      end
      probs_sum = sum(probs)
      p_accum *= max(0.0, 1.0 - probs_sum)

      if r_threshold >= p_accum

        # Select channel
        target     = probs_sum * rand(rng)
        cumulative = 0.0
        selected_k = 1
        for (k, p) in enumerate(probs)
          cumulative += p
          if target <= cumulative
            selected_k = k
            break
          end
        end

        # Apply exact pre-compiled jump operator
        psi = apply(jump_ops[selected_k], psi; cutoff=1e-10, maxdim=256)
        normalize!(psi)

        p_accum     = 1.0
        r_threshold = rand(rng)

        # Record Jump
        push!(times, current_time)
        push!(jump_types, (selected_k - 1) % 2)
        n_arr = real.(expect(psi, "ProjUp"))
        push!(n_vals, sum(n_arr) / L)
        push!(nnn_vals, real(inner(psi', nnn_MPO, psi)) / L)
        push!(fid_vals, abs2(inner(psi_init, psi)))

      else
        # 3 Applications of TEBD
        psi = apply(gates_1, psi; cutoff=1e-8, maxdim=256)
        psi = apply(gates_2, psi; cutoff=1e-8, maxdim=256)
        psi = apply(gates_3, psi; cutoff=1e-8, maxdim=256)
        normalize!(psi) 

        # Record No-Jump
        push!(times, current_time)
        push!(jump_types, NO_JUMP)
        n_arr = real.(expect(psi, "ProjUp"))
        push!(n_vals, sum(n_arr) / L)
        push!(nnn_vals, real(inner(psi', nnn_MPO, psi)) / L)
        push!(fid_vals, abs2(inner(psi_init, psi)))

      end

      current_time += dt
    end

    # --- File I/O ---
    filename = joinpath(output_dir, "traj_$(traj_id).csv")
    open(filename, "w") do io
        write(io, "time,jump_type,n,nnn_corr,fidelity\n")
        for i in eachindex(times)
            @printf(io, "%.5f,%d,%.6f,%.6f,%.6f\n",
                    times[i], jump_types[i], n_vals[i], nnn_vals[i], fid_vals[i])
        end
    end
    
    Threads.atomic_add!(completed_trajectories, 1)
    println("Thread $thread_id finished trajectory $traj_id. Progress: $(completed_trajectories[]) / $num_trajectories")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time simulate_pxp_trajectories()
end