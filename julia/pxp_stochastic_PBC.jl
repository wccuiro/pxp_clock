using ITensors
using ITensorMPS   
using Random
using Printf

function simulate_pxp_trajectories(;
  # --- 1. System Parameters ---
  L = 10,
  Omega = 1.0,
  gamma_plus  = 0.001,
  gamma_minus = 0.2,
  dt = 1e-1,
  total_time = 25.0,
  num_trajectories = 50,
  output_dir = "../data/trajectoriesTN"
  )

  output_dir = "../data/trajectoriesTN_L$(L)_gamma+$(gamma_plus)_gamma-$(gamma_minus)_dt$(dt)"

  mkpath(output_dir)
  steps = round(Int, total_time / dt)

  println("--- PXP Tensor Network Simulation ---")
  println("Parameters: L=$L, Omega=$Omega, Gamma+=$gamma_plus, Gamma-=$gamma_minus, dt=$dt")

  # Spin-1/2 mapping: "Up" = Occupied (1), "Dn" = Empty (0)
  s = siteinds("S=1/2", L)

  # --- 2. Build Operators ---
  # TEBD Gates (3-site blocks for PBC)
  gates_1 = ITensor[]
  gates_2 = ITensor[]
  gates_3 = ITensor[]

  for i in 1:L
    prev_i = i == 1 ? L : i - 1
    next_i = i == L ? 1 : i + 1

    # Effective non-Hermitian Hamiltonian term for site i
    # PXP term requires neighbors to be empty (ProjDn)
    h_term = (2.0 * Omega) * op("ProjDn", s[prev_i]) * op("Sx", s[i]) * op("ProjDn", s[next_i]) +
             (-0.5im * gamma_plus) * op("ProjDn", s[prev_i]) * op("ProjDn", s[i]) * op("ProjDn", s[next_i]) +
             (-0.5im * gamma_minus) * op("ProjDn", s[prev_i]) * op("ProjUp", s[i]) * op("ProjDn", s[next_i])
             
    # Exponentiate the 3-site term for TEBD
    gate = exp(-1.0im * dt * h_term)

    # Separate into 3 commuting groups
    if i % 3 == 1
      push!(gates_1, gate)
    elseif i % 3 == 2
      push!(gates_2, gate)
    else
      push!(gates_3, gate)
    end
  end

  # Jump rate MPOs for PBC: <psi|rate_MPOs[k]|psi> = <L_k^dag L_k>
  rate_MPOs = MPO[]
  gammas    = Float64[]
  for i in 1:L
    prev_i = i == 1 ? L : i - 1
    next_i = i == L ? 1 : i + 1

    # L+^dag L+ = P_{prev} ProjDn_i P_{next}  (site empty, can be excited)
    os_plus = OpSum()
    os_plus += 1.0, "ProjDn", prev_i, "ProjDn", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_plus, s))
    push!(gammas, gamma_plus)

    # L-^dag L- = P_{prev} ProjUp_i P_{next}  (site occupied, can decay)
    os_minus = OpSum()
    os_minus += 1.0, "ProjDn", prev_i, "ProjUp", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_minus, s))
    push!(gammas, gamma_minus)
  end

  # MPO for <n_{j-1} n_{j+1}> which is equivalent to distance-2 correlator <n_j n_{j+2}>
  os_nnn = OpSum()
  for j in 1:L
    j2 = j + 2 > L ? j + 2 - L : j + 2
    # n is measured by ProjUp (Occupied)
    os_nnn += 1.0, "ProjUp", j, "ProjUp", j2
  end
  nnn_MPO = MPO(os_nnn, s)

  # --- 3. Trajectory Loop ---
  Threads.@threads for traj_id in 0:(num_trajectories - 1)

    rng = MersenneTwister(traj_id)

    # Superposition of Neel state and translated Neel state
    # Neel state: alternating Occupied (Up) and Empty (Dn)
    neel_1 = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    # neel_2 = [isodd(i) ? "Dn" : "Up" for i in 1:L]
    psi1 = complex(MPS(s, neel_1))
    # psi2 = complex(MPS(s, neel_2))
    # psi = +(psi1, psi2; cutoff=1e-10)
    psi = psi1
    normalize!(psi)
    
    # Save a deepcopy of the initial state to compute fidelity
    psi_init = deepcopy(psi)

    times      = Float64[]
    jump_types = Int[]
    n_vals     = Float64[]
    nnn_vals   = Float64[]
    fid_vals   = Float64[]

    current_time    = 0.0
    p_accum         = 1.0
    r_threshold     = rand(rng)
    NO_JUMP = -1   

    # Record initial jump event & observables
    push!(times, current_time)
    push!(jump_types, 0)  # 0 = L+ (excitation), 1 = L- (decay)
    
    n_arr = real.(expect(psi, "ProjUp"))
    push!(n_vals, sum(n_arr) / L)
    push!(nnn_vals, real(inner(psi', nnn_MPO, psi)) / L)
    push!(fid_vals, abs2(inner(psi_init, psi)))

    for step in 1:steps

      # --- Calculate Jump Probabilities ---
      probs = zeros(Float64, 2 * L)
      for k in 1:(2 * L)
        probs[k] = real(inner(psi', rate_MPOs[k], psi)) * gammas[k] * dt
      end
      probs_sum = sum(probs)
      p_accum *= max(0.0, 1.0 - probs_sum)

      # --- Check for Jump ---
      if r_threshold >= p_accum

        # Select jump channel proportional to its rate
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

        site_idx = ceil(Int, selected_k / 2)
        prev_idx = site_idx == 1 ? L : site_idx - 1
        next_idx = site_idx == L ? 1 : site_idx + 1

        # Odd k  → L+ channel (empty→occupied: Dn→Up = "S+")
        # Even k → L- channel (occupied→empty: Up→Dn = "S-")
        op_str = (selected_k % 2 != 0) ? "S+" : "S-"

        gate = op("ProjDn", s[prev_idx]) *
               op(op_str,   s[site_idx]) *
               op("ProjDn", s[next_idx])
        psi = apply(gate, psi; cutoff=1e-10, maxdim=256)
        normalize!(psi)

        p_accum     = 1.0
        r_threshold = rand(rng)

        # Record jump event & observables
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

        # Record no-jump event & observables
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
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time simulate_pxp_trajectories()
end