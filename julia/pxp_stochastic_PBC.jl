using ITensors
using ITensorMPS   # Fix 1: replaces deprecated ITensorTDVP
using Random
using Printf

function simulate_pxp_trajectories(;
  # --- 1. System Parameters ---
  L = 10,
  Omega = 1.0,
  gamma_plus  = 0.2,
  gamma_minus = 0.2,
  dt = 1e-1,
  total_time = 50.0,
  num_trajectories = 40,
  output_dir = "../data/trajectoriesTN"
  )

  mkpath(output_dir)
  # Fix 11: round instead of ceil to avoid off-by-one from floating-point arithmetic
  steps = round(Int, total_time / dt)

  println("--- PXP Tensor Network Simulation ---")
  println("Parameters: L=$L, Omega=$Omega, Gamma+=$gamma_plus, Gamma-=$gamma_minus, dt=$dt")

  # Spin-1/2 mapping: "Up" = Empty (0), "Dn" = Occupied (1)
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

    # L+^dag L+ = P_{prev} ProjUp_i P_{next}  (site empty, can be excited)
    os_plus = OpSum()
    os_plus += 1.0, "ProjDn", prev_i, "ProjDn", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_plus, s))
    push!(gammas, gamma_plus)

    # L-^dag L- = P_{prev} ProjDn_i P_{next}  (site occupied, can decay)
    os_minus = OpSum()
    os_minus += 1.0, "ProjDn", prev_i, "ProjUp", i, "ProjDn", next_i
    push!(rate_MPOs, MPO(os_minus, s))
    push!(gammas, gamma_minus)
  end

  # MPO for <n_{j-1} n_{j+1}> which is equivalent to distance-2 correlator <n_j n_{j+2}>
  os_nnn = OpSum()
  for j in 1:L
    j2 = j + 2 > L ? j + 2 - L : j + 2
    os_nnn += 1.0, "ProjDn", j, "ProjDn", j2
  end
  nnn_MPO = MPO(os_nnn, s)

  # --- 3. Trajectory Loop ---
  Threads.@threads for traj_id in 0:(num_trajectories - 1)

    # Fix 4: per-thread seeded RNG — thread-safe and reproducible
    rng = MersenneTwister(traj_id)

    # Superposition of Neel state and translated Neel state
    neel_1 = [isodd(i) ? "Dn" : "Up" for i in 1:L]
    neel_2 = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    psi1 = complex(MPS(s, neel_1))
    psi2 = complex(MPS(s, neel_2))
    psi = +(psi1, psi2; cutoff=1e-10)
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
    record_dt       = 1e-2
    next_record_time = 0.0
    NO_JUMP = -1   # Fix 9: use -1 as no-jump sentinel (was 2*L, fragile)

    # Record initial jump event & observables
    push!(times, current_time)
    push!(jump_types, 0)  # 0 = L+ (excitation), 1 = L- (decay)
    
    n_arr = real.(expect(psi, "ProjDn"))
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
      # Fix 7: clamp to prevent negative p_accum when probs_sum > 1
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

        # Odd k  → L+ channel (empty→occupied: Up→Dn = "S-")
        # Even k → L- channel (occupied→empty: Dn→Up = "S+")
        op_str = (selected_k % 2 != 0) ? "S+" : "S-"

        # Fix 5: contract into a single 3-site ITensor before applying,
        # and add maxdim to prevent bond-dimension blowup.
        # Note: for PBC boundary sites (i=1 or i=L) prev/next are non-contiguous;
        # ITensorMPS.apply handles this internally via SWAP gates.
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
        
        n_arr = real.(expect(psi, "ProjDn"))
        push!(n_vals, sum(n_arr) / L)
        push!(nnn_vals, real(inner(psi', nnn_MPO, psi)) / L)
        push!(fid_vals, abs2(inner(psi_init, psi)))

      else
        # 3 Applications of TEBD instead of TDVP
        psi = apply(gates_1, psi; cutoff=1e-8, maxdim=256)
        psi = apply(gates_2, psi; cutoff=1e-8, maxdim=256)
        psi = apply(gates_3, psi; cutoff=1e-8, maxdim=256)
        normalize!(psi) 

        # Record no-jump event & observables
        push!(times, current_time)
        push!(jump_types, NO_JUMP)
        
        n_arr = real.(expect(psi, "ProjDn"))
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