using ITensors
using ITensorMPS   # Fix 1: replaces deprecated ITensorTDVP
using Random
using Printf

function simulate_pxp_trajectories()
  # --- 1. System Parameters ---
  L = 8
  Omega = 1.0
  gamma_plus  = 0.2
  gamma_minus = 0.2
  dt = 1e-1
  total_time = 50.0
  num_trajectories = 40
  output_dir = "data/trajectoriesTN"

  mkpath(output_dir)
  # Fix 11: round instead of ceil to avoid off-by-one from floating-point arithmetic
  steps = round(Int, total_time / dt)

  println("--- PXP Tensor Network Simulation ---")
  println("Parameters: L=$L, Omega=$Omega, Gamma+=$gamma_plus, Gamma-=$gamma_minus, dt=$dt")

  # Spin-1/2 mapping: "Up" = Empty (0), "Dn" = Occupied (1)
  s = siteinds("S=1/2", L)

  # --- 2. Build Operators ---
  # Effective non-Hermitian Hamiltonian MPO:
  # H_eff = H_PXP - (i/2) * sum_i [ gamma_+ L_+^dag L_+ + gamma_- L_-^dag L_- ]
  os_H = OpSum()
  for i in 2:(L-1)
    prev_i = i - 1
    next_i =  i + 1

    # PXP term: ITensor "Sx" = sigma_x / 2, so use 2*Omega to get Omega * sigma_x
    os_H += (2.0 * Omega), "ProjDn", prev_i, "Sx", i, "ProjDn", next_i

    # -i/2 * gamma_+ * L_+^dag L_+ ; L_+^dag L_+ = P_{prev} ProjUp_i P_{next}
    os_H += (-0.5im * gamma_plus),  "ProjDn", prev_i, "ProjDn", i, "ProjDn", next_i
    # -i/2 * gamma_- * L_-^dag L_- ; L_-^dag L_- = P_{prev} ProjDn_i P_{next}
    os_H += (-0.5im * gamma_minus), "ProjDn", prev_i, "ProjUp", i, "ProjDn", next_i
  end
  H_eff = MPO(os_H, s)

  # Jump rate MPOs: <psi|rate_MPOs[k]|psi> = <L_k^dag L_k>
  rate_MPOs = MPO[]
  gammas    = Float64[]
  for i in 2:(L-1)
    prev_i = i - 1
    next_i = i + 1

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

  # --- 3. Trajectory Loop ---
  Threads.@threads for traj_id in 0:(num_trajectories - 1)

    # Fix 4: per-thread seeded RNG — thread-safe and reproducible
    rng = MersenneTwister(traj_id)

    # Fix 6: complex MPS — required because H_eff has complex-valued tensors
    psi = complex(MPS(s, "Dn"))

    times      = Float64[]
    jump_types = Int[]
    sz_vals    = Float64[]
    occ_vals   = Float64[]

    current_time    = 0.0
    p_accum         = 1.0
    r_threshold     = rand(rng)
    record_dt       = 0.1
    next_record_time = 0.0
    NO_JUMP = -1   # Fix 9: use -1 as no-jump sentinel (was 2*L, fragile)

    for step in 1:steps

      # --- Calculate Jump Probabilities ---
      probs = zeros(Float64, 2 * L-2)
      for k in 1:(2 * L-4)
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

        site_idx = ceil(Int, selected_k / 2) + 1
        prev_idx = site_idx - 1
        next_idx = site_idx + 1

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

        # Record jump event
        push!(times,      current_time)
        push!(jump_types, (selected_k - 1) % 2)  # 0 = L+ (excitation), 1 = L- (decay)
        # Fix 8: real.() needed because psi is now complex
        occ_arr = real.(expect(psi, "ProjUp"))
        push!(occ_vals, sum(occ_arr) / L)
        push!(sz_vals,  sum(2 .* occ_arr .- 1.0) / L)


      else
        psi = tdvp(H_eff, -im * dt, psi; cutoff=1e-8, maxdim=256)
        normalize!(psi) 

        occ_arr = real.(expect(psi, "ProjUp"))  # Fix 8
        push!(times,      current_time)
        push!(jump_types, NO_JUMP)
        push!(occ_vals,   sum(occ_arr) / L)
        push!(sz_vals,    sum(2 .* occ_arr .- 1.0) / L)

      end

      current_time += dt
    end

    # --- File I/O ---
    filename = joinpath(output_dir, "traj_$(traj_id).csv")
    open(filename, "w") do io
        write(io, "time,jump_type,sz,n\n")
        for i in eachindex(times)
            @printf(io, "%.5f,%d,%.6f,%.6f\n",
                    times[i], jump_types[i], sz_vals[i], occ_vals[i])
        end
    end
  end
end

@time simulate_pxp_trajectories()