using ITensors, ITensorMPS
using Printf      # For formatted printing of benchmark results
using LinearAlgebra # To calculate the norm difference between results


ITensors.op(::OpName"ProjUp", ::SiteType"S=1/2", s::Index) = 0.5 * op("Id", s) + op("Sz", s)
ITensors.op(::OpName"ProjDn", ::SiteType"S=1/2", s::Index) = 0.5 * op("Id", s) - op("Sz", s)

"""
    build_W_opsum(N::Int; gamma_plus::Float64, gamma_minus::Float64, V_penalty::Float64)

Creates the OpSum for the classical transition matrix W, including physical
rates and a penalty term for the hard-core constraint forbidding adjacent |↑↑⟩ states.
"""
function build_W_opsum(N::Int; gamma_plus::Float64, gamma_minus::Float64, V_penalty::Float64)
  W_os = OpSum()
  for j in 1:N
    jm1 = mod1(j - 1, N)
    jp1 = mod1(j + 1, N)
    # L_j^+ process
    W_os += gamma_plus, "ProjDn", jm1, "S+", j, "ProjDn", jp1
    W_os += -gamma_minus, "ProjDn", jm1, "ProjUp", j, "ProjDn", jp1
    # L_j^- process
    W_os += gamma_minus, "ProjDn", jm1, "S-", j, "ProjDn", jp1
    W_os += -gamma_plus, "ProjDn", jm1, "ProjDn", j, "ProjDn", jp1
  end
  # # Hard-core constraint
  # for j in 1:N
  #   jp1 = mod1(j + 1, N)
  #   W_os += -V_penalty, "ProjUp", j, "ProjUp", jp1
  # end
  return W_os
end


# --- Main Script ---
function main()
  ## 1. Define Parameters
  N = 100
  gamma_plus = 1.0
  gamma_minus = 1.5
  V_penalty = 200.0
  g_ratio = gamma_plus / gamma_minus
  sites = siteinds("S=1/2", N)


  ## 3. Build the W MPO
  println("Building the W MPO with rates and constraints...")
  W_os = build_W_opsum(N; gamma_plus=gamma_plus, gamma_minus=gamma_minus, V_penalty=V_penalty)
  W = MPO(W_os, sites)
  println("✅ W MPO constructed.")

  # ---
  ## 4. Method A: Time Evolution with TDVP
  println("\n--- METHOD A: Time Evolution (TDVP) with Sum-of-Components Normalization ---")

  # Let's call the MPS `p_mps` to clarify it represents a population vector.
  # initial_state_config = ["Dn" for n in 1:N]
  # psi0_tdvp = MPS(sites, initial_state_config)
  psi0_tdvp = randomMPS(sites; linkdims=6)

  # --- Construct the "Summation Vector" as an MPS ---
  # This is an MPS where each local tensor is a vector of all ones.
  # The inner product <sum_mps|p_mps> yields the sum of all components of p_mps.
  sum_tensors = ITensor[]
  for j in 1:N
    s = sites[j]
    T = ITensor(1.0, s) # Fills the tensor with the value 1.0
    push!(sum_tensors, T)
  end
  sum_mps = MPS(sum_tensors)
  println("✅ Summation MPS for normalization constructed.")

  # --- Perform initial normalization ---
  initial_sum = inner(sum_mps, psi0_tdvp)
  psi0_tdvp /= initial_sum
  println("Initial sum of components is: ", inner(sum_mps, psi0_tdvp))

  # Evolution parameters
  total_time = 50.0
  time_step = 0.1
  num_steps = Int(div(total_time, time_step))

  # --- Custom TDVP Loop ---
  tdvp_time = @elapsed begin
    for step in 1:num_steps
      # Evolve by a single time_step. The solver propagates by exp(W * time_step).
      # We disable the default L2 norm normalization provided by ITensor.
      tdvp(W, time_step, psi0_tdvp; 
          normalize=false, 
          maxdim=500, 
          cutoff=1e-15)

      # Manually normalize by the sum of components after the step
      current_sum = inner(sum_mps, psi0_tdvp)
      psi0_tdvp /= current_sum
    end
    # Assign the final state to the global variable
    global p_ss_tdvp = psi0_tdvp
  end

  println("Final sum of components after evolution: ", inner(sum_mps, p_ss_tdvp))
  println("✅ TDVP complete.")
  
  # p_ss_tdvp = psi0_tdvp
  # tdvp_time = 0.0
  # ---
  ## 5. Method B: DMRG on W'W
  println("\n--- METHOD B: DMRG on W'W ---")

  # println("Constructing W'W MPO...")
  W_dag = dag(prime(W, "Link"))
  WdagW = apply(W_dag, W; cutoff=0, alg="zipup")
  # WdagW = apply(W_dag, W; cutoff=1e-15)
  # WdagW = contract(W_dag, W; method="naive", cutoff=0)
  # println("✅ W'W MPO constructed.")

  psi0_dmrg = randomMPS(sites; linkdims=12)
  # psi0_dmrg = MPS(sites, initial_state_config)
  sweeps = Sweeps(10)
  setmaxdim!(sweeps, 50, 100, 150, 200)
  setcutoff!(sweeps, 1e-18)

  dmrg_time = @elapsed begin
    global energy_dmrg, p_ss_dmrg = dmrg(WdagW, psi0_dmrg, sweeps; outputlevel=0)
  end
  println("✅ DMRG complete.")

  # ---
  ## 6. 📊 Benchmark and Comparison
  println("\n\n--- BENCHMARK RESULTS ---")
  println("Comparing methods using the analytical identity:")
  println("g * <(1-n_j-1)(1-n_j)(1-n_j+1)> = <(1-n_j-1)(n_j)(1-n_j+1)>")

  # --- Performance ---
  println("\n⏱️  Performance:")
  println("TDVP wall time:       ", tdvp_time, " seconds")
  println("DMRG (on W'W) time:   ", dmrg_time, " seconds")
  println("DMRG final energy (should be ≈ 0): ", energy_dmrg)


  """
      three_site_corr(psi::MPS, op1::String, op2::String, op3::String, j1::Int, j2::Int, j3::Int)

  Correctly and robustly calculates the three-site correlation function
  <ψ|O₁(j₁) O₂(j₂) O₃(j₃)|ψ> by building a lightweight MPO for the operator.
  """
  function three_site_corr(p_mps::MPS, sum_mps::MPS, op1::String, op2::String, op3::String, j1::Int, j2::Int, j3::Int)
    s = siteinds(p_mps)
    
    # Build an OpSum for the diagonal operator O.
    os = OpSum()
    os += 1.0, op1, j1, op2, j2, op3, j3
    
    # [cite_start]Convert the OpSum into an MPO[cite: 1572].
    O_mpo = MPO(os, s)
    
    # This computes Σᵢ pᵢ Oᵢ, the correct classical expectation value.
    # It is equivalent to inner(sum_mps, apply(O_mpo, p_mps)).
    val = inner(sum_mps, apply(O_mpo, p_mps))
    
    return real(val)
  end  
  
  # --- For the TDVP result ---
  println("\n🔬 Checking identity for TDVP steady state:")
  rhs_tdvp = 0.0
  corr_ddd_tdvp = 0.0
  for i in 1:N
    j = i
    jm1 = mod1(j - 1, N)
    jp1 = mod1(j + 1, N)
    rhs_tdvp += three_site_corr(p_ss_tdvp, sum_mps, "ProjDn", "ProjUp", "ProjDn", jm1, j, jp1)
    corr_ddd_tdvp += three_site_corr(p_ss_tdvp, sum_mps, "ProjDn", "ProjDn", "ProjDn", jm1, j, jp1)
  end
  rhs_tdvp /= N
  lhs_tdvp = g_ratio * corr_ddd_tdvp / N
  @printf("  LHS (g * <↓↓↓>): %.12f\n", lhs_tdvp)
  @printf("  RHS (<↓↑↓>):     %.12f\n", rhs_tdvp)
  @printf("  Difference Error:  %.2e\n", abs(lhs_tdvp - rhs_tdvp) )

  # --- For the DMRG result ---
  println("\n🔬 Checking identity for DMRG steady state:")
  rhs_dmrg = 0.0
  corr_ddd_dmrg = 0.0
  for i in 1:N
    j = i
    jm1 = mod1(j - 1, N)
    jp1 = mod1(j + 1, N)
    rhs_dmrg += three_site_corr(p_ss_dmrg, sum_mps, "ProjDn", "ProjUp", "ProjDn", jm1, j, jp1)
    corr_ddd_dmrg += three_site_corr(p_ss_dmrg, sum_mps, "ProjDn", "ProjDn", "ProjDn", jm1, j, jp1)
  end
  rhs_dmrg /= N
  lhs_dmrg = g_ratio * corr_ddd_dmrg/N
  @printf("  LHS (g * <↓↓↓>): %.12f\n", lhs_dmrg)
  @printf("  RHS (<↓↑↓>):     %.12f\n", rhs_dmrg)
  @printf("  Difference Error:  %.2e\n", abs(lhs_dmrg - rhs_dmrg))

  # # Check hard-core constraint violations
  # constraint_violation = 0.0
  # for j in 1:N
  #     jp1 = mod1(j + 1, N)
  #     constraint_violation += expect(p_ss_tdvp, "ProjUp", j) * expect(p_ss_tdvp, "ProjUp", jp1)
  # end
  # println("Constraint violation: ", constraint_violation / N)

  # --- Direct Comparison of Methods ---
  println("\n🔬 Direct comparison of the two methods:")
  rhs_diff = abs(rhs_tdvp - rhs_dmrg)
  lhs_corr_diff = abs(corr_ddd_tdvp - corr_ddd_dmrg)
  println("  Absolute difference in RHS correlator <↓↑↓>: ", @sprintf("%.2e", rhs_diff))
  println("  Absolute difference in <↓↓↓> correlator:   ", @sprintf("%.2e", lhs_corr_diff))

  if rhs_diff < 1e-6 && lhs_corr_diff < 1e-6
      println("\n✅ Agreement between TDVP and DMRG is excellent.")
  else
      println("\n⚠️  Agreement is not perfect. Try increasing tdvp time/sweeps or MPS bond dimensions.")
  end
end

main()