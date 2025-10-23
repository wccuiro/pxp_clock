using ITensors, ITensorMPS
using Printf      # For formatted printing of benchmark results
using LinearAlgebra # To calculate the norm difference between results

#
# This script benchmarks two methods for finding the steady state of a
# classical master equation for a constrained spin model:
# 1. Real-time evolution using TDVP.
# 2. Ground state search on W'W using DMRG.
# The benchmark is performed using a known three-site correlation identity.
#

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
  # Hard-core constraint
  for j in 1:N
    jp1 = mod1(j + 1, N)
    W_os += -V_penalty, "ProjUp", j, "ProjUp", jp1
  end
  return W_os
end


# --- Main Script ---
function main()
  ## 1. Define Parameters
  N = 100
  gamma_plus = 1.0
  gamma_minus = 1.5
  V_penalty = 100.0
  g_ratio = gamma_plus / gamma_minus
  sites = siteinds("S=1/2", N)


  ## 3. Build the W MPO
  println("Building the W MPO with rates and constraints...")
  W_os = build_W_opsum(N; gamma_plus=gamma_plus, gamma_minus=gamma_minus, V_penalty=V_penalty)
  W = MPO(W_os, sites)
  println("✅ W MPO constructed.")

  # ---
  ## 4. Method A: Time Evolution with TDVP
  println("\n--- METHOD A: Time Evolution (TDVP) ---")

  # Creates a state like ["Up", "Dn", "Up", "Dn", ...]
  # initial_state_config = ["Dn" for n in 1:N]
  # psi0_tdvp = MPS(sites, initial_state_config)
  psi0_tdvp = randomMPS(sites; linkdims=6)
  normalize!(psi0_tdvp)
  total_time = 20.0
  time_step = 0.01

  tdvp_time = @elapsed begin
    global p_ss_tdvp = tdvp(W, total_time, psi0_tdvp; 
                            time_step=time_step,
                            normalize=true, 
                            maxdim=150, 
                            cutoff=1e-15)
  end
  println("✅ TDVP complete.")

  # ---
  ## 5. Method B: DMRG on W'W
  println("\n--- METHOD B: DMRG on W'W ---")

  println("Constructing W'W MPO...")
  W_dag = dag(prime(W, "Link"))
  WdagW = apply(W_dag, W; cutoff=1e-15)
  println("✅ W'W MPO constructed.")

  psi0_dmrg = randomMPS(sites; linkdims=6)
  # psi0_dmrg = MPS(sites, initial_state_config)
  sweeps = Sweeps(50)
  setmaxdim!(sweeps, 10, 20, 50, 100, 150)
  setcutoff!(sweeps, 1e-10)

  dmrg_time = @elapsed begin
    # CORRECTED: Replaced observer with `quiet=true` to suppress sweep output
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
  function three_site_corr(psi::MPS, op1::String, op2::String, op3::String, j1::Int, j2::Int, j3::Int)
      s = siteinds(psi)
      
      # Build an OpSum containing only the desired three-site operator term
      os = OpSum()
      os += 1.0, op1, j1, op2, j2, op3, j3
      
      # Convert this simple OpSum into an MPO. This MPO is mostly identity operators.
      O_mpo = MPO(os, s)
      
      # Use the efficient "sandwich" inner product with the MPO
      val = inner(psi', O_mpo, psi)
      
      # Return the real part to discard negligible numerical noise
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
    rhs_tdvp += three_site_corr(p_ss_tdvp, "ProjDn", "ProjUp", "ProjDn", jm1, j, jp1)
    corr_ddd_tdvp += three_site_corr(p_ss_tdvp, "ProjDn", "ProjDn", "ProjDn", jm1, j, jp1)
  end
  rhs_tdvp /= N
  lhs_tdvp = g_ratio * corr_ddd_tdvp / N
  @printf("  LHS (g * <↓↓↓>): %.12f\n", lhs_tdvp)
  @printf("  RHS (<↓↑↓>):     %.12f\n", rhs_tdvp)
  @printf("  Relative Error:  %.2e\n", abs(lhs_tdvp - rhs_tdvp) / abs(rhs_tdvp))

  # --- For the DMRG result ---
  println("\n🔬 Checking identity for DMRG steady state:")
  rhs_dmrg = 0.0
  corr_ddd_dmrg = 0.0
  for i in 1:N
    j = i
    jm1 = mod1(j - 1, N)
    jp1 = mod1(j + 1, N)
    rhs_dmrg += three_site_corr(p_ss_dmrg, "ProjDn", "ProjUp", "ProjDn", jm1, j, jp1)
    corr_ddd_dmrg += three_site_corr(p_ss_dmrg, "ProjDn", "ProjDn", "ProjDn", jm1, j, jp1)
  end
  rhs_dmrg /= N
  lhs_dmrg = g_ratio * corr_ddd_dmrg/N
  @printf("  LHS (g * <↓↓↓>): %.12f\n", lhs_dmrg)
  @printf("  RHS (<↓↑↓>):     %.12f\n", rhs_dmrg)
  @printf("  Relative Error:  %.2e\n", abs(lhs_dmrg - rhs_dmrg) / abs(rhs_dmrg))

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