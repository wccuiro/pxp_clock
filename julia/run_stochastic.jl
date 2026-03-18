include("pxp_stochastic.jl")

@time simulate_pxp_trajectories(
  L = 10,
  Omega = 1.0,
  gamma_plus  = 0.2,
  gamma_minus = 0.2,
  dt = 1e-1,
  total_time = 50.0,
  num_trajectories = 100,
  output_dir = "../data/trajectoriesTN"

)