using ITensors, ITensorMPS

function main()
  # Use a larger N to show the power of DMRG
  N = 12
  # Number of states to find (1 ground state + 2 excited states)
  nstates = 4096
  sites = siteinds("S=1/2", N)

  ## 1. Build the PXP Hamiltonian MPO
  os = OpSum()
  for i in 1:N
    im1 = mod1(i - 1, N)
    ip1 = mod1(i + 1, N)
    os += 0.25, "Id", im1, "X", i, "Id", ip1
    os -= 0.5, "Id", im1, "X", i, "Sz", ip1
    os -= 0.5, "Sz", im1, "X", i, "Id", ip1
    os += 1.0, "Sz", im1, "X", i, "Sz", ip1
  end
  H = MPO(os, sites)

  ## 2. Set up DMRG parameters
  sweeps = Sweeps(15) # 10 sweeps are good for convergence
  setmaxdim!(sweeps, 20, 60, 100, 200) # Gradually increase bond dimension
  setcutoff!(sweeps, 1E-10)

  ## 3. Iteratively find the eigenstates
  # println("Starting DMRG to find the $nstates lowest energy states of the PXP model for N=$N.")

  # A list to store the states (eigenvectors) we have already found
  previous_states = MPS[]
  # A list to store the energies (eigenvalues)
  energies = []

  for n in 1:nstates
    # println("\nStarting DMRG for state #$n")
    # Create a new random MPS for the initial guess
    psi_initial = randomMPS(sites)

    # Run DMRG. The key is the 'previous_states' argument.
    # It tells DMRG to find a state orthogonal to all states in the list.
    energy, psi = dmrg(H, previous_states, psi_initial, sweeps)
    
    # println("  Energy of state #$n = $energy")
    
    # Add the new state and energy to our lists
    push!(previous_states, psi)
    push!(energies, energy)
  end

  ## 4. Print the final results
  println("\n--- DMRG Calculation Complete ---")
  println("Found $nstates eigenvalues:")
  # for (n, E) in enumerate(energies)
  #     println("E$n = $E")
  # end

  println("\nEnergy gap (E2 - E1) = ", energies[2] - energies[1])
  println("---------------------------------")
  
  return energies, previous_states
end

# Run the main function
main()