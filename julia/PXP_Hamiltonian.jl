using ITensors, ITensorMPS
using LinearAlgebra # Import Julia's linear algebra library

function main()
  # N must be small for exact diagonalization!
  N = 12
  sites = siteinds("S=1/2", N)

  # Build the Hamiltonian OpSum (same as before)
  os = OpSum()
  for i in 1:N
    im1 = mod1(i - 1, N)
    ip1 = mod1(i + 1, N)

    os += 0.25, "Id", im1, "X", i, "Id", ip1
    os -= 0.5, "Id", im1, "X", i, "Sz", ip1
    os -= 0.5, "Sz", im1, "X", i, "Id", ip1
    os += 1.0, "Sz", im1, "X", i, "Sz", ip1
  end
  H_mpo = MPO(os, sites)

  H_itensor = H_mpo[1]
  for i in 2:N
      H_itensor *= H_mpo[i]
  end

  # 2. Get the physical site indices (the "input" or column indices)
  site_inds = sites

  # 3. Create a combiner to group all site_inds into a single new index
  col_combiner = combiner(site_inds...; tags="col")

  # 4. Apply the combiner to H_itensor. This bundles the site_inds.
  # The prime(...) primes all the site indices to get the "output" or row indices.
  H_combined = (H_itensor * col_combiner) * combiner(prime(site_inds)...; tags="row")

  # 5. Now H_combined is an ITensor with only 2 indices.
  # We can convert it to a standard Julia matrix.
  println("Converting combined ITensor to a dense matrix...")
  H_matrix = matrix(H_combined)
  println("Matrix size: ", size(H_matrix))

  # Diagonalize the matrix to get the full spectrum
  println("Finding eigenvalues...")
  spectrum = eigen(H_matrix)

  # 'spectrum' is an object containing both eigenvalues and eigenvectors
  eigenvalues = spectrum.values
  eigenvectors = spectrum.vectors

  println("\nLowest 5 eigenvalues:")
  for n in 1:5
      println(eigenvalues[n])
  end

  return eigenvalues, eigenvectors
end

# Run the main function
main()