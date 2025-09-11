using ITensors

function test_itensor()
    println("Testing ITensor.jl installation...")
    
    # Create simple indices
    i = Index(4, "i")
    j = Index(3, "j")
    
    # Create an ITensor
    T = ITensor(i, j)
    
    # Set some elements
    T[i => 1, j => 1] = 1.5
    T[i => 2, j => 2] = 2.5
    T[i => 3, j => 3] = 3.5
    
    # Print tensor info
    println("✓ ITensor.jl installation successful!")
    println("Tensor T has $(ndims(T)) indices")
    println("Index i has dimension $(dim(i))")
    println("Index j has dimension $(dim(j))")
    println("T[1,1] = $(T[i => 1, j => 1])")
    
    # Test basic tensor operations
    T2 = 2.0 * T
    println("After scaling by 2: T2[1,1] = $(T2[i => 1, j => 1])")
    
    # Test tensor contraction
    k = Index(3, "k")
    A = randomITensor(i, k)
    B = randomITensor(k, j)
    C = A * B  # Contract over shared index k
    println("✓ Tensor contraction test passed")
    println("Contracted tensor C has $(ndims(C)) indices")
    
    println("\nAll tests passed! ITensor.jl is working correctly.")
end

# Run the test
test_itensor()