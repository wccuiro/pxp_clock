# Minimal test to isolate the segfault
println("=== Minimal ITensor Test ===")

println("1. Testing basic Julia operations...")
x = [1.0 2.0; 3.0 4.0]
y = x * x
println("Basic matrix operations work")

println("2. Testing LinearAlgebra...")
using LinearAlgebra
I2 = Matrix{Float64}(I, 2, 2)
println("LinearAlgebra works, I2 = ", I2)

println("3. Testing ITensors import...")
try
    using ITensors
    println("ITensors imported successfully")
catch e
    println("ERROR importing ITensors: $e")
    exit(1)
end

println("4. Testing ITensorMPS import...")
try
    using ITensorMPS
    println("ITensorMPS imported successfully")
catch e
    println("ERROR importing ITensorMPS: $e")
    exit(1)
end

println("5. Testing basic Index creation...")
try
    i = Index(2, "test")
    println("Basic Index created: $i")
catch e
    println("ERROR creating Index: $e")
    exit(1)
end

println("6. Testing multiple Index creation...")
try
    sites = [Index(2, "s$j") for j in 1:3]
    println("Multiple indices created successfully: $(length(sites)) indices")
catch e
    println("ERROR creating multiple indices: $e")
    exit(1)
end

println("7. Testing Index with dimension 4...")
try
    i4 = Index(4, "test4")
    println("Index with dim 4 created: $i4")
catch e
    println("ERROR creating Index with dim 4: $e")
    exit(1)
end

println("8. Testing MPS constructor...")
try
    sites = [Index(2, "s$j") for j in 1:3]
    # Try different MPS constructors
    psi1 = MPS(3)  # Empty MPS
    println("Empty MPS created")
    
    psi2 = MPS(ComplexF64, sites, "0")  # Product state
    println("Product state MPS created")
catch e
    println("ERROR creating MPS: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

println("9. Testing ITensor creation...")
try
    i = Index(2, "i")
    j = Index(2, "j")
    T = ITensor(i, j)
    println("ITensor created successfully")
catch e
    println("ERROR creating ITensor: $e")
    exit(1)
end

println("10. All basic tests passed!")
println("The issue might be elsewhere. Let's check version info:")

try
    using Pkg
    println("\nPackage versions:")
    Pkg.status(["ITensors", "ITensorMPS"])
catch e
    println("Could not get package status: $e")
end