# Debug version with extensive logging
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using Random
const im = complex(0,1)

println("Starting debug version...")

# ---------------- Parameters ----------------
L_min = 4
L_max = 4  # Start with just L=4 for debugging
Ω = 1.0
γ_p = 0.3
γ_m = 0.7

# Reduce parameters for debugging
maxdim_apply = 100  # Reduced from 600
cutoff_apply  = 1e-6  # Relaxed from 1e-9
krylov_dim    = 10   # Reduced from 30

println("Parameters set successfully")

# ---------------- Local (2x2) operators ----------------
sx = [0.0 1.0; 1.0 0.0]
sp = [0.0 1.0; 0.0 0.0]
sm = [0.0 0.0; 1.0 0.0]
P  = [1.0 0.0; 0.0 0.0]
I2 = Matrix{Float64}(I,2,2)

println("Local operators defined")

# 2x2 -> 4x4 local superoperators (vectorized ordering)
left_super(A)  = kron(A, I2)            # A ⊗ I
right_super(A) = kron(I2, transpose(A)) # I ⊗ A^T
identity4 = kron(I2, I2)

println("Superoperators defined, identity4 size: ", size(identity4))

# ---------------- Utilities ----------------
function make_local4_list_for_three_site(L, i, mats2::Vector{Array{Float64,2}})
    println("  Making local4 list for site $i")
    local4 = [ copy(identity4) for _ in 1:L ]
    idxs = [ ((i-1) % L) + 1, ((i  ) % L) + 1, ((i+1) % L) + 1 ]
    println("    Three-site indices: $idxs")
    for (k,site) in enumerate(idxs)
        local4[site] = mats2[k]
    end
    println("  Successfully created local4 list")
    return local4
end

function product_mpo_from_local4(sites::Vector{<:Index}, local4::Vector{Array{Float64,2}})
    println("  Building MPO from local4 operators...")
    L = length(sites)
    
    # Create bond indices with smaller dimension for debugging
    links = [ Index(1, "link,$i") for i in 1:(L-1) ]
    println("    Created $L sites and $(L-1) links")
    
    # Initialize MPO
    W = MPO(L)  # Try different MPO constructor
    
    for n in 1:L
        println("    Processing site $n")
        leftlink  = n == 1 ? Index(1,"l0") : links[n-1]
        rightlink = n == L ? Index(1,"rL") : links[n]
        
        # Create tensor
        T = ITensor(leftlink, rightlink, prime(sites[n]), sites[n])
        op4 = local4[n]
        
        # Fill tensor elements
        for a in 1:4, b in 1:4
            val = op4[a,b]
            if abs(val) > 1e-14  # Only set non-zero elements
                T[leftlink => 1, rightlink => 1, prime(sites[n]) => a, sites[n] => b] = val
            end
        end
        W[n] = T
        println("    Site $n tensor created successfully")
    end
    println("  MPO creation completed")
    return W
end

function simple_mps_test(sites::Vector{<:Index})
    println("Testing simple MPS creation...")
    L = length(sites)
    psi = MPS(ComplexF64, sites, "0")  # Try simple product state
    println("Simple MPS created successfully")
    return psi
end

# ---------------- Main debugging loop ----------------
Random.seed!(1234)

for L in L_min:L_max
    @printf("\n=== L = %d ===\n", L)
    
    # Create sites
    println("Creating sites...")
    sites = [ Index(4, "s,$i") for i in 1:L ]
    println("Sites created: $L sites of dimension 4")
    
    # Test simple MPS creation first
    try
        test_mps = simple_mps_test(sites)
        println("Basic MPS test passed")
    catch e
        println("ERROR in basic MPS test: $e")
        break
    end
    
    # Test a single simple MPO
    println("Testing single MPO creation...")
    try
        mats2 = [P, sx, P]
        local4_test = make_local4_list_for_three_site(L, 1, [left_super(m) for m in mats2])
        mpo_test = product_mpo_from_local4(sites, local4_test)
        println("Single MPO test passed")
    catch e
        println("ERROR in single MPO test: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        break
    end
    
    println("Basic tests completed successfully for L=$L")
    
    # If we get here, try building a few more MPOs
    println("Testing multiple MPO creation...")
    mpo_terms = MPO[]
    
    try
        # Just test Hamiltonian terms first
        for i in 1:2  # Only test first 2 sites
            println("Processing Hamiltonian for site $i")
            mats2 = [P, sx, P]
            local4_left  = make_local4_list_for_three_site(L, i, [left_super(m)  for m in mats2])
            local4_right = make_local4_list_for_three_site(L, i, [right_super(m) for m in mats2])
            
            mpo_left = product_mpo_from_local4(sites, local4_left)
            mpo_right = product_mpo_from_local4(sites, local4_right)
            
            push!(mpo_terms, mpo_left * Ω)
            push!(mpo_terms, mpo_right * (-Ω))
            println("Site $i Hamiltonian terms added successfully")
        end
        
        println("Multiple MPO creation test passed")
        
    catch e
        println("ERROR in multiple MPO test: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        break
    end
end

println("Debug session completed")