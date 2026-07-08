using ITensors
using ITensorMPS
using LinearAlgebra

# P projects onto |0> (Spin Down, which is state 2 in ITensor S=1/2)
ITensors.op(::OpName"P0", ::SiteType"S=1/2") = [0.0 0.0; 0.0 1.0]
ITensors.op(::OpName"X0", ::SiteType"S=1/2") = [0.0 1.0; 1.0 0.0]

function build_obc_scar(s, v1, v2)
    L = length(s)
    @assert L % 2 == 0 "System size L must be even for exact scars."
    
    # Array has L+1 elements. Indices are 1 to L+1.
    links = [Index(2, "Link,l=$l") for l in 0:L]
    psi = MPS(s)
    
    for k in 1:2:L-1
        # links[k] maps to bond k-1, links[k+2] maps to bond k+1
        T = ITensor(ComplexF64, s[k], s[k+1], links[k], links[k+2])
        
        # State 1 = |1> (Rydberg), State 2 = |0> (Ground)
        T[s[k]=>2, s[k+1]=>2, links[k]=>1, links[k+2]=>2] = -1.0
        T[s[k]=>2, s[k+1]=>2, links[k]=>2, links[k+2]=>1] = 1.0
        T[s[k]=>2, s[k+1]=>1, links[k]=>1, links[k+2]=>1] = sqrt(2)
        T[s[k]=>1, s[k+1]=>2, links[k]=>2, links[k+2]=>2] = -sqrt(2)
        
        U, S, V = svd(T, (s[k], links[k]); lefttags="Link,l=$k")
        psi[k] = U
        psi[k+1] = S * V
    end
    
    # Safely contract boundary vectors at the edges using 1-based indexing
    vL = ITensor(ComplexF64, links[1])
    vL[links[1]=>1] = v1[1]
    vL[links[1]=>2] = v1[2]
    
    vR = ITensor(ComplexF64, links[L+1])
    vR[links[L+1]=>1] = v2[1]
    vR[links[L+1]=>2] = v2[2]
    
    psi[1] *= vL
    psi[L] *= vR
    
    normalize!(psi)
    return psi
end

function main_obc()
    L = 12
    t_total = 2.0
    dt = 0.01
    steps = round(Int, t_total / dt)
    
    s = siteinds("S=1/2", L)
    psi0 = build_obc_scar(s, [1.0, 1.0], [1.0, -1.0])
    psi_t = deepcopy(psi0)
    
    layers = []
    for layer in 0:2
        layer_gates = ITensor[]
        for i in (1+layer):3:L
            if i == 1
                h = op("X0", s[1]) * op("P0", s[2])
            elseif i == L
                h = op("P0", s[L-1]) * op("X0", s[L])
            else
                h = op("P0", s[i-1]) * op("X0", s[i]) * op("P0", s[i+1])
            end
            push!(layer_gates, exp(-1.0im * dt * h))
        end
        push!(layers, layer_gates)
    end
    
    println("--- TEBD Evolution: PXP Exact Scar (OBC) ---")
    for step in 0:steps
        t = step * dt
        if step > 0
            for layer_gates in layers
                psi_t = apply(layer_gates, psi_t; cutoff=1e-12, maxdim=256)
            end
            normalize!(psi_t)
        end
        
        fidelity = abs(inner(psi0, psi_t))^2
        println("t = $(round(t, digits=3)) | Fidelity = $(round(fidelity, digits=6))")
    end
end

main_obc()