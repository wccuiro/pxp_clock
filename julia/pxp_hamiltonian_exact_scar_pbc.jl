using ITensors
using ITensorMPS
using LinearAlgebra

# P projects onto |0> (Spin Down, which is state 2 in ITensor S=1/2)
ITensors.op(::OpName"P0", ::SiteType"S=1/2") = [0.0 0.0; 0.0 1.0]
ITensors.op(::OpName"X0", ::SiteType"S=1/2") = [0.0 1.0; 1.0 0.0]

function build_pbc_scar(s)
    L = length(s)
    @assert L % 6 == 0 "L must be a multiple of 6 for PBC exact scars with a non-overlapping 3-layer TEBD."
    
    links = [Index(2, "Link,l=$l") for l in 0:L]
    
    T_list = ITensor[]
    for k in 1:2:L-1
        # links[k] maps to bond k-1, links[k+2] maps to bond k+1
        T = ITensor(ComplexF64, s[k], s[k+1], links[k], links[k+2])
        
        # State 1 = |1> (Rydberg), State 2 = |0> (Ground)
        T[s[k]=>2, s[k+1]=>2, links[k]=>1, links[k+2]=>2] = -1.0
        T[s[k]=>2, s[k+1]=>2, links[k]=>2, links[k+2]=>1] = 1.0
        T[s[k]=>2, s[k+1]=>1, links[k]=>1, links[k+2]=>1] = sqrt(2)
        T[s[k]=>1, s[k+1]=>2, links[k]=>2, links[k+2]=>2] = -sqrt(2)
        
        push!(T_list, T)
    end
    
    # Impose periodic boundary trace safely using 1-based array indices
    last_T = T_list[end]
    last_T = replaceind(last_T, links[L+1] => links[1])
    T_list[end] = last_T
    
    full_psi = T_list[1]
    for k in 2:length(T_list)
        full_psi *= T_list[k]
    end
    
    psi = MPS(full_psi, s; cutoff=1e-14, maxdim=256)
    normalize!(psi)
    return psi
end

function main_pbc()
    L = 12
    t_total = 2.0
    dt = 0.01
    steps = round(Int, t_total / dt)
    
    s = siteinds("S=1/2", L)
    psi0 = build_pbc_scar(s)
    psi_t = deepcopy(psi0)
    
    layers = []
    for layer in 0:2
        layer_gates = ITensor[]
        for i in (1+layer):3:L
            left = (i == 1) ? L : i - 1
            right = (i == L) ? 1 : i + 1
            h = op("P0", s[left]) * op("X0", s[i]) * op("P0", s[right])
            push!(layer_gates, exp(-1.0im * dt * h))
        end
        push!(layers, layer_gates)
    end
    
    println("--- TEBD Evolution: PXP Exact Scar (PBC) ---")
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

main_pbc()