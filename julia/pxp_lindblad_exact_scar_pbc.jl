using ITensors
using ITensorMPS
using LinearAlgebra

# ... [COPY OPERATORS AND GATE FUNCTIONS FROM SCRIPT 1] ...

function build_pbc_scar_vectorized(sites)
    L = length(sites)
    @assert L % 6 == 0 "L must be a multiple of 6 for non-overlapping PBC 3-layer TEBD."
    links = [Index(4, "LinkVec,l=$l") for l in 0:L]
    
    T_list = ITensor[]
    for k in 1:2:L-1
        T_vec = ITensor(ComplexF64, sites[k], sites[k+1], links[k], links[k+2])
        T_ket = zeros(Float64, 2, 2, 2, 2)
        T_ket[2, 2, 1, 2] = -1.0
        T_ket[2, 2, 2, 1] = 1.0
        T_ket[2, 1, 1, 1] = sqrt(2)
        T_ket[1, 2, 2, 2] = -sqrt(2)
        
        for s1 in 1:4, s2 in 1:4
            k1, b1 = map_s_to_kb(s1); k2, b2 = map_s_to_kb(s2)
            for lvec_in in 1:4, lvec_out in 1:4
                lk_in, lb_in = map_s_to_kb(lvec_in)
                lk_out, lb_out = map_s_to_kb(lvec_out)
                val = T_ket[k1, k2, lk_in, lk_out] * conj(T_ket[b1, b2, lb_in, lb_out])
                if abs(val) > 1e-14
                    T_vec[sites[k]=>s1, sites[k+1]=>s2, links[k]=>lvec_in, links[k+2]=>lvec_out] = val
                end
            end
        end
        push!(T_list, T_vec)
    end
    
    last_T = T_list[end]
    last_T = replaceind(last_T, links[L+1] => links[1])
    T_list[end] = last_T
    
    full_psi = T_list[1]
    for k in 2:length(T_list)
        full_psi *= T_list[k]
    end
    
    psi = MPS(full_psi, sites; cutoff=1e-14, maxdim=256)
    
    # Enforce Tr(ρ) = 1
    Id_mps = build_identity_mps(sites)
    tr_rho = real(inner(Id_mps, psi))
    psi = (1.0 / tr_rho) * psi
    return psi
end

function main_pbc()
    L = 12
    t_total = 2.0
    dt = 0.05
    steps = round(Int, t_total / dt)
    gamma_plus = 0.2
    gamma_minus = 0.2
    
    s = siteinds("Qudit", L; dim=4)
    Id_mps = build_identity_mps(s)
    psi0 = build_pbc_scar_vectorized(s)
    psi_t = deepcopy(psi0)
    
    layers = []
    for layer in 0:2
        layer_gates = ITensor[]
        for i in (1+layer):3:L
            left = (i == 1) ? L : i - 1
            right = (i == L) ? 1 : i + 1
            push!(layer_gates, make_gate_3site(s, left, i, right, dt, gamma_plus, gamma_minus))
        end
        push!(layers, layer_gates)
    end
    
    println("--- Open Lindbladian Evolution: Vectorized Exact Scar (PBC) ---")
    for step in 0:steps
        t = step * dt
        if step > 0
            for layer_gates in layers
                psi_t = apply(layer_gates, psi_t; cutoff=1e-12, maxdim=256)
            end
            # Enforce trace normalization after evolution step
            tr_rho = real(inner(Id_mps, psi_t))
            psi_t = (1.0 / tr_rho) * psi_t
        end
        
        fidelity = real(inner(psi0, psi_t))
        println("t = $(round(t, digits=3)) | Fidelity = $(round(fidelity, digits=6)) | Trace = $(real(inner(Id_mps, psi_t)))")
    end
end

main_pbc()