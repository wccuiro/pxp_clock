using ITensors
using ITensorMPS
using LinearAlgebra

# Map local site states s ∈ 1..4 to ket and bra basis k,b ∈ 1..2
# 1 = |1> (Rydberg), 2 = |0> (Ground)
function map_s_to_kb(s)
    k = (s - 1) ÷ 2 + 1
    b = (s - 1) % 2 + 1
    return k, b
end

P(i,j)  = (i == 2 && j == 2) ? 1.0 : 0.0
X(i,j)  = (i != j) ? 1.0 : 0.0
Sp(i,j) = (i == 1 && j == 2) ? 1.0 : 0.0 
Sm(i,j) = (i == 2 && j == 1) ? 1.0 : 0.0 
I2(i,j) = (i == j) ? 1.0 : 0.0

H3(k1,k2,k3, j1,j2,j3)  = P(k1,j1) * X(k2,j2) * P(k3,j3)
Lp3(k1,k2,k3, j1,j2,j3) = P(k1,j1) * Sp(k2,j2) * P(k3,j3)
Lm3(k1,k2,k3, j1,j2,j3) = P(k1,j1) * Sm(k2,j2) * P(k3,j3)

H2L(k1,k2, j1,j2)  = X(k1,j1) * P(k2,j2)
Lp2L(k1,k2, j1,j2) = Sp(k1,j1) * P(k2,j2)
Lm2L(k1,k2, j1,j2) = Sm(k1,j1) * P(k2,j2)

H2R(k1,k2, j1,j2)  = P(k1,j1) * X(k2,j2)
Lp2R(k1,k2, j1,j2) = P(k1,j1) * Sp(k2,j2)
Lm2R(k1,k2, j1,j2) = P(k1,j1) * Sm(k2,j2)

function L_mat_el_3site(s1,s2,s3, s1_in,s2_in,s3_in, gamma_plus, gamma_minus)
    k1, b1 = map_s_to_kb(s1); k2, b2 = map_s_to_kb(s2); k3, b3 = map_s_to_kb(s3)
    j1, a1 = map_s_to_kb(s1_in); j2, a2 = map_s_to_kb(s2_in); j3, a3 = map_s_to_kb(s3_in)
    
    val = 0.0im
    val += -1.0im * H3(k1,k2,k3, j1,j2,j3) * I2(b1,a1)*I2(b2,a2)*I2(b3,a3)
    val -= -1.0im * I2(k1,j1)*I2(k2,j2)*I2(k3,j3) * H3(a1,a2,a3, b1,b2,b3)
    
    function dissipator(L)
        res = 0.0im
        res += L(k1,k2,k3, j1,j2,j3) * conj(L(b1,b2,b3, a1,a2,a3))
        LdL_kj = sum(conj(L(m1,m2,m3, k1,k2,k3)) * L(m1,m2,m3, j1,j2,j3) for m1 in 1:2, m2 in 1:2, m3 in 1:2)
        res -= 0.5 * LdL_kj * I2(b1,a1)*I2(b2,a2)*I2(b3,a3)
        LdL_ab = sum(conj(L(m1,m2,m3, a1,a2,a3)) * L(m1,m2,m3, b1,b2,b3) for m1 in 1:2, m2 in 1:2, m3 in 1:2)
        res -= 0.5 * I2(k1,j1)*I2(k2,j2)*I2(k3,j3) * LdL_ab
        return res
    end
    
    val += gamma_plus * dissipator(Lp3)
    val += gamma_minus * dissipator(Lm3)
    return val
end

function L_mat_el_2site(s1,s2, s1_in,s2_in, gamma_plus, gamma_minus, H, Lp, Lm)
    k1, b1 = map_s_to_kb(s1); k2, b2 = map_s_to_kb(s2)
    j1, a1 = map_s_to_kb(s1_in); j2, a2 = map_s_to_kb(s2_in)
    
    val = 0.0im
    val += -1.0im * H(k1,k2, j1,j2) * I2(b1,a1)*I2(b2,a2)
    val -= -1.0im * I2(k1,j1)*I2(k2,j2) * H(a1,a2, b1,b2) 
    
    function dissipator(L)
        res = 0.0im
        res += L(k1,k2, j1,j2) * conj(L(b1,b2, a1,a2))
        LdL_kj = sum(conj(L(m1,m2, k1,k2)) * L(m1,m2, j1,j2) for m1 in 1:2, m2 in 1:2)
        res -= 0.5 * LdL_kj * I2(b1,a1)*I2(b2,a2)
        LdL_ab = sum(conj(L(m1,m2, a1,a2)) * L(m1,m2, b1,b2) for m1 in 1:2, m2 in 1:2)
        res -= 0.5 * I2(k1,j1)*I2(k2,j2) * LdL_ab
        return res
    end
    
    val += gamma_plus * dissipator(Lp)
    val += gamma_minus * dissipator(Lm)
    return val
end

function make_gate_3site(s, i1, i2, i3, dt, gamma_plus, gamma_minus)
    L_mat = zeros(ComplexF64, 64, 64)
    for idx_out in 1:64, idx_in in 1:64
        s1 = (idx_out-1)%4+1; s2 = ((idx_out-1)÷4)%4+1; s3 = ((idx_out-1)÷16)%4+1
        s1_in = (idx_in-1)%4+1; s2_in = ((idx_in-1)÷4)%4+1; s3_in = ((idx_in-1)÷16)%4+1
        L_mat[idx_out, idx_in] = L_mat_el_3site(s1,s2,s3, s1_in,s2_in,s3_in, gamma_plus, gamma_minus)
    end
    U_mat = exp(dt * L_mat)
    
    T_gate = ITensor(ComplexF64, s[i1]', s[i2]', s[i3]', s[i1], s[i2], s[i3])
    for idx_out in 1:64, idx_in in 1:64
        s1 = (idx_out-1)%4+1; s2 = ((idx_out-1)÷4)%4+1; s3 = ((idx_out-1)÷16)%4+1
        s1_in = (idx_in-1)%4+1; s2_in = ((idx_in-1)÷4)%4+1; s3_in = ((idx_in-1)÷16)%4+1
        T_gate[s[i1]'=>s1, s[i2]'=>s2, s[i3]'=>s3, s[i1]=>s1_in, s[i2]=>s2_in, s[i3]=>s3_in] = U_mat[idx_out, idx_in]
    end
    return T_gate
end

function make_gate_2site(s, i1, i2, dt, gamma_plus, gamma_minus, H, Lp, Lm)
    L_mat = zeros(ComplexF64, 16, 16)
    for idx_out in 1:16, idx_in in 1:16
        s1 = (idx_out-1)%4+1; s2 = ((idx_out-1)÷4)%4+1
        s1_in = (idx_in-1)%4+1; s2_in = ((idx_in-1)÷4)%4+1
        L_mat[idx_out, idx_in] = L_mat_el_2site(s1,s2, s1_in,s2_in, gamma_plus, gamma_minus, H, Lp, Lm)
    end
    U_mat = exp(dt * L_mat)
    
    T_gate = ITensor(ComplexF64, s[i1]', s[i2]', s[i1], s[i2])
    for idx_out in 1:16, idx_in in 1:16
        s1 = (idx_out-1)%4+1; s2 = ((idx_out-1)÷4)%4+1
        s1_in = (idx_in-1)%4+1; s2_in = ((idx_in-1)÷4)%4+1
        T_gate[s[i1]'=>s1, s[i2]'=>s2, s[i1]=>s1_in, s[i2]=>s2_in] = U_mat[idx_out, idx_in]
    end
    return T_gate
end

# Vectorized Identity Operator |I>> for computing Tr(ρ)
function build_identity_mps(s)
    L = length(s)
    Id = MPS(s)
    for i in 1:L
        T = ITensor(ComplexF64, s[i])
        T[s[i]=>1] = 1.0 # |1><1|
        T[s[i]=>4] = 1.0 # |0><0|
        Id[i] = T
    end
    return Id
end

function build_obc_scar_vectorized(sites, v1, v2)
    L = length(sites)
    @assert L % 2 == 0 "System size L must be even for exact scars."
    links = [Index(4, "LinkVec,l=$l") for l in 0:L]
    psi = MPS(sites)
    
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
        U, S, V = svd(T_vec, (sites[k], links[k]); lefttags="LinkVec,l=$k")
        psi[k] = U; psi[k+1] = S * V
    end
    
    vL = ITensor(ComplexF64, links[1])
    for lvec in 1:4
        lk, lb = map_s_to_kb(lvec)
        vL[links[1]=>lvec] = v1[lk] * conj(v1[lb])
    end
    
    vR = ITensor(ComplexF64, links[L+1])
    for lvec in 1:4
        lk, lb = map_s_to_kb(lvec)
        vR[links[L+1]=>lvec] = v2[lk] * conj(v2[lb])
    end
    
    psi[1] *= vL; psi[L] *= vR
    
    # Enforce Tr(ρ) = 1 instead of Tr(ρ^2) = 1 (normalize!)
    Id_mps = build_identity_mps(sites)
    tr_rho = real(inner(Id_mps, psi))
    psi = (1.0 / tr_rho) * psi
    return psi
end

function main_obc()
    L = 20
    t_total = 20.0
    dt = 0.05
    steps = round(Int, t_total / dt)
    gamma_plus = 0.2
    gamma_minus = 0.001
    
    s = siteinds("Qudit", L; dim=4)
    Id_mps = build_identity_mps(s)
    psi0 = build_obc_scar_vectorized(s, [1.0, 1.0], [1.0, -1.0])
    psi_t = deepcopy(psi0)
    
    layers = []
    for layer in 0:2
        layer_gates = ITensor[]
        for i in (1+layer):3:L
            if i == 1
                push!(layer_gates, make_gate_2site(s, 1, 2, dt, gamma_plus, gamma_minus, H2L, Lp2L, Lm2L))
            elseif i == L
                push!(layer_gates, make_gate_2site(s, L-1, L, dt, gamma_plus, gamma_minus, H2R, Lp2R, Lm2R))
            else
                push!(layer_gates, make_gate_3site(s, i-1, i, i+1, dt, gamma_plus, gamma_minus))
            end
        end
        push!(layers, layer_gates)
    end
    
    println("--- Open Lindbladian Evolution: Vectorized Exact Scar (OBC) ---")
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

main_obc()