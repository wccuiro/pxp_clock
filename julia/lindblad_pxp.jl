# itensor_lindblad_mpo_nodephasing_fix_checked.jl
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using Random
const im = complex(0,1)

# ---------------- Parameters ----------------
L_min = 4
L_max = 6
Ω = 1.0
γ_p = 0.3
γ_m = 0.7

# TN truncation / Krylov tunables
maxdim_apply = 600
cutoff_apply  = 1e-9
krylov_dim    = 30

# ---------------- Local (2x2) operators ----------------
sx = [0.0 1.0; 1.0 0.0]
sp = [0.0 1.0; 0.0 0.0]
sm = [0.0 0.0; 1.0 0.0]
P  = [1.0 0.0; 0.0 0.0]
I2 = Matrix{Float64}(I,2,2)

# 2x2 -> 4x4 local superoperators (vectorized ordering)
left_super(A)  = kron(A, I2)            # A ⊗ I
right_super(A) = kron(I2, transpose(A)) # I ⊗ A^T
identity4 = kron(I2, I2)

# ---------------- Utilities ----------------
function make_local4_list_for_three_site(L, i, mats2::Vector{Array{Float64,2}})
    # Correct periodic 3-site indices: i-1, i, i+1 (1-based)
    idxs = [ ((i - 2) % L) + 1, ((i - 1) % L) + 1, (i % L) + 1 ]
    local4 = [ copy(identity4) for _ in 1:L ]
    for (k, site) in enumerate(idxs)
        local4[site] = mats2[k]
    end
    return local4
end

function product_mpo_from_local4(sites::Vector{<:Index}, local4::Vector{Array{Float64,2}})
    L = length(sites)
    links = [ Index(1, "link,$i") for i in 1:(L-1) ]
    W = MPO(sites)
    for n in 1:L
        leftlink  = n == 1 ? Index(1,"l0") : links[n-1]
        rightlink = n == L ? Index(1,"rL") : links[n]
        T = ITensor(leftlink, rightlink, prime(sites[n]), sites[n])
        op4 = local4[n]
        @assert size(op4) == (4,4) "op4 has wrong size at site $n: $(size(op4))"
        # ensure complex element type
        if eltype(op4) != ComplexF64
            op4 = ComplexF64.(op4)
        end
        for a in 1:4, b in 1:4
            T[leftlink => 1, rightlink => 1, prime(sites[n]) => a, sites[n] => b] = op4[a,b]
        end
        W[n] = T
    end
    return W
end

function sum_mpos(mpo_list::Vector{MPO})
    @assert !isempty(mpo_list)
    S = deepcopy(mpo_list[1])
    for k in 2:length(mpo_list)
        S = S + mpo_list[k]
    end
    return S
end

function apply_mpo_to_mps(mpo::MPO, psi::MPS; maxdim=maxdim_apply, cutoff=cutoff_apply)
    return apply(mpo, psi; maxdim=maxdim, cutoff=cutoff)
end

function normalize_mps!(psi::MPS)
    nrm = sqrt(real(inner(psi, psi)))
    @assert nrm > 0 "normalize: norm <= 0"
    psi .= psi / nrm
    return psi
end

function maximally_mixed_mps(sites::Vector{Index})
    L = length(sites)
    vecI = zeros(ComplexF64, 4)
    vecI[1] = 1.0; vecI[4] = 1.0
    psi = MPS()
    for n in 1:L
        A = ITensor(sites[n])
        for a in 1:4
            A[sites[n] => a] = vecI[a]
        end
        psi[n] = A
    end
    normalize_mps!(psi)
    return psi
end

function mps_to_density_matrix(psi::MPS)
    vec = dense(psi)
    L = length(psi)
    dim = 2^L
    @assert length(vec) == 4^L "dense(psi) length mismatch: $(length(vec)) != 4^$L"
    ρ = reshape(vec, (dim, dim))
    ρ = (ρ + ρ')/2
    trval = real(tr(ρ))
    if abs(trval) > 1e-16
        ρ ./= trval
    else
        @warn "trace near zero when converting MPS->rho"
    end
    return ρ
end

# ---------------- Arnoldi (matrix-free) on MPS ----------------
function arnoldi_on_mps(applyA, v0::MPS; m::Int=krylov_dim, maxdim=maxdim_apply, cutoff=cutoff_apply)
    V = Vector{MPS}(undef, m+1)
    H = zeros(ComplexF64, m+1, m)
    V[1] = deepcopy(v0)
    normalize_mps!(V[1])
    for j in 1:m
        w = applyA(V[j])
        for i in 1:j
            Hij = inner(V[i], w)
            H[i,j] = Hij
            w = w - V[i] * Hij
            w = compress(w; maxdim=maxdim, cutoff=cutoff)
        end
        hnorm = sqrt(real(inner(w, w)))
        H[j+1, j] = hnorm
        if hnorm < 1e-14
            Hsmall = H[1:j, 1:j]
            evs, evecs = eigen(Hsmall)
            idx = argmin(abs.(evs))
            y = evecs[:, idx]
            x = V[1] * y[1]
            for ii in 2:j
                x = x + V[ii] * y[ii]
            end
            x = compress(x; maxdim=maxdim, cutoff=cutoff)
            normalize_mps!(x)
            return evs[idx], x
        end
        V[j+1] = w / hnorm
    end
    Hm = H[1:m, 1:m]
    evs, evecs = eigen(Hm)
    idx = argmin(abs.(evs))
    y = evecs[:, idx]
    x = V[1] * y[1]
    for ii in 2:m
        x = x + V[ii] * y[ii]
    end
    x = compress(x; maxdim=maxdim, cutoff=cutoff)
    normalize_mps!(x)
    return evs[idx], x
end

# ---------------- Main loop ----------------
Random.seed!(1234)

for L in L_min:L_max
    @printf("\n=== L = %d ===\n", L)
    sites = [ Index(4, "s,$i") for i in 1:L ]

    mpo_terms = MPO[]

    # Hamiltonian commutator -i(H⊗I - I⊗H^T)
    for i in 1:L
        mats2 = [P, sx, P]
        local4_left  = make_local4_list_for_three_site(L, i, [left_super(m)  for m in mats2])
        local4_right = make_local4_list_for_three_site(L, i, [right_super(m) for m in mats2])
        push!(mpo_terms, product_mpo_from_local4(sites, local4_left) * Ω)
        push!(mpo_terms, product_mpo_from_local4(sites, local4_right) * (-Ω))
    end
    L_mpo = sum_mpos(mpo_terms)
    # safer: assign the complex scaled MPO
    L_mpo = L_mpo * (-im)
    empty!(mpo_terms)

    # Dissipators (three-site jumps), no dephasing
    for i in 1:L
        matsC_sp = [P, sp, P]
        matsC_sm = [P, sm, P]

        local4_C_sp = make_local4_list_for_three_site(L, i, [kron(m, conj(m)) for m in matsC_sp])
        local4_C_sm = make_local4_list_for_three_site(L, i, [kron(m, conj(m)) for m in matsC_sm])

        matsCdagC_sp = [adjoint(m)*m for m in matsC_sp]
        matsCdagC_sm = [adjoint(m)*m for m in matsC_sm]

        local4_CdagC_left_sp  = make_local4_list_for_three_site(L, i, [left_super(m)  for m in matsCdagC_sp])
        local4_CdagC_right_sp = make_local4_list_for_three_site(L, i, [right_super(m) for m in matsCdagC_sp])

        local4_CdagC_left_sm  = make_local4_list_for_three_site(L, i, [left_super(m)  for m in matsCdagC_sm])
        local4_CdagC_right_sm = make_local4_list_for_three_site(L, i, [right_super(m) for m in matsCdagC_sm])

        c_pref_sp = sqrt(γ_p)
        c_pref_sm = sqrt(γ_m)

        push!(mpo_terms, product_mpo_from_local4(sites, local4_C_sp) * (2.0 * c_pref_sp^2))
        push!(mpo_terms, product_mpo_from_local4(sites, local4_C_left_sp) * (-c_pref_sp^2))
        push!(mpo_terms, product_mpo_from_local4(sites, local4_CdagC_right_sp) * (-c_pref_sp^2))

        push!(mpo_terms, product_mpo_from_local4(sites, local4_C_sm) * (2.0 * c_pref_sm^2))
        push!(mpo_terms, product_mpo_from_local4(sites, local4_CdagC_left_sm) * (-c_pref_sm^2))
        push!(mpo_terms, product_mpo_from_local4(sites, local4_CdagC_right_sm) * (-c_pref_sm^2))
    end

    # Sum dissipators onto L_mpo (non in-place)
    L_mpo = L_mpo + sum_mpos(mpo_terms)

    # Define applyA: apply Liouvillian MPO to MPS
    function applyA(mps::MPS)
        y = apply_mpo_to_mps(L_mpo, mps; maxdim=maxdim_apply, cutoff=cutoff_apply)
        return y
    end

    # initial MPS (product vec(I))
    v0 = maximally_mixed_mps(sites)

    @printf("Running Arnoldi (krylov_dim=%d, maxdim_apply=%d) ...\n", krylov_dim, maxdim_apply)
    ritz_val, ritz_vec = arnoldi_on_mps(applyA, v0; m=krylov_dim, maxdim=maxdim_apply, cutoff=cutoff_apply)

    @printf("Ritz eigenvalue ~= %.6e + %.6ei\n", real(ritz_val), imag(ritz_val))

    # Convert to dense density matrix for entropy (only reasonable for small L)
    rho = mps_to_density_matrix(ritz_vec)
    # von Neumann entropy (base 2)
    vals = eigen(Hermitian(Matrix((rho + rho')/2))).values
    probs = real.(clamp.(vals, 0.0, 1.0))
    probs = probs[probs .> 0.0]
    S = -sum(probs .* log2.(probs))

    @printf("L=%d  Von Neumann entropy (full state) = %.8f bits\n", L, S)
end

println("Finished (no single-site dephasing).")
