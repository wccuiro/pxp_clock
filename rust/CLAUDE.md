# PXP Lindbladian: exact diagonalization and exact time evolution (Rust)

Physics project: open PXP chain (periodic, L sites, Rydberg constraint n_j n_{j+1} = 0) with
PXP-dressed σ± jumps. Goal: exact (no approximate algorithms) spectra, Néel-state overlaps,
operator entanglement and dynamics, up to L = 14 routinely and L = 16 on the cluster.

    L[ρ] = -iΩ [H, ρ] + Σ_{γ∈{+,-}} γ Σ_j ( A^γ_j ρ A^γ_j† - ½ {A^γ_j† A^γ_j, ρ} )
    H = Σ_j P_{j-1} X_j P_{j+1},  A^+_j = P_{j-1} σ^+_j P_{j+1},  A^-_j = P_{j-1} σ^-_j P_{j+1}

The user wants **exact methods only** (dense diagonalization, matrix exponential to machine
precision). Never propose truncated Krylov/Arnoldi partial spectra, Euler steps, etc. as replacements.

## Layout and commands

```
Cargo.toml              one crate, two binaries, autobins = false
.cargo/config.toml      -C target-cpu=native (compile on the node type you run on)
src/common.rs           shared, included textually with include!("../common.rs"):
                        Chain, Model, Sector, Block (symmetry blocks), dense + sparse builders, Csr
src/bin/lindblad.rs     diagonalization (LAPACK dgeev via MKL), OEE, steady state
src/bin/evolution.rs    time evolution of the Néel state (pure Rust, no LAPACK)
tools/                  independent Python references + comparison scripts (see Validation)
```

```bash
cargo run --release --bin lindblad  -- [L] [gp gm omega]          # no params: grid in Config::default_grid
cargo run --release --bin lindblad_evol -- [L] [T] [dt] [gp gm omega]
# without MKL (testing): cargo build --release --no-default-features --features netlib
```
Threads: `MKL_NUM_THREADS` (eigensolver), `RAYON_NUM_THREADS` (matrix build, OEE, evolution).
Flags at the top of lindblad.rs: `COMPUTE_OEE`, `COMPUTE_STEADY_STATE`, `COMPUTE_CONDITION_NUMBERS`.

## Conventions (do not change silently; outputs and the analysis depend on them)

- Configurations: u64 bit strings, PBC Lucas basis (no two adjacent 1s, including bits 0 and L-1).
  T = cyclic shift left by one bit. Representative = smallest member of the orbit, period p.
- Momentum states: |r,k> = (1/√p) Σ_{d<p} e^{-i2πkd/L} T^d |r>, allowed iff k·p ≡ 0 (mod L).
- Q sectors (the user's generalized translation vectorization, ρ_Q = Σ_k P_k ρ P_{k-Q}):
  basis |n,k><m,k-Q|. Only Q = 0 and Q = L/2 are computed (they contain the Néel state).
- Real-space expansion of a sector vector: the coefficient of |T^j n><T^j' m| is
  e^{i2π(-k j + (k-Q) j')/L} / (√p_n √p_m). (An old version used +k on both sides and ignored Q; that was a bug.)
- Néel initial vector in the sector basis: weight ½ on |N,k><N,k-Q| for the two allowed k (correct).

## Exact symmetries in use (all verified numerically)

1. Translation → Q sectors (above).
2. Reflection R: j → -j mod L. R|n,k> = e^{-i2πk m_n/L} |r_n,-k>, where reflect(n) = T^{m_n} r_n.
3. S: ρ → C ρᵀ C with C = Π_j Z_j. Exact because H is real with CHC = -H and the jumps are
   real with CAC = -A (the transpose flips the sign of the Hamiltonian part, C flips it back).
   S|n,k><m,k-Q| = (-1)^{|n|+|m|} |m,Q-k><n,-k|  (no momentum phase).
4. Hermiticity Θ: ρ → ρ† (antilinear). In a Hermitian operator basis every block is a REAL matrix.
   Θ|n,k><m,k-Q| = |m,k-Q><n,k| (no phase).

Blocks = characters (σ, τ) of the group {1, R, S, RS}, built from group orbits in `Block::new`, then
made real with Θ (pairs → (e + φe')/√2, i(e - φe')/√2; fixed vectors → e·e^{i arg φ/2}).
The Néel state, the trace and the observables n, nn live only in (σ,τ) = (+,+); other blocks have
c_k = o_k = 0 exactly. S breaks if a detuning Δ Σ n_j or complex jumps are added.
A commutant computation (L = 6, 8, 10) found no further symmetry valid for all (Ω, γ+, γ-).

Block sizes (+,+)/(+,-)/(-,+)/(-,-):
L=14 Q=0: 13347/12864/12127/12487, Q=7: 13033/13033/12319/12319;
L=16 Q=0: 77812/76566/74618/75579, Q=8: 77149/77008/75279/75138.

## Algorithms

- lindblad: per block, real dense matrix → `dgeev` (right eigenvectors only). Néel coefficients by
  solving VR·x = ρ0 with LU (`dgetrf/dgetrs`), never by forming VR⁻¹ and never by diagonalizing L†
  and pairing eigenvalues by sorting (that breaks for degenerate/conjugate eigenvalues).
  dgeev complex pairs: columns (u, w) → v = u ± i w; c = (x_j ∓ i x_{j+1})/2.
- OEE of ρ_k†ρ_k: dense real-space C (843×843 at L=14), S = C^H C, reshape to the A|B Schmidt
  matrix, singular values. Done with **faer** (pure Rust) inside rayon.
- Steady state: eigenvector with λ = 0, normalized by its trace before the Hermitian eigensolve.
- evolution: sparse CSR (+,+) block, exp(Δt·L)v with the scaling-and-Taylor algorithm of
  Al-Mohy & Higham (θ_m table as in scipy expm_multiply, tolerance 2^-53, trace shift μ).
  No trace renormalization (trace is conserved; |ΔTr| is printed as a check).

## Outputs

lindblad: eigenvalues.csv, decay.csv (per mode: Re λ, Im λ, c, o, w; w_k = c_k Tr(n r_k) so that
<n>(t) = Σ w_k e^{λ_k t}; c and o individually carry an arbitrary LAPACK phase, |c|, |o|, c·conj(o)
and w are phase-invariant), oee.csv, std_eigenvalues.csv, occupation.csv, optional cond.csv.
Néel return probability: F(t) = Σ_k c_k conj(o_k) e^{λ_k t} summed over both Q sectors.
evolution: occupation_time.csv, one line per parameter point: gp,gm,omega, then n, nn, F at t = i·dt.

## Validation (rerun after any change to common.rs or the builders)

```bash
python3 tools/reference_diag.py 8 0.2 0.001 1.0 ref8.json
(mkdir -p run8 && cd run8 && ../target/release/lindblad 8 0.2 0.001 1.0)
python3 tools/compare_diag.py ref8.json run8        # expect eigenvalues ~1e-10, |c|,|o| ~1e-10, OEE ~1e-6 (print precision)
python3 tools/reference_evolution.py 8 25 0.01 0.2 0.001 1.0 ref8.npy
(mkdir -p evo8 && cd evo8 && ../target/release/evolution 8 25 0.01 0.2 0.001 1.0)
python3 tools/compare_evolution.py ref8.npy evo8/occupation_time.csv   # expect ~1e-14
```
Built-in checks printed at runtime: Σ c·Tr(r) = 1 and Σ w = 0.5 in Q=0 (Néel), 0 in Q=π;
"max dropped Im" ~1e-15; blocks cover the sector (asserted).
Always match eigenvalues between runs by assignment (Hungarian), never by sorting both lists.
Reference results: L=14, gp=0.2, gm=0.001, Ω=1: steady <n> = 0.2944017632, <nn> = 0.1283994477;
the evolution at L=14 agrees with the eigen-expansion of the diagonalization to 7e-11 (F).

## Performance (measured)

- L=14 lindblad before S (2 blocks of ~26k per sector, 112-core Leonardo node): eig 2.0 h, OEE ~70 min, total 3.2 h.
  With S the eigensolves should drop ~4× (n³ scaling). OEE (~101k modes × ~1 s) is now the bottleneck.
- L=14 evolution (T=25, dt=0.01): 76 s on 2 cores; L=12: ~10 s.
- Peak RAM lindblad ≈ 2 × 8 × n_max² bytes (+ ~90 MB per thread for OEE at L=14).

## Pitfalls already hit (do not reintroduce)

- **Never call MKL from rayon threads**: concurrent MKL calls segfault with the static Intel OpenMP
  runtime (crashed in the self-test with 16 threads). MKL is used only from the main thread; anything
  parallel uses faer or plain Rust. `lapack_self_test()` runs at startup.
- ndarray-linalg's `intel-mkl-static` feature is the SEQUENTIAL MKL; MKL is linked via
  `intel-mkl-src/mkl-static-lp64-iomp` instead. Only `lindblad` links it; `evolution` must stay LAPACK-free.
- `libiomp5.so: cannot open shared object file`: `python3 -m pip install --user intel-openmp` and
  `export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH` (or load the cluster's Intel module).
- LAPACK integers are LP64 (`type Int = i32`); L=16 blocks need ILP64 MKL
  (`mkl-static-ilp64-iomp`) and `type Int = i64` (n² > 2³¹).
- Slurm: run the binary directly (no `srun`, or `srun --cpus-per-task=112`); one process with
  threads; `--cpus-per-task=112`, `MKL_NUM_THREADS=112`, `RAYON_NUM_THREADS=112`.

## Open items

- Per-block CLI option (`--sector Q --block στ`) + merge script, to run the 8 blocks of one
  parameter point on separate nodes (L=14 ~20-30 min wall; L=16 Néel blocks ~10-15 h each, ~100 GB).
- ILP64 switch for L=16.
- Optional phase fixing of eigenvectors (e.g. o_k real positive) for reproducible raw c_k, o_k.
- OEE speed (memory-bandwidth bound with 112 concurrent SVDs; try fewer rayon threads).