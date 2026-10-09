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
Cargo.toml                         one crate; every src/bin/<name>/main.rs is a binary
config.toml                        -C target-cpu=native, but NOT read by cargo (it is not in .cargo/)
src/common.rs                      shared, included textually with include!("../../common.rs"):
                                   Chain, Model, Sector, Block (symmetry blocks), dense + sparse builders, Csr
src/bin/lindblad/main.rs           diagonalization (LAPACK dgeev via MKL), OEE, steady state
src/bin/lindblad_evol/main.rs      time evolution of the Néel state (pure Rust, no LAPACK)
src/bin/lindblad_alpha/main.rs     the same two for the partial projection (alpha) model,
src/bin/lindblad_alpha_evol/main.rs  full 2^L basis (see the alpha section)
src/bin/lindblad_staggered/main.rs       the same two for staggered jumps (σ+ on one sublattice,
src/bin/lindblad_staggered_evol/main.rs  σ- on the other), see the staggered section
src/bin/lindblad_asymmetric*, trajectories   old code, not on common.rs; the asymmetric ones need `sprs`,
                                   which is not in Cargo.toml, so always build with --bin <name>
tools/                             independent Python references + comparison scripts (see Validation)
ALPHA_NOTES.md                     evaluation, measurements and state of the alpha work (2026-10-09)
```

```bash
cargo run --release --bin lindblad  -- [L] [gp gm omega]          # no params: grid in Config::default_grid
cargo run --release --bin lindblad_evol -- [L] [T] [dt] [gp gm omega]
cargo run --release --bin lindblad_alpha      -- [L] [gp gm omega alpha]            # partial projection, see below
cargo run --release --bin lindblad_alpha_evol -- [L] [T] [dt] [gp gm omega alpha]
cargo run --release --bin lindblad_staggered      -- [L] [gp gm omega] [plus_site]          # staggered jumps, see below
cargo run --release --bin lindblad_staggered_evol -- [L] [T] [dt] [gp gm omega] [plus_site]
cargo build --release --bin lindblad --bin lindblad_evol --bin lindblad_alpha --bin lindblad_alpha_evol \
                      --bin lindblad_staggered --bin lindblad_staggered_evol
# without MKL (testing): cargo build --release --no-default-features --features netlib
```
Threads: `MKL_NUM_THREADS` (eigensolver), `RAYON_NUM_THREADS` (matrix build, OEE, evolution).
Flags at the top of src/bin/lindblad/main.rs and src/bin/lindblad_alpha/main.rs: `COMPUTE_OEE`, `COMPUTE_STEADY_STATE`, `COMPUTE_CONDITION_NUMBERS`.

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

## Partial projection (alpha) model: lindblad_alpha, lindblad_alpha_evol

H keeps the strict blockade; the jumps use A^±_j = P^α_{j-1} σ^±_j P^α_{j+1} with
P^α = [(1+α)|0><0| + (1-α)|1><1|] / (1+|α|). For α < 1 adjacent excitations are created, so the
full 2^L basis is used: `Chain::new(l, false)`, `Model::new(&ch, Some(α))` (the constrained binaries
call `Chain::new(l, true)`, `Model::new(&ch, None)`). The jumps stay real with CAC = -A, so the same
symmetries, blocks and algorithms apply; the two binaries are copies of lindblad / lindblad_evol
with the alpha grid. The Model is rebuilt for every alpha (cheap).

- Outputs: *_alpha.csv with rows q,gp,gm,omega,alpha,... (decay_alpha.csv has the same 8 columns
  per mode as decay.csv, last pair = w); occupation_time_alpha.csv rows alpha,gp,gm,omega,(n,nn,F)*.
  A single point from the command line gives names ending in _alpha_L.._gp.._gm.._omega.._alpha...
- Block sizes (+,+)/(+,-)/(-,+)/(-,-): L=8 Q=0: 2299/2136/1851/1944, Q=4: 2169/2136/1977/1944;
  L=10 Q=0: 27190/26574/25398/25806, Q=5: 26574/26574/25806/25806; L=12 (+,+): 353384 / 351324.
  Dense diagonalization stops at L=10 (cluster); L=12 only for the evolution, memory not measured.
- α = 1 in the full basis is fragmented: a site only flips if both neighbours are empty, so
  configurations with adjacent excitations are frozen (L=6: 29 disconnected sets, 49 zero modes
  in Q=0). The Néel state only sees the constrained set. lindblad_alpha therefore writes ONE steady
  state per point, ρ∞ = Σ_{λ_k=0} c_k r_k (the long-time limit of the Néel state, equal to the
  steady state when it is unique; at α = 1 it equals the constrained steady state, checked at L=6).
  Never write the single zero eigenvectors: LAPACK returns an arbitrary basis of the zero eigenspace.
- With γ- = 0.001 many eigenvalues are nearly degenerate (gaps down to 1e-8 and below), so
  individual c_k, o_k, OEE of those modes are only determined to ~ eps |L| / gap. At generic
  parameters (L=6) nothing is degenerate inside or between blocks: no further symmetry for 0 ≤ α < 1.
- |c_k| is large for some modes (340 at L=8): decay*.csv (10 decimals) gives c conj(o) to |c|·5e-11.
- Validated against the Python references (numbers in ALPHA_NOTES.md): L=4 and L=6 at four points
  with α = 0, 0.4, 0.8, 1: eigenvalues ≤ 2e-13, |c|, |o|, c conj(o), w ≤ 8e-11, OEE at print
  precision in both sectors, steady state ≤ 3e-12, evolution ≤ 2e-13. L=8 (gp 0.2, gm 0.001, α 0.4):
  eigenvalues 4e-13, |c|, c conj(o), w ≤ 4e-10 (relative, gap-weighted as in compare_diag.py),
  steady state 5e-15; evolution vs the eigen-expansion of the reference: n 9e-13, F 8e-15.
- Performance: default L=6 grid (24 points) 7 s (diagonalization with OEE) and 6 s (evolution);
  L=8 one point 61 s / 160 MB (8 threads); evolution T=25, dt=0.01: L=8 13 s, L=10 199 s / 278 MB
  (nnz 7.8e6 per block); L=12 evolution not measured (stopped at 5.7 GB during the build).
- The old alpha binaries (git history before this change) had the Q = π real-space expansion bug in
  the OEE, RK4 time stepping with 1e-10 thresholds, and paired left/right eigenvectors by sorting.

## Staggered jumps: lindblad_staggered, lindblad_staggered_evol (2026-10-09)

σ+ jumps (P σ+ P, rate γ+) only on the sublattice j ≡ plus_site (mod 2), σ- jumps (rate γ-) only on
the other one; H unchanged. The Néel state (bits 0, 2, ...) occupies the sublattice 0.
- plus_site = 1 (default, `PLUS_SITE`): σ- on the Néel-occupied sites, σ+ on the others. This is the
  convention of the old lindblad_asymmetric binaries; the shifted Néel state is dark for the jumps.
- plus_site = 0: σ+ on the Néel-occupied sites (julia/pxp_lindblad_staggered.jl); the Néel state is dark.
- The two are the same Lindbladian shifted by one site: same spectrum, different Néel dynamics.
- Symmetries (checked on the full Lindbladian, L = 6, 8, 10, `tools/check_symmetries.py ... plus=1`):
  T (one site) and the bond-centred reflections are LOST (residual ~γ); T², R (j -> -j), S and Θ hold.
  No further symmetry: no degeneracy inside or between the four blocks at generic parameters.
- common.rs: `Chain::with_step(l, true, 2)` (orbits, momenta k = 0..L/2-1 and phases of U = T²;
  `ch.step`, `ch.nt = L/step`; every momentum loop runs over nt, `ch.l` stays the number of sites) and
  `Model::staggered(&ch, plus_site)` (one jump table per sublattice, H = Σ_u U^u (h_0 + h_1) U^-u).
  `Chain::new` / `Model::new` are the step = 1 case; the uniform lindblad_evol output is byte-identical
  to the binary built before the change (L = 8, 10 grid, 12).
- The Néel state is T²-invariant: ONE sector, Q = 0 of T², = the old Q = 0 ⊕ Q = L/2. Its (+,+) block
  holds the Néel state, the trace, n and nn; block sizes (+,+)/(+,-)/(-,+)/(-,-) are the sums of the
  two old sectors: L=8 203/166/95/105, L=10 906/831/624/672, L=12 4736/4519/3997/4102,
  L=14 26380/25897/24446/24806; L=16 (+,+) = 154961. Dense diagonalization: one block of twice the
  size, i.e. ~4× the time and RAM of `lindblad` (L=14: 2 × 8 × 26380² = 11 GB); L=16 needs ILP64.
- Outputs: *_staggered.csv, rows q,gp,gm,omega,... with q = 0 (single point from the command line:
  names end in _staggered_L.._gp.._gm.._omega..); occupation_time_staggered.csv rows gp,gm,omega,(n,nn,F)*.
  plot_scripts/sym_vs_asym_evol.py still reads the old occupation_time_asymmetric.csv.
- Validation against the Python references (test builds, see below; the release binaries had not
  been built yet): lindblad_staggered_evol at L = 4, 6, 8, 10, both plus_site, five parameter points:
  n, nn, F ≤ 2e-14. lindblad_staggered at L = 6, 8 (both plus_site) and L = 10 (plus_site = 1):
  eigenvalues ≤ 2e-13, |c|, |o|, c conj(o), w ≤ 7e-11, OEE 5e-7 (print precision), steady state 5e-16.
  The uniform `lindblad` rebuilt from the changed common.rs still passes compare_diag at L = 8.
  ```bash
  python3 tools/check_symmetries.py 8 0.2 0.001 1.0 plus=1          # also: sizes 10 staggered
  python3 tools/reference_evolution.py 8 25 0.01 0.2 0.001 1.0 ref8s.npy plus=1
  python3 tools/compare_evolution.py ref8s.npy occupation_time_staggered.csv
  python3 tools/reference_diag.py 8 0.2 0.001 1.0 ref8s.json plus=1
  python3 tools/compare_diag.py ref8s.json run_dir                  # run: lindblad_staggered 8 0.2 0.001 1.0 1
  ```
- Performance of lindblad_staggered_evol (T=15, dt=0.01, 4 threads): L=12 1.3 s / 11 MB (nnz 0.13 M),
  L=14 6 s / 65 MB (nnz 1.33 M): cheaper than the uniform model (half the jump channels).
- Physics (L ≤ 8, exact, target/claude_tmp/staggered/revivals*.py, stag_mag.py): both conventions
  raise the Néel fidelity revivals over the uniform model, but only as much as the uniform model with
  halved rates does (within 1-3%); plus_site = 1 has the slightly higher first revival, plus_site = 0
  the slightly higher late-time fidelity. The staggered magnetization shows the same.
  Search over local static variants (local_search.py, local_long.py, local_control.py, L = 6, 8):
  the sublattice never matters (a jump on N, on O, or on all sites at half the rate give the same F
  within ~1%), σ- costs more than σ+ at equal rate, a staggered field in H does not help, and no
  variant exceeds the closed-system revivals. Jumps conditioned on the next-nearest neighbours so
  that they vanish on both Néel states do far less damage: "heal" n_{j-2} Pσ+_jP n_{j+2} (first
  revival 0.76 at rate 0.2, L=8, against 0.22 for plain staggered and 0.90 closed) and "clean"
  (1-n_{j-2}) Pσ-_jP (1-n_{j+2}) (0.71). Not implemented in Rust.
  This gain is trivial too (the user's point, confirmed with equal_jumps.py): at an EQUAL NUMBER OF
  JUMP EVENTS up to the revival, J = ∫ Σ γ Tr(A†A ρ) dt, heal and plain σ+ give the same fidelity
  (L=8: 0.760/0.615 against 0.771/0.612 at J = 0.50), and clean is only modestly better than plain
  σ- (0.705/0.520 against 0.646/0.430). To leading order F at a revival depends only on J.
- Test builds that are safe on the workstation (throwaway crates under target/claude_tmp/staggered/,
  sources are symlinks to src/, run with `ulimit -v 2500000` and `cargo build --release --offline -j 1`):
  `light/` (rayon + num-complex only; lindblad_evol, lindblad_staggered_evol; 3 s, 0.3 GB) and
  `light_diag/` (lindblad, lindblad_staggered; LAPACK from anaconda's libmkl_rt.so through build.rs,
  dependencies at opt-level 1, no LTO; 2 min, peak 1.0 GB; run with MKL_NUM_THREADS=2).
  Use them instead of the static-MKL fat-LTO release build to test changes to common.rs.

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
Partial projection (alpha) model of src/bin/lindblad_alpha*: the same tools with a trailing alpha
(full 2^L basis; L = 6 takes seconds, reference_diag.py at L = 8 needs 4.8 GB and 11-17 min):
```bash
python3 tools/check_symmetries.py 6 0.2 0.001 1.0 0.4     # residuals of T, R, S, Θ and block sizes
python3 tools/check_symmetries.py sizes 10 full            # block sizes only (no matrices)
python3 tools/reference_diag.py 6 0.2 0.001 1.0 ref6a.json 0.4
python3 tools/compare_diag.py ref6a.json run_dir           # rows must start with q,gp,gm,omega,alpha
python3 tools/reference_evolution.py 6 10 0.001 0.2 0.001 1.0 ref6a.npy 0.4
python3 tools/compare_evolution.py ref6a.npy occupation_time_alpha.csv 0.2 0.001 1.0 0.4
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
- Diagonalization cost model (2026-10-09). dgeev with vectors on the 112-core node: 1.1e-10 · n³ s
  (from the 2.0 h above, Σ n³ = 6.56e13); eigenvalues only costs 0.6 of that (measured locally,
  n ≈ 2.2k: 1.1e-10 against 1.8e-10 n³ on 4 laptop threads). Peak RSS measured locally:
  staggered L=12 (n = 4736, OEE on) 479 MB = 19 n² bytes after the ~45 MB baseline, so use 16-20 n².
  Estimates per parameter point with OEE: staggered L=14 ≈ 3.2 h, 11-15 GB (same block sizes and
  mode count as the measured run); alpha L=10 (8 blocks of ~26k, 209.7k modes) ≈ 4.4 h eig + 2.4-4.3 h
  OEE, 12-16 GB. Not feasible: staggered L=16 (n = 154961: 385-480 GB, 4.7 days per block, ILP64),
  alpha L=12 (n = 353384: 2 TB).
- lindblad_evol with the (+,+) blocks (2026-10-09, i7-12650H, gp=0.2 gm=0.001 Ω=1; logs and the
  scripts count_nnz.py / predict.py in target/claude_tmp/ramtime/):
  nnz per block (Q=0): L=10 18522, L=12 194986, L=14 2935080, L=16 24792228 (319 per row);
  ||A-μ||_1 = 22.2 / 28.4 / 35.5 / 43.9 for L = 10..16 (≈ 2ΩL + a γ+ term growing like L²).
  Peak RSS (in build_csr, where `cols` and the CSR coexist; independent of T, dt, threads and of the
  number of parameter points): L=12 13 MiB, L=14 108 MiB, L=16 0.95 GiB ≈ 37 B·nnz_block + 310 B·D_sector.
  Extrapolated with the exact sector pattern count × measured block/sector ratio (0.14-0.18):
  L=18 ≈ 10 GiB (8.6-11.3), L=20 ≈ 80 GiB (70-98). L=18 was NOT run here (too close to 15 GiB).
  Time per matrix product (one block): L=14 0.68 ms, L=16 6.7 ms at 4 threads; no gain beyond
  4 threads (memory bandwidth), 1 thread is 2.6× slower. Products per step ≈ 7-10 for dt ≤ 0.01,
  the total for fixed T is nearly flat for dt ≥ 0.1. T=15, dt=0.01, both sectors: L=14 20 s, L=16 3.7 min.

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

- **This workstation has 15.7 GB (WSL2) and is shared with the user's own runs.** It crashed three
  times on 2026-10-09: two L=8 full-basis Python references at once (4.8 GB each; the OOM killer
  took a 10 GB lindblad_evol run), and twice during `cargo build --release` of the MKL binary
  (cause not confirmed). Rules: one heavy job at a time, few threads, ask before anything above
  ~1 GB; the user runs the release builds; expensive outputs go to target/claude_tmp/ (the session
  scratchpad in /tmp is wiped on restart); `target/claude_tmp/run_guarded.sh MIN_KB cmd...` stops a
  job when MemAvailable falls below MIN_KB. Run test binaries in a scratch folder, never in rust/
  (they overwrite the *.csv results there).

## Open items

- Per-block CLI option (`--sector Q --block στ`) + merge script, to run the 8 blocks of one
  parameter point on separate nodes (L=14 ~20-30 min wall; L=16 Néel blocks ~10-15 h each, ~100 GB).
- ILP64 switch for L=16.
- Optional phase fixing of eigenvectors (e.g. o_k real positive) for reproducible raw c_k, o_k.
- OEE speed (memory-bandwidth bound with 112 concurrent SVDs; try fewer rayon threads).
- Alpha model: L=10 diagonalization on the cluster; memory of the L=12 evolution; full-precision
  decay files; <nn>(t) at L=8 is not checked against a reference.