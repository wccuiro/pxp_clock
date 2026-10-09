# Partial projection (alpha) model: evaluation, measurements and state of the work

Written 2026-10-09. Everything here was measured in that session unless marked otherwise.
Short rules that must be followed are in `CLAUDE.md`; this file keeps the numbers behind them.

## 1. Question and answer

Can `lindblad_alpha` / `lindblad_alpha_evol` use the same reductions as `lindblad` / `lindblad_evol`?
**Yes, unchanged.** H keeps the strict blockade and the jumps are
A^±_j = P^α_{j-1} σ^±_j P^α_{j+1}, P^α = [(1+α)|0><0| + (1-α)|1><1|] / (1+|α|): real, translation and
reflection invariant, and C A C = -A. So translation (Q sectors), R, S and Hermiticity all hold.
The only differences are the full 2^L basis and a real weight on the jump tables.

Numerical check (`tools/check_symmetries.py`, L = 4 and 6, α = -0.5, 0, 0.4, 1, and the constrained
model): residuals of R, S, Hermiticity = 0 to 2e-16, translation 2.2e-16; the Néel state has weight
0.5 in (+,+) of each sector and 0 elsewhere; the trace is only in (+,+) of Q = 0.

## 2. Block sizes (from `check_symmetries.py sizes`, verified against Rust where run)

Full 2^L basis, (+,+)/(+,-)/(-,+)/(-,-):

| L | Q | sector | (+,+) | (+,-) | (-,+) | (-,-) |
|---|---|---|---|---|---|---|
| 4 | 0 | 70 | 34 | 21 | 6 | 9 |
| 4 | 2 | 66 | 24 | 21 | 12 | 9 |
| 6 | 0 | 700 | 237 | 193 | 125 | 145 |
| 6 | 3 | 676 | 193 | 193 | 145 | 145 |
| 8 | 0 | 8230 | 2299 | 2136 | 1851 | 1944 |
| 8 | 4 | 8226 | 2169 | 2136 | 1977 | 1944 |
| 10 | 0 | 104968 | 27190 | 26574 | 25398 | 25806 |
| 10 | 5 | 104760 | 26574 | 26574 | 25806 | 25806 |
| 12 | 0 | 1398500 | 353384 | 350986 | 346216 | 347914 |
| 12 | 6 | 1398476 | 351324 | 350986 | 348252 | 347914 |
| 14 | 0 | 19175140 | 4808707 | 4799343 | 4780035 | 4787055 |
| 14 | 7 | 19172796 | 4799343 | 4799343 | 4787055 | 4787055 |

Constrained basis (for comparison): L=6 Q=0 30/17/4/9, Q=3 21/21/6/6; L=8 Q=0 109/78/41/57,
Q=4 94/88/54/48; L=10 Q=0 478/403/300/348, Q=5 428/428/324/324.

## 3. The old alpha binaries (git history before the rewrite) against the references, L = 6

Run from a throwaway build (they do not compile in the repo: `sprs` missing in Cargo.toml and no MKL
link line). 24-point grid: 30 min 35 s (diagonalization), 6 min 48 s (evolution).

- eigenvalues: ≤ 1.3e-13. |c|, |o| of non-degenerate modes: ≤ 1.5e-9. Group sums of c conj(o): ≤ 2e-9.
- steady state (α < 1): n, nn ≤ 6e-12.
- **OEE: correct in Q = 0, wrong by ~0.8 in Q = π** (real-space expansion with +k on both sides and
  no Q; in Q = 0 the bug cancels in ρ†ρ).
- `decay_alpha.csv`: no alpha column; the last column pair was Tr(n l_k)/z_k (left eigenvector),
  not w_k, so <n>(t) could not be rebuilt from the file.
- evolution (RK4, dt = 0.001): n, nn agree to 1e-9 only (largest in the first steps; probably the
  1e-10 thresholds, not verified), F to 1.5e-11.
- left/right eigenvectors paired by sorting (worked at these points, fragile).
- α = 1: one steady-state row per zero mode with nonzero trace (19 per point).

## 4. The new binaries against the references

Same code as `lindblad` / `lindblad_evol` on the full basis (`Chain::new(l, false)`,
`Model::new(&ch, Some(α))`). Tools: `tools/reference_diag.py`, `reference_evolution.py`,
`compare_diag.py`, `compare_evolution.py`, all with a trailing alpha.

L = 4 (gp 0.13, gm 0.07, Ω 1.3, α 0.3): eigenvalues 1.6e-14, |c|, |o|, c conj(o), w ≤ 5e-11,
OEE 5e-7 (print precision), steady 3e-15; evolution n, nn, F ≤ 1.4e-14.

L = 6, Ω = 1, points (gp, gm, α) = (0.2, 0.001, 0.4), (0.001, 0.2, 0), (0.001, 0.001, 0.8), (0.2, 0.2, 1):
- eigenvalues ≤ 2.1e-13; |c|, |o|, c conj(o), w ≤ 7.8e-11; OEE ≤ 5e-7 in both sectors;
  steady n, nn ≤ 2.9e-12, spectrum ≤ 4.4e-12.
- evolution (T = 10, dt = 0.001): n, nn, F ≤ 1.9e-13 (max |Tr ρ - 1| ≤ 3.2e-13).
- default grid (24 points): diagonalization with OEE 7.2 s / 13 MB, evolution 5.6 s / 6 MB.

Constrained regression after the change to `common.rs` (rebuilt `lindblad`, `lindblad_evol`,
L = 8, gp 0.2, gm 0.001, Ω 1): eigenvalues 5.9e-14, |c| 6.3e-11, |o| 5.8e-11, c conj(o) 2.5e-11,
w 6.6e-11, steady n, nn 5.6e-17; evolution T = 25, dt = 0.01: n 2.1e-14, nn 9.9e-15, F 3.3e-15.

### L = 8 (gp 0.2, gm 0.001, Ω 1, α 0.4)

Python reference (`reference_diag.py 8 0.2 0.001 1.0 ref.json 0.4`): 17 min 05 s, peak 4.84 GB
(Q = 0: 366 s, Q = 4: 658 s); steady <n> = 0.9950468443, <nn> = 0.9901018028.
Rust: `lindblad_alpha` 61 s / 159 MB with 8 threads (150 s / 137 MB with 4 threads at low priority),
steady <n> = 0.9950468442971849, <nn> = 0.9901018027650763; `lindblad_alpha_evol` T = 10, dt = 0.01: 4.7 s / 12 MB.

First comparison (prototype = same code, done before the reference file was lost in a crash, with
the earlier version of `compare_diag.py`):
- eigenvalues 6.8e-13 (Q=0), 2.7e-13 (Q=4); |o| 6.4e-11; steady n, nn 8.3e-15, spectrum 2.3e-14;
- 3040 (Q=0) / 3082 (Q=4) modes in degenerate groups (|Δλ| < 1e-8);
- |c| 9.1e-8, c conj(o) 1.3e-8, w 2.7e-8, OEE 1.0e-5: above the old absolute tolerances. Cause:

| gap to nearest eigenvalue (Q=0) | modes | max d\|c\| | max \|c\| | max dOEE |
|---|---|---|---|---|
| < 1e-8 | 3040 | (group, not compared) | 0.53 | |
| 1e-8 .. 1e-6 | 1674 | 1.3e-8 | 0.29 | 7.5e-6 |
| 1e-6 .. 1e-4 | 3040 | 1.6e-8 | 7.8 | 6.4e-7 |
| 1e-4 .. 1e-2 | 293 | 5.1e-8 | 3.4e2 | 5.0e-7 |
| > 1e-2 | 183 | 4.9e-11 | 1.7 | 4.9e-7 |

  (Q=4: gap 1e-8..1e-6: d|c| 9.1e-8, dOEE 1.0e-5; above 1e-6: d|c| ≤ 2.1e-10.)
  Gap histogram Q=0 (log10 bins of width 2 from -16): 54, 328, 782, 1876, 1674, 3040, 293, 181.
  The c conj(o) and w differences came from one conjugate pair with |c| ≈ 340, |w| ≈ 72,
  |c conj(o)| = 2.3e-4: relative differences 3.7e-10 (w); the c conj(o) one is the 10-decimal print
  of o times |c| (340 × 5e-11).
- Hence the rule now in `compare_diag.py`: |c|, c conj(o), w differences relative to max(1, size);
  per-mode differences of |c|, |o|, OEE multiplied by min(1, gap / 1e-4).
  The constrained comparisons are unchanged by it (all gaps > 1e-4 there) and it still flags the old
  code's w and OEE.

Final comparison (final binaries, final `compare_diag.py`, reference recomputed: 10 min 52 s alone
on the machine, peak 4.84 GB; the comparison itself 6 s, 1.7 GB): **all ok**

| quantity | Q = 0 | Q = 4 | tolerance |
|---|---|---|---|
| eigenvalues | 4.4e-13 | 3.2e-13 | 1e-9 |
| \|c\| (relative, gap-weighted) | 3.8e-10 | 5.9e-11 | 1e-8 |
| \|o\| (gap-weighted) | 5.7e-11 | 6.4e-11 | 1e-8 |
| c conj(o) (group sums, relative) | 6.2e-11 | 2.4e-11 | 1e-8 |
| w (group sums, relative) | 4.1e-10 | 1.2e-17 | 1e-8 |
| OEE (gap-weighted) | 5.0e-7 | 5.0e-7 | 5e-6 |

steady n, nn 4.9e-15, steady spectrum 3.8e-15 (occupations 2.8e-7, print precision).
Modes: 3040 / 3082 in degenerate groups, 4714 / 4666 more with a gap below 1e-4 (so only ~480
modes per sector are compared mode by mode at full weight; the rest through the gap weight and the
group sums).

Evolution at L = 8 (`lindblad_alpha_evol 8 10 0.01 0.2 0.001 1.0 0.4`):
- vs the eigen-expansion of the Python reference (`target/claude_tmp/evo_vs_modes.py`, full
  precision, independent of the Rust code): max |d<n>| = 9.0e-13, max |dF| = 7.6e-15;
- vs the eigen-expansion of the Rust `decay` file: max |d<n>| = 3.3e-8, max |dF| = 6.0e-9
  (limit of the 10 decimals with |c| up to 340).
<nn>(t) is not checked at L = 8 (the reference file has no nn weights), and there is no
`reference_evolution.py` run at L = 8: the dense expm of a complex 8230 matrix needs roughly 10 GB.

## 5. Reach

Evolution, `lindblad_alpha_evol` prototype, gp 0.2, gm 0.001, Ω 1, α 0.4, T = 25, dt = 0.01, 6 threads:

| L | (+,+) blocks | nnz (Q=0 / Q=L/2) | nnz per column | time | peak RAM |
|---|---|---|---|---|---|
| 6 | 237 / 193 | 5065 / 3601 | 21 | 0.5 s (T=10, dt=0.001) | few MB |
| 8 | 2299 / 2169 | 156889 / 143201 | 68 | 12.6 s | 12 MB |
| 10 | 27190 / 26574 | 7791570 / 7530778 | 287 | 199 s | 278 MB |
| 12 | 353384 / 351324 | not measured | | stopped at 5.7 GB while building | |

At α = 1 the L=6 blocks have nnz 1008 / 767 (the frozen configurations decouple).
Diagonalization: L = 10 has 8 blocks of ~26-27k (5.9 GB per dense matrix, about twice the L = 14
constrained blocks, so roughly 8× their eigensolve time each): a cluster job. L = 12 (353k) is out
of reach for dense diagonalization. OEE at L = 10: Schmidt matrices 1024 × 1024, 210k modes.

## 6. Degeneracies and extra symmetry

- Generic parameters (gp 0.13, gm 0.31, Ω 0.7), L = 6, α = 0.37 and α = 0: no eigenvalue degenerate
  inside a block and none shared between blocks → no sign of a further symmetry for 0 ≤ α < 1.
  Same for the constrained model at L = 6 and 8.
- gp 0.2, gm 0.001: many eigenvalues coincide to better than 1e-8 between (+,+)~(+,-) and
  (-,+)~(-,-) (L=6, α=0.4: 53 and 26 in Q=0). These are near-degeneracies from the small γ- (they
  vanish at generic parameters), not a symmetry.
- α = -0.6 (generic gp, gm, Ω): exact shared eigenvalues (+,+)~(+,-) 17, (-,+)~(-,-) 7 in Q=0
  (10 and 9 in Q=3). Not investigated; α < 0 is outside the grid in use.
- L = 4 has a few accidental coincidences.

## 7. α = 1 in the full basis

With strict projectors a site flips only if both neighbours are empty, so configurations with
adjacent excitations are frozen. L = 6: the 64 configurations split into 29 disconnected sets,
7 up to translation and reflection (sizes 18 [constrained, contains Néel], 3, 2, 1, 1, 1, 1).
Zero eigenvalues: 49 in Q = 0, 42 in Q = 3 (Python reference); the (+,+) block of Q = 0 has 24.
`lindblad_alpha` writes one steady state per point, ρ∞ = Σ_{λ_k=0} c_k r_k. Check at L = 6, Ω = 1:

| gp, gm | <n> from ρ∞ at α = 1 | constrained steady <n> | difference n / nn |
|---|---|---|---|
| 0.2, 0.2 | 0.2777777777 | 0.2777777778 | 1.7e-16 / 1.9e-16 |
| 0.2, 0.001 | 0.3096488691 | 0.3096488690 | 2.2e-16 / 3.3e-16 |
| 0.001, 0.2 | 0.2475797898 | 0.2475797898 | 2.8e-16 / 2.4e-16 |
| 0.001, 0.001 | | | 8.0e-15 / 7.8e-15 |

Spectrum of ρ∞: 18 weights equal to the constrained ones (≤ 2.1e-14), the other 46 are 0 (≤ 8e-15).

## 8. Other observations

- `decay*.csv` is printed with 10 decimals: c conj(o) is only good to |c| × 5e-11, and |c| reached
  340 at L = 8 (it grows with the non-normality). Printing these files with full precision would
  remove that limit (not changed: output conventions).
- `rust/config.toml` (`-C target-cpu=native`) is not inside `.cargo/`, so cargo does not read it.
- `cargo build` without `--bin` fails: `lindblad_asymmetric*` and `trajectories` are old code;
  the asymmetric ones `use sprs`, which is not in Cargo.toml.
- `plot_scripts/plot_decay_alpha.py` expects 7 values per mode; the file has 8 (as before).
- `plot_scripts/obs_fid_alpha.py` reads `occupation_time_alpha.csv` with dt = 1e-3 hard-coded; the
  default grid of `lindblad_alpha_evol` keeps T = 10, dt = 0.001 for that reason.

## 9. Machine

WSL2, 15.7 GB RAM, 16 cores. The session crashed three times on 2026-10-09: once with two L = 8
Python references (4.8 GB each) running at once (the OOM killer killed a 10 GB `lindblad_evol`
that was not started by the assistant), twice during `cargo build --release` of the MKL binary
started by the assistant (cause not confirmed). Builds are now run by the user.
`target/claude_tmp/run_guarded.sh MIN_KB cmd...` runs a job at low priority and stops it when
MemAvailable falls below MIN_KB.

## 10. State of the repository (2026-10-09, nothing committed)

Changed: `src/common.rs`, `src/bin/lindblad/main.rs`, `src/bin/lindblad_evol/main.rs` (call sites
only; `COMPUTE_OEE = false` in lindblad is the user's own edit), `src/bin/lindblad_alpha/main.rs`,
`src/bin/lindblad_alpha_evol/main.rs` (rewritten), `tools/*.py` (alpha argument, tolerances),
`tools/check_symmetries.py` (new), `CLAUDE.md`, this file.
All four binaries were rebuilt by the user after the last source change.

Scratch (git-ignored, can be deleted): `target/claude_tmp/`
- `alpha6/` L = 6 references (`ref_<gp>_<gm>_<alpha>.json`, `evo_*.npy`), one-off scripts
- `alpha8/` L = 8 reference (`ref_0.2_0.001_0.4.json`), `alpha_run/` output of the old alpha code (L = 6 grid)
- `final/` runs of the final binaries (`grid2/` = default grid, `alpha8/` = L = 8, `con8/`)
- `proto/`, `proto_*.diff`, `patch_proto.py` the prototype the rewrite started from
