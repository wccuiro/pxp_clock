//! Exact diagonalization of the PXP Lindbladian with PXP-dressed local σ± jumps.
//!
//!   L[ρ] = -iΩ [H, ρ] + Σ_{γ∈{+,-}} γ Σ_j ( A^γ_j ρ A^γ_j† - ½ {A^γ_j† A^γ_j, ρ} )
//!   H = Σ_j P_{j-1} X_j P_{j+1},   A^+_j = P_{j-1} σ^+_j P_{j+1},   A^-_j = P_{j-1} σ^-_j P_{j+1}
//!
//! Symmetries used (all exact, implemented in common.rs):
//!  1. Translations:  sectors ρ_Q = Σ_k P_k ρ P_{k-Q}, basis |n,k><m,k-Q|.
//!     Only Q = 0 and Q = L/2 are computed (as before).
//!  2. Reflection R: j -> -j mod L.
//!  3. S: ρ -> C ρᵀ C with C = Π_j Z_j.
//!     Each Q sector splits into the four blocks (σ, τ) = characters of {1, R, S, RS}.
//!  4. Hermiticity ρ -> ρ†: each block is a REAL matrix in a Hermitian basis
//!     (diagonalized with dgeev instead of zgeev).
//! The Néel state, the trace and the observables n, nn live only in (σ, τ) = (+, +).
//!
//! Outputs (same files and column layouts as before, rows ordered by sector then parameters):
//!   eigenvalues.csv     q,gp,gm,omega, (re,im)*
//!   decay.csv           q,gp,gm,omega, (re,im, c.re,c.im, o.re,o.im, w.re,w.im)*
//!                         c_k: Néel expansion coefficient on the unit-norm right eigenvector r_k
//!                         o_k: r_k† ρ0
//!                         w_k: c_k Tr(n r_k)  ->  <n>(t) = Σ_k w_k e^{λ_k t}   (phase-invariant)
//!   oee.csv             q,gp,gm,omega, (re,im,entropy)*            [if COMPUTE_OEE]
//!   std_eigenvalues.csv q,gp,gm,omega, (p, n)*                     [if COMPUTE_STEADY_STATE]
//!   occupation.csv      q,gp,gm,omega,n,nn                         [if COMPUTE_STEADY_STATE]
//!   cond.csv            q,gp,gm,omega, (re,im,κ)*                  [if COMPUTE_CONDITION_NUMBERS]
//!
//! Usage:  lindblad_solver [L] [gp] [gm] [omega]
//!   with no arguments the parameter grid defined in `Config::default_grid` is used.
//!
//! Threads: set MKL_NUM_THREADS (eigensolver) and RAYON_NUM_THREADS (matrix build, OEE)
//! to the number of physical cores.

// Links the threaded MKL (see Cargo.toml); nothing from the crate is used directly.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src as _;

use ndarray::Array2;
use faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par};
use ndarray_linalg::{Eigh, UPLO};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

include!("../../common.rs");

// ============================================================================
// Run configuration
// ============================================================================

const COMPUTE_OEE: bool = true;
const COMPUTE_STEADY_STATE: bool = true;
const COMPUTE_CONDITION_NUMBERS: bool = false;

/// Eigenvalues with |Re|,|Im| below this are treated as steady states.
const TOL_ZERO_EIG: f64 = 1e-8;

struct Config {
    l: usize,
    params: Vec<Params>,
}

impl Config {
    fn default_grid() -> Self {
        let l = 12;
        let gp_values = linspace(0.2, 0.2, 1);
        let gm_values = linspace(0.001, 0.2, 1);
        let omega_values = linspace(1.0, 2.0, 1);
        let mut params = Vec::new();
        for &gp in &gp_values {
            for &gm in &gm_values {
                for &omega in &omega_values {
                    params.push(Params { gp, gm, omega });
                }
            }
        }
        Config { l, params }
    }

    fn from_args() -> Result<Self, Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut cfg = Config::default_grid();
        if !args.is_empty() {
            cfg.l = args[0].parse()?;
        }
        if args.len() >= 4 {
            cfg.params = vec![Params {
                gp: args[1].parse()?,
                gm: args[2].parse()?,
                omega: args[3].parse()?,
            }];
        }
        Ok(cfg)
    }
}

// ============================================================================
// LAPACK / BLAS (Fortran interface, provided by MKL or the system LAPACK)
// ============================================================================

/// LAPACK integer. LP64 is fine up to L = 14 (largest block n^2 < 2^31).
/// For L >= 16 switch to an ILP64 MKL (mkl-static-ilp64-iomp) and set this to i64.
type Int = i32;

extern "C" {
    fn dgeev_(
        jobvl: *const u8, jobvr: *const u8, n: *const Int, a: *mut f64, lda: *const Int,
        wr: *mut f64, wi: *mut f64, vl: *mut f64, ldvl: *const Int, vr: *mut f64,
        ldvr: *const Int, work: *mut f64, lwork: *const Int, info: *mut Int,
    );
    fn dgetrf_(m: *const Int, n: *const Int, a: *mut f64, lda: *const Int, ipiv: *mut Int, info: *mut Int);
    fn dgetrs_(
        trans: *const u8, n: *const Int, nrhs: *const Int, a: *const f64, lda: *const Int,
        ipiv: *const Int, b: *mut f64, ldb: *const Int, info: *mut Int,
    );
    fn dgetri_(
        n: *const Int, a: *mut f64, lda: *const Int, ipiv: *const Int, work: *mut f64,
        lwork: *const Int, info: *mut Int,
    );
}

// MKL is only ever called from the main thread. Concurrent MKL calls from rayon
// workers crash with the statically linked Intel OpenMP runtime, so everything that
// runs in parallel (matrix build, OEE) is pure Rust (faer for the OEE linear algebra).

/// Real nonsymmetric eigensolver. `a` (column-major n×n) is destroyed.
/// Returns (wr, wi, vr); vr is empty when `vectors == false`.
fn dgeev(n: usize, a: &mut [f64], vectors: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let ni = n as Int;
    let jobvl = b'N';
    let jobvr = if vectors { b'V' } else { b'N' };
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let mut vr = if vectors { vec![0.0; n * n] } else { vec![0.0; 1] };
    let ldvr: Int = if vectors { ni } else { 1 };
    let mut vl = [0.0f64; 1];
    let one: Int = 1;
    let mut info: Int = 0;
    let mut wq = [0.0f64; 1];
    let lw_query: Int = -1;
    unsafe {
        dgeev_(&jobvl, &jobvr, &ni, a.as_mut_ptr(), &ni, wr.as_mut_ptr(), wi.as_mut_ptr(),
               vl.as_mut_ptr(), &one, vr.as_mut_ptr(), &ldvr, wq.as_mut_ptr(), &lw_query, &mut info);
    }
    if info != 0 {
        return Err(format!("dgeev workspace query failed, info = {info}"));
    }
    let lwork = wq[0] as Int;
    let mut work = vec![0.0; lwork.max(1) as usize];
    unsafe {
        dgeev_(&jobvl, &jobvr, &ni, a.as_mut_ptr(), &ni, wr.as_mut_ptr(), wi.as_mut_ptr(),
               vl.as_mut_ptr(), &one, vr.as_mut_ptr(), &ldvr, work.as_mut_ptr(), &lwork, &mut info);
    }
    if info != 0 {
        return Err(format!("dgeev failed, info = {info}"));
    }
    if !vectors {
        vr.clear();
    }
    Ok((wr, wi, vr))
}

/// Solves VR·x = rhs by LU (never forms VR⁻¹ for the solution). `scratch` (n²) is overwritten.
/// If `row_norms` is requested, VR⁻¹ is formed afterwards (diagnostic only) and the
/// Euclidean norms of its rows are returned (used for eigenvalue condition numbers).
fn lu_solve(
    n: usize, vr: &[f64], scratch: &mut [f64], rhs: &[f64], row_norms: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>), String> {
    let ni = n as Int;
    scratch.copy_from_slice(vr);
    let mut ipiv = vec![0 as Int; n];
    let mut info: Int = 0;
    unsafe { dgetrf_(&ni, &ni, scratch.as_mut_ptr(), &ni, ipiv.as_mut_ptr(), &mut info) };
    if info != 0 {
        return Err(format!("dgetrf: eigenvector matrix singular/defective (info = {info})"));
    }
    let mut x = rhs.to_vec();
    let one: Int = 1;
    unsafe {
        dgetrs_(&b'N', &ni, &one, scratch.as_ptr(), &ni, ipiv.as_ptr(), x.as_mut_ptr(), &ni, &mut info)
    };
    if info != 0 {
        return Err(format!("dgetrs failed, info = {info}"));
    }
    let norms = if row_norms {
        let mut wq = [0.0f64; 1];
        let lw_query: Int = -1;
        unsafe { dgetri_(&ni, scratch.as_mut_ptr(), &ni, ipiv.as_ptr(), wq.as_mut_ptr(), &lw_query, &mut info) };
        let lwork = (wq[0] as Int).max(1);
        let mut work = vec![0.0; lwork as usize];
        unsafe { dgetri_(&ni, scratch.as_mut_ptr(), &ni, ipiv.as_ptr(), work.as_mut_ptr(), &lwork, &mut info) };
        if info != 0 {
            return Err(format!("dgetri failed, info = {info}"));
        }
        let mut sq = vec![0.0; n];
        for col in scratch.chunks(n) {
            for (s, &v) in sq.iter_mut().zip(col) {
                *s += v * v;
            }
        }
        Some(sq.into_iter().map(f64::sqrt).collect())
    } else {
        None
    };
    Ok((x, norms))
}

// ============================================================================
// Eigenmode bookkeeping (dgeev stores complex pairs as two real columns)
// ============================================================================

#[derive(Clone, Copy)]
enum Kind {
    Real,
    PairPlus,  // λ = wr + i wi (wi > 0), v = col_j + i col_{j+1}
    PairMinus, // λ* , v = col_{j-1} - i col_j
}

struct Mode {
    lambda: C64,
    col: usize, // first column of the (pair of) columns
    kind: Kind,
}

fn modes_from(wr: &[f64], wi: &[f64]) -> Vec<Mode> {
    let n = wr.len();
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    while j < n {
        if wi[j] == 0.0 {
            out.push(Mode { lambda: C64::new(wr[j], 0.0), col: j, kind: Kind::Real });
            j += 1;
        } else {
            out.push(Mode { lambda: C64::new(wr[j], wi[j]), col: j, kind: Kind::PairPlus });
            out.push(Mode { lambda: C64::new(wr[j + 1], wi[j + 1]), col: j, kind: Kind::PairMinus });
            j += 2;
        }
    }
    out
}

/// Complex eigenvector of a mode in real-basis coordinates.
fn mode_vector(vr: &[f64], n: usize, md: &Mode) -> Vec<C64> {
    let u = &vr[md.col * n..(md.col + 1) * n];
    match md.kind {
        Kind::Real => u.iter().map(|&x| C64::new(x, 0.0)).collect(),
        Kind::PairPlus | Kind::PairMinus => {
            let w = &vr[(md.col + 1) * n..(md.col + 2) * n];
            let s = if matches!(md.kind, Kind::PairPlus) { 1.0 } else { -1.0 };
            u.iter().zip(w).map(|(&a, &b)| C64::new(a, s * b)).collect()
        }
    }
}

/// Expansion coefficient of the mode from the real solution x of VR·x = ρ0.
fn mode_coeff(x: &[f64], md: &Mode) -> C64 {
    match md.kind {
        Kind::Real => C64::new(x[md.col], 0.0),
        Kind::PairPlus => 0.5 * C64::new(x[md.col], -x[md.col + 1]),
        Kind::PairMinus => 0.5 * C64::new(x[md.col], x[md.col + 1]),
    }
}

struct ModeRecord {
    lambda: C64,
    c: C64,
    o: C64,
    w: C64,
    cond: f64,
    entropy: f64,
}

// ============================================================================
// Operator entanglement entropy of ρ_k† ρ_k  (dense, real space)
// ============================================================================

struct OeeContext {
    d: usize,
    d_a: usize,
    d_b: usize,
    low_a: Vec<u32>, // config index -> Fibonacci index of the lower L/2 bits
    up_b: Vec<u32>,  // config index -> Fibonacci index of the upper bits
}

impl OeeContext {
    fn new(ch: &Chain) -> Self {
        let l = ch.l;
        let l_a = l / 2;
        let l_b = l - l_a;
        let fib = |n: usize| -> Vec<u32> {
            let mut idx = vec![NONE; 1 << n];
            let mut c = 0u32;
            for s in 0..(1u64 << n) {
                if s & (s >> 1) == 0 {
                    idx[s as usize] = c;
                    c += 1;
                }
            }
            idx
        };
        let fa = fib(l_a);
        let fb = fib(l_b);
        let d_a = fa.iter().filter(|&&x| x != NONE).count();
        let d_b = fb.iter().filter(|&&x| x != NONE).count();
        let mask_a = (1u64 << l_a) - 1;
        let low_a = ch.configs.iter().map(|&c| fa[(c & mask_a) as usize]).collect();
        let up_b = ch.configs.iter().map(|&c| fb[(c >> l_a) as usize]).collect();
        OeeContext { d: ch.configs.len(), d_a, d_b, low_a, up_b }
    }

    /// v: eigenmatrix in the sector basis.
    fn entropy(&self, ch: &Chain, sec: &Sector, v: &[C64]) -> f64 {
        let d = self.d;
        let l = ch.l;
        // 1. real-space coefficient matrix C (column-major: C[a + b d] for |a><b|)
        let mut cm = vec![ZERO; d * d];
        for (t, &vt) in v.iter().enumerate() {
            if vt.norm() <= 1e-14 {
                continue;
            }
            let (k, n, m) = sec.elems[t];
            let (k, n, m) = (k as i64, n as usize, m as usize);
            let kb = (k + l as i64 - sec.q as i64) % l as i64;
            let on = &ch.orbit[n];
            let om = &ch.orbit[m];
            let norm = ((on.len() * om.len()) as f64).sqrt();
            for (dn, &a) in on.iter().enumerate() {
                let ca = vt * ch.ph(-k * dn as i64) / norm;
                for (dm, &b) in om.iter().enumerate() {
                    cm[a as usize + b as usize * d] += ca * ch.ph(kb * dm as i64);
                }
            }
        }
        // 2. S = ρ†ρ = C^H C   (faer, sequential inside this rayon task)
        let c = MatRef::from_column_major_slice(&cm, d, d);
        let mut s = Mat::<C64>::zeros(d, d);
        matmul(s.as_mut(), Accum::Replace, c.adjoint(), c, ONE, Par::Seq);
        drop(cm);
        // 3. reshape S[(a_ket,a_bra),(b_ket,b_bra)] and take singular values
        let mut mx = Mat::<C64>::zeros(self.d_a * self.d_a, self.d_b * self.d_b);
        for b2 in 0..d {
            for b1 in 0..d {
                let val = s[(b1, b2)];
                if val == ZERO {
                    continue;
                }
                let row = self.low_a[b1] as usize * self.d_a + self.low_a[b2] as usize;
                let col = self.up_b[b1] as usize * self.d_b + self.up_b[b2] as usize;
                mx[(row, col)] = val;
            }
        }
        drop(s);
        let sv = match mx.singular_values() {
            Ok(sv) => sv,
            Err(_) => return f64::NAN,
        };
        let total: f64 = sv.iter().map(|x| x * x).sum();
        if total < 1e-12 {
            return 0.0;
        }
        sv.iter()
            .map(|x| x * x / total)
            .filter(|&lam| lam > 1e-12)
            .map(|lam| -lam * lam.ln())
            .sum()
    }
}

// ============================================================================
// Steady state
// ============================================================================

/// Spectrum of the normalized steady state (per momentum block) with the occupation
/// of each eigenvector, sorted by descending weight (same as before).
fn steady_state_spectrum(ch: &Chain, sec: &Sector, v: &[C64]) -> Vec<(f64, f64)> {
    let l = ch.l;
    let mut res = Vec::new();
    for k in 0..l {
        let nk = ch.mom[k].len();
        if nk == 0 {
            continue;
        }
        let mut blk = Array2::<C64>::zeros((nk, nk));
        for i in 0..nk {
            for j in 0..nk {
                blk[[i, j]] = v[sec.off[k] + i * nk + j];
            }
        }
        let herm = (&blk + &blk.t().mapv(|c| c.conj())) * 0.5;
        if let Ok((ev, evec)) = herm.eigh(UPLO::Upper) {
            let occ: Vec<f64> =
                ch.mom[k].iter().map(|&r| ch.reps[r as usize].count_ones() as f64 / l as f64).collect();
            for (idx, &lam) in ev.iter().enumerate() {
                let col = evec.column(idx);
                let (mut num, mut den) = (0.0, 0.0);
                for (c, o) in col.iter().zip(&occ) {
                    num += c.norm_sqr() * o;
                    den += c.norm_sqr();
                }
                res.push((lam, if den > 1e-16 { num / den } else { 0.0 }));
            }
        }
    }
    let tr: f64 = res.iter().map(|x| x.0).sum();
    if tr.abs() > 1e-15 {
        for x in res.iter_mut() {
            x.0 /= tr;
        }
    }
    res.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    res
}

// ============================================================================
// Driver
// ============================================================================

struct Outputs {
    eigenvalues: BufWriter<File>,
    decay: BufWriter<File>,
    oee: Option<BufWriter<File>>,
    std_eigenvalues: Option<BufWriter<File>>,
    occupation: Option<BufWriter<File>>,
    cond: Option<BufWriter<File>>,
}

impl Outputs {
    fn create(l: usize, p: &Params) -> std::io::Result<Self> {
        let suffix = format!("_L{}_gp{}_gm{}_omega{}", l, p.gp, p.gm, p.omega);
        let f = |name: &str| File::create(format!("{name}{suffix}.csv")).map(BufWriter::new);
        
        let mut occupation = if COMPUTE_STEADY_STATE { Some(f("occupation")?) } else { None };
        if let Some(o) = occupation.as_mut() {
            writeln!(o, "q_sector,gp,gm,omega,n,nn")?;
        }
        Ok(Outputs {
            eigenvalues: f("eigenvalues")?,
            decay: f("decay")?,
            oee: if COMPUTE_OEE { Some(f("oee")?) } else { None },
            std_eigenvalues: if COMPUTE_STEADY_STATE { Some(f("std_eigenvalues")?) } else { None },
            occupation,
            cond: if COMPUTE_CONDITION_NUMBERS { Some(f("cond")?) } else { None },
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_block(
    ch: &Chain, model: &Model, sec: &Sector, blk: &Block, p: &Params, oee: &OeeContext,
    records: &mut Vec<ModeRecord>, steady: &mut Vec<(Vec<(f64, f64)>, f64, f64)>,
) -> Result<(), Box<dyn Error>> {
    let n = blk.dim();
    if n == 0 {
        return Ok(());
    }
    let has_rho0 = blk.rho0.iter().any(|&x| x != 0.0);
    let has_trace = blk.tr.iter().any(|&x| x.abs() > 1e-14);
    let want_vectors =
        has_rho0 || COMPUTE_OEE || COMPUTE_CONDITION_NUMBERS || (COMPUTE_STEADY_STATE && has_trace);

    let stage = |what: &str| eprintln!("    Q={} (σ,τ)=({:+},{:+}) dim={}: {what}", sec.q, blk.sigma, blk.tau, n);
    // 1. build the real block
    stage("building matrix");
    let t0 = Instant::now();
    let mut a = vec![0.0f64; n * n];
    let max_im = blk.build_matrix(ch, model, sec, p, &mut a);
    let t_build = t0.elapsed().as_secs_f64();

    // 2. diagonalize (A is overwritten by the Schur form)
    stage(if want_vectors { "dgeev (eigenvalues + right eigenvectors)" } else { "dgeev (eigenvalues only)" });
    let t0 = Instant::now();
    let (wr, wi, vr) = dgeev(n, &mut a, want_vectors)?;
    let t_eig = t0.elapsed().as_secs_f64();
    let modes = modes_from(&wr, &wi);

    // 3. Néel coefficients: solve VR x = ρ0 by LU (A's memory reused as scratch)
    let mut x = vec![0.0; n];
    let mut row_norms = None;
    if want_vectors && (has_rho0 || COMPUTE_CONDITION_NUMBERS) {
        stage("LU solve for Néel coefficients");
        match lu_solve(n, &vr, &mut a, &blk.rho0, COMPUTE_CONDITION_NUMBERS) {
            Ok((xs, rn)) => {
                x = xs;
                row_norms = rn;
            }
            Err(e) => {
                // Exactly singular eigenvector matrix (exceptional point): no eigen-expansion exists.
                eprintln!("    WARNING Q={} (σ,τ)=({:+},{:+}): {e}; coefficients set to NaN", sec.q, blk.sigma, blk.tau);
                x = vec![f64::NAN; n];
            }
        }
    }
    drop(a);

    // 4. per-mode quantities
    let col = |j: usize| &vr[j * n..(j + 1) * n];
    let dot = |u: &[f64], w: &[f64]| -> f64 { u.iter().zip(w).map(|(a, b)| a * b).sum() };
    let mut entropies = vec![0.0; modes.len()];
    if want_vectors && COMPUTE_OEE {
        let one = |md: &Mode| {
            let y = mode_vector(&vr, n, md);
            let v = blk.to_sector(sec.dim, &y);
            oee.entropy(ch, sec, &v)
        };
        stage("operator entanglement entropies (parallel over modes)");
        entropies = modes.par_iter().map(one).collect();
    }
    let mut sum_ctr = ZERO;
    let mut sum_w = ZERO;
    for (idx, md) in modes.iter().enumerate() {
        let mut rec = ModeRecord { lambda: md.lambda, c: ZERO, o: ZERO, w: ZERO, cond: f64::NAN, entropy: entropies[idx] };
        if want_vectors {
            // y = u + i s w  (s = ±1 for pairs, w = 0 for real modes)
            let u = col(md.col);
            let (w, s) = match md.kind {
                Kind::Real => (None, 0.0),
                Kind::PairPlus => (Some(col(md.col + 1)), 1.0),
                Kind::PairMinus => (Some(col(md.col + 1)), -1.0),
            };
            let lin = |f: &[f64]| -> C64 {
                C64::new(dot(u, f), w.map_or(0.0, |w| s * dot(w, f)))
            };
            let c = mode_coeff(&x, md);
            let tr_n = lin(&blk.nocc);
            let tr_r = lin(&blk.tr);
            rec.c = c;
            rec.o = lin(&blk.rho0).conj(); // r† ρ0 with ρ0 real
            rec.w = c * tr_n;
            sum_ctr += c * tr_r;
            sum_w += rec.w;
            if let Some(rn) = &row_norms {
                let vnorm2 = dot(u, u) + w.map_or(0.0, |w| dot(w, w));
                rec.cond = match md.kind {
                    Kind::Real => rn[md.col] * vnorm2.sqrt(),
                    _ => 0.5 * (rn[md.col].powi(2) + rn[md.col + 1].powi(2)).sqrt() * vnorm2.sqrt(),
                };
            }
            // steady state
            if COMPUTE_STEADY_STATE
                && matches!(md.kind, Kind::Real)
                && md.lambda.re.abs() < TOL_ZERO_EIG
                && md.lambda.im.abs() < TOL_ZERO_EIG
            {
                let tr = tr_r.re;
                if tr.abs() > 1e-10 {
                    let y: Vec<C64> = u.iter().map(|&a| C64::new(a / tr, 0.0)).collect();
                    let exp_n = dot(u, &blk.nocc) / tr;
                    let exp_nn = dot(u, &blk.nnn) / tr;
                    let v = blk.to_sector(sec.dim, &y);
                    steady.push((steady_state_spectrum(ch, sec, &v), exp_n, exp_nn));
                }
            }
        }
        records.push(rec);
    }
    eprintln!(
        "    Q={} (σ,τ)=({:+},{:+}) dim={:>6}  build {:7.2}s (max dropped Im {:.1e})  eig {:8.2}s  vectors={}{}",
        sec.q, blk.sigma, blk.tau, n, t_build, max_im, t_eig, want_vectors,
        if has_rho0 { format!("  Σc·Tr(r)={:.12}  Σw={:.12}", sum_ctr.re, sum_w.re) } else { String::new() }
    );
    Ok(())
}

/// Calls every LAPACK/BLAS routine used later on tiny inputs with known answers.
/// A crash here (rather than later) points to an ABI/linking problem, typically an
/// ILP64 MKL being linked while this program passes 32-bit integers (type Int).
fn lapack_self_test() -> Result<(), Box<dyn Error>> {
    eprint!("LAPACK self-test: dgeev");
    // column-major [[1,2,3],[0,4,5],[0,0,6]] -> eigenvalues 1,4,6
    let mut a = vec![1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0];
    let (wr, wi, vr) = dgeev(3, &mut a, true)?;
    let mut ev = wr.clone();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if (ev[0] - 1.0).abs() > 1e-12 || (ev[1] - 4.0).abs() > 1e-12 || (ev[2] - 6.0).abs() > 1e-12
        || wi.iter().any(|&x| x != 0.0)
    {
        return Err(format!("dgeev returned wrong eigenvalues {wr:?} {wi:?}").into());
    }
    eprint!(", dgetrf/dgetrs/dgetri");
    let mut scratch = vec![0.0; 9];
    let rhs = [1.0, 2.0, 3.0];
    let (x, rn) = lu_solve(3, &vr, &mut scratch, &rhs, true)?;
    for i in 0..3 {
        let r: f64 = (0..3).map(|j| vr[i + 3 * j] * x[j]).sum();
        if (r - rhs[i]).abs() > 1e-10 || rn.as_ref().map_or(true, |v| !v[i].is_finite()) {
            return Err("LU solve check failed".into());
        }
    }
    eprint!(", zheev");
    let (eh, _) = Array2::from_shape_vec((2, 2), vec![C64::new(2.0, 0.0), IMAG, -IMAG, C64::new(2.0, 0.0)])?
        .eigh(UPLO::Upper)?;
    if (eh[0] - 1.0).abs() > 1e-12 {
        return Err("zheev check failed".into());
    }
    eprint!(" | faer (parallel OEE path)");
    let ok = (0..4 * rayon::current_num_threads()).into_par_iter().all(|t| {
        let m = Mat::<C64>::from_fn(2, 2, |i, j| {
            if i != j { ZERO } else if i == 0 { C64::new(3.0 + t as f64, 0.0) } else { C64::new(0.0, 1.0) }
        });
        m.singular_values().map(|sv| (sv[0] - 3.0 - t as f64).abs() < 1e-12 && (sv[1] - 1.0).abs() < 1e-12).unwrap_or(false)
    });
    if !ok {
        return Err("faer singular value check failed".into());
    }
    eprintln!(" -> ok");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_args()?;
    let l = cfg.l;
    let t_all = Instant::now();
    // Larger stacks for rayon workers (they call LAPACK during the OEE stage).
    rayon::ThreadPoolBuilder::new().stack_size(256 << 20).build_global().ok();
    faer::set_global_parallelism(Par::Seq);
    lapack_self_test()?;
    let ch = Chain::new(l);
    let model = Model::new(&ch);
    let oee = OeeContext::new(&ch);
    eprintln!("L = {l}: {} constrained configurations, {} orbits", ch.configs.len(), ch.reps.len());

    let mut out = Outputs::create(l, &cfg.params[0])?;
    for q in [0, l / 2] {
        let sec = Sector::new(&ch, q);
        let blocks: Vec<Block> =
            [(1, 1), (1, -1), (-1, 1), (-1, -1)].iter().map(|&(sg, tg)| Block::new(&ch, &sec, sg, tg)).collect();
        eprintln!(
            "Sector Q={q}: dim {} -> blocks (+,+) {}, (+,-) {}, (-,+) {}, (-,-) {}",
            sec.dim, blocks[0].dim(), blocks[1].dim(), blocks[2].dim(), blocks[3].dim()
        );
        assert_eq!(blocks.iter().map(Block::dim).sum::<usize>(), sec.dim, "blocks do not cover the sector");
        for p in &cfg.params {
            eprintln!("  gp={} gm={} omega={}", p.gp, p.gm, p.omega);
            let mut records = Vec::with_capacity(sec.dim);
            let mut steady = Vec::new();
            for blk in &blocks {
                solve_block(&ch, &model, &sec, blk, p, &oee, &mut records, &mut steady)?;
            }
            records.sort_by(|a, b| {
                b.lambda.re.partial_cmp(&a.lambda.re).unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.lambda.im.partial_cmp(&a.lambda.im).unwrap_or(std::cmp::Ordering::Equal))
            });
            let head = format!("{},{},{},{}", q, p.gp, p.gm, p.omega);

            let w = &mut out.eigenvalues;
            write!(w, "{head}")?;
            for r in &records {
                write!(w, ",{},{}", r.lambda.re, r.lambda.im)?;
            }
            writeln!(w)?;

            let w = &mut out.decay;
            write!(w, "{head}")?;
            for r in &records {
                write!(w, ",{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",
                       r.lambda.re, r.lambda.im, r.c.re, r.c.im, r.o.re, r.o.im, r.w.re, r.w.im)?;
            }
            writeln!(w)?;

            if let Some(w) = out.oee.as_mut() {
                write!(w, "{head}")?;
                for r in &records {
                    write!(w, ",{:.10},{:.10},{:.6}", r.lambda.re, r.lambda.im, r.entropy)?;
                }
                writeln!(w)?;
            }
            if let Some(w) = out.cond.as_mut() {
                write!(w, "{head}")?;
                for r in &records {
                    write!(w, ",{:.10},{:.10},{:.6e}", r.lambda.re, r.lambda.im, r.cond)?;
                }
                writeln!(w)?;
            }
            for (spec, exp_n, exp_nn) in &steady {
                if let Some(w) = out.std_eigenvalues.as_mut() {
                    let body: Vec<String> = spec.iter().map(|(pp, nn)| format!("{:.16}, {:.6}", pp, nn)).collect();
                    writeln!(w, "{head},{}", body.join(", "))?;
                }
                if let Some(w) = out.occupation.as_mut() {
                    writeln!(w, "{head},{exp_n},{exp_nn}")?;
                }
            }
        }
    }
    for w in [Some(&mut out.eigenvalues), Some(&mut out.decay), out.oee.as_mut(),
              out.std_eigenvalues.as_mut(), out.occupation.as_mut(), out.cond.as_mut()].into_iter().flatten() {
        w.flush()?;
    }
    eprintln!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sizes of the blocks (+,+), (+,-), (-,+), (-,-) of the sector Q.
    fn block_sizes(l: usize, q: usize) -> Vec<usize> {
        let ch = Chain::new(l);
        let sec = Sector::new(&ch, q);
        let sizes: Vec<usize> =
            [(1, 1), (1, -1), (-1, 1), (-1, -1)].iter().map(|&(sg, tg)| Block::new(&ch, &sec, sg, tg).dim()).collect();
        assert_eq!(sizes.iter().sum::<usize>(), sec.dim);
        sizes
    }

    #[test]
    fn block_sizes_l14() {
        assert_eq!(block_sizes(14, 0), [13347, 12864, 12127, 12487]);
        assert_eq!(block_sizes(14, 7), [13033, 13033, 12319, 12319]);
    }

    #[test]
    fn block_sizes_l16() {
        assert_eq!(block_sizes(16, 0), [77812, 76566, 74618, 75579]);
        assert_eq!(block_sizes(16, 8), [77149, 77008, 75279, 75138]);
    }
}
