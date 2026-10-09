//! Exact time evolution of the Néel state under the PXP Lindbladian with partially projected
//! local σ± jumps.
//!
//!   L[ρ] = -iΩ [H, ρ] + Σ_{γ∈{+,-}} γ Σ_j ( A^γ_j ρ A^γ_j† - ½ {A^γ_j† A^γ_j, ρ} )
//!   H = Σ_j P_{j-1} X_j P_{j+1}                       (strict blockade, P = |0><0|)
//!   A^±_j = P^α_{j-1} σ^±_j P^α_{j+1},   P^α = [(1+α)|0><0| + (1-α)|1><1|] / (1+|α|)
//!
//! For α < 1 the jumps create adjacent excitations, so the full 2^L basis is used.
//!
//! Symmetries (implemented in common.rs, the same as in `lindblad_evol`): translations (sectors
//! Q = 0 and Q = L/2), reflection R, S (ρ -> C ρᵀ C) and Hermiticity. The Néel state lives only
//! in the real blocks (σ, τ) = (+, +) of the two sectors, so only these two sparse real matrices
//! are built.
//!
//! Method (exact to machine precision, no Euler or Runge-Kutta steps and no trace renormalization):
//!   v(t + Δt) = exp(Δt·L) v(t)
//! with the scaling and truncated Taylor algorithm for the action of the matrix exponential of
//! Al-Mohy & Higham, SIAM J. Sci. Comput. 33, 488 (2011): θ_m table and tolerance 2^-53 as in
//! scipy's expm_multiply, trace shift μ = Tr(L)/n. The trace is conserved; max |Tr ρ(t) - 1|
//! is printed as a check.
//!
//! Output: occupation_time_alpha.csv, one line per parameter point:
//!   alpha,gp,gm,omega, then n, nn, F at t = i·dt, i = 0..=round(T/dt)
//!   n = (1/L) Σ_j <n_j>,  nn = (1/L) Σ_j <n_{j-1} n_{j+1}>,  F = <Néel|ρ(t)|Néel>
//!
//! Usage:  lindblad_alpha_evol [L] [T] [dt] [gp gm omega alpha]
//!   missing arguments are taken from `Config::default_grid`.
//!
//! Threads: RAYON_NUM_THREADS (matrix build and sparse products). No LAPACK.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

include!("../../common.rs");

// ============================================================================
// Run configuration
// ============================================================================

struct Config {
    l: usize,
    t_final: f64,
    dt: f64,
    /// Parameter points with their alpha.
    points: Vec<(Params, f64)>,
}

impl Config {
    fn default_grid() -> Self {
        let l = 6;
        let t_final = 10.0;
        let dt = 1e-3;
        let gp_values = linspace(0.001, 0.2, 2);
        let gm_values = linspace(0.001, 0.2, 2);
        let omega_values = linspace(1.0, 2.0, 1);
        let alpha_values = linspace(0.0, 1.0, 6);
        let mut points = Vec::new();
        for &gp in &gp_values {
            for &gm in &gm_values {
                for &omega in &omega_values {
                    for &alpha in &alpha_values {
                        points.push((Params { gp, gm, omega }, alpha));
                    }
                }
            }
        }
        Config { l, t_final, dt, points }
    }

    fn from_args() -> Result<Self, Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut cfg = Config::default_grid();
        if !args.is_empty() {
            cfg.l = args[0].parse()?;
        }
        if args.len() >= 2 {
            cfg.t_final = args[1].parse()?;
        }
        if args.len() >= 3 {
            cfg.dt = args[2].parse()?;
        }
        if args.len() >= 7 {
            let p = Params {
                gp: args[3].parse()?,
                gm: args[4].parse()?,
                omega: args[5].parse()?,
            };
            cfg.points = vec![(p, args[6].parse()?)];
        }
        if !(cfg.dt > 0.0 && cfg.t_final >= 0.0) {
            return Err("T must be >= 0 and dt > 0".into());
        }
        Ok(cfg)
    }
}

// ============================================================================
// Action of the matrix exponential (Al-Mohy & Higham)
// ============================================================================

/// Backward error tolerance (unit roundoff of f64).
const TOL: f64 = 1.1102230246251565e-16; // 2^-53

/// (m, θ_m): the truncated Taylor series of degree m has backward error <= TOL if ||A||_1 <= θ_m.
const THETA: [(usize, f64); 35] = [
    (1, 2.29e-16), (2, 2.58e-8), (3, 1.39e-5), (4, 3.40e-4), (5, 2.40e-3),
    (6, 9.07e-3), (7, 2.38e-2), (8, 5.00e-2), (9, 8.96e-2), (10, 1.44e-1),
    (11, 2.14e-1), (12, 3.00e-1), (13, 4.00e-1), (14, 5.14e-1), (15, 6.41e-1),
    (16, 7.81e-1), (17, 9.31e-1), (18, 1.09), (19, 1.26), (20, 1.44),
    (21, 1.62), (22, 1.82), (23, 2.01), (24, 2.22), (25, 2.43),
    (26, 2.64), (27, 2.86), (28, 3.08), (29, 3.31), (30, 3.54),
    (35, 4.7), (40, 6.0), (45, 7.2), (50, 8.5), (55, 9.9),
];

fn norm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// exp(dt·A) = [ e^{dt μ/s} T_m((dt/s)(A - μ)) ]^s with T_m the Taylor polynomial of degree m.
struct Propagator {
    dt: f64,
    mu: f64,
    norm1: f64,
    m: usize,
    s: usize,
}

impl Propagator {
    /// (m, s) minimize the number of products m·⌈dt ||A - μ||_1 / θ_m⌉. The exact 1-norm is
    /// used instead of estimates of ||A^p||^{1/p}, which can only overestimate the cost.
    fn new(a: &Csr, dt: f64) -> Self {
        let mu = a.trace() / a.n as f64;
        let norm1 = a.norm1_shifted(mu);
        let (mut m, mut s) = (0, 1);
        if dt * norm1 > 0.0 {
            let mut best = f64::INFINITY;
            for &(mm, theta) in &THETA {
                let ss = (dt * norm1 / theta).ceil().max(1.0);
                if mm as f64 * ss < best {
                    best = mm as f64 * ss;
                    m = mm;
                    s = ss as usize;
                }
            }
        }
        Propagator { dt, mu, norm1, m, s }
    }

    /// v <- exp(dt·A) v. `b` and `w` are work vectors. Returns the number of matrix products.
    fn step(&self, a: &Csr, v: &mut Vec<f64>, b: &mut Vec<f64>, w: &mut Vec<f64>) -> usize {
        let eta = (self.dt * self.mu / self.s as f64).exp();
        let mut products = 0;
        for _ in 0..self.s {
            b.copy_from_slice(v);
            let mut c1 = norm_inf(b);
            for j in 1..=self.m {
                a.mul_shifted(self.mu, self.dt / (self.s * j) as f64, b, w);
                std::mem::swap(b, w);
                products += 1;
                let c2 = norm_inf(b);
                for (f, x) in v.iter_mut().zip(b.iter()) {
                    *f += x;
                }
                if c1 + c2 <= TOL * norm_inf(v) {
                    break;
                }
                c1 = c2;
            }
            for f in v.iter_mut() {
                *f *= eta;
            }
        }
        products
    }
}

// ============================================================================
// Driver
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_args()?;
    let l = cfg.l;
    let steps = (cfg.t_final / cfg.dt).round() as usize;
    let t_all = Instant::now();
    let ch = Chain::new(l, false);
    eprintln!(
        "L = {l}: {} configurations (full basis), {} orbits; T = {}, dt = {}, {} steps",
        ch.configs.len(), ch.reps.len(), cfg.t_final, cfg.dt, steps
    );

    let mut sectors = Vec::new();
    for q in [0, l / 2] {
        let sec = Sector::new(&ch, q);
        let blk = Block::new(&ch, &sec, 1, 1);
        let weight: f64 = blk.rho0.iter().map(|x| x * x).sum();
        eprintln!("Sector Q={q}: dim {} -> block (+,+) {}", sec.dim, blk.dim());
        assert!((weight - 0.5).abs() < 1e-12, "the Néel state is not contained in the (+,+) block");
        sectors.push((sec, blk));
    }

    let dot = |u: &[f64], w: &[f64]| -> f64 { u.iter().zip(w).map(|(a, b)| a * b).sum() };
    let mut file_occupation = BufWriter::new(File::create("occupation_time_alpha.csv")?);
    for (p, alpha) in &cfg.points {
        // the jump tables depend on alpha
        let model = Model::new(&ch, Some(*alpha));
        eprintln!("  gp={} gm={} omega={} alpha={}", p.gp, p.gm, p.omega, alpha);
        let mut obs = vec![(0.0f64, 0.0f64, 0.0f64); steps + 1];
        let mut max_dtr: f64 = 0.0;
        for (sec, blk) in &sectors {
            let n = blk.dim();
            let t0 = Instant::now();
            let (a, max_im) = blk.build_csr(&ch, &model, sec, p);
            let t_build = t0.elapsed().as_secs_f64();
            let prop = Propagator::new(&a, cfg.dt);

            let t0 = Instant::now();
            let mut v = blk.rho0.clone();
            let mut b = vec![0.0; n];
            let mut w = vec![0.0; n];
            let mut products = 0;
            for i in 0..=steps {
                obs[i].0 += dot(&blk.nocc, &v);
                obs[i].1 += dot(&blk.nnn, &v);
                obs[i].2 += dot(&blk.rho0, &v);
                if sec.q == 0 {
                    max_dtr = max_dtr.max((dot(&blk.tr, &v) - 1.0).abs());
                }
                if i < steps {
                    products += prop.step(&a, &mut v, &mut b, &mut w);
                }
            }
            eprintln!(
                "    Q={} dim={:>6} nnz={:>10}  build {:7.2}s (max dropped Im {:.1e})  ||A-μ||_1={:.3} m={} s={}  evolution {:8.2}s ({} products)",
                sec.q, n, a.nnz(), t_build, max_im, prop.norm1, prop.m, prop.s, t0.elapsed().as_secs_f64(), products
            );
        }
        eprintln!("    max |Tr ρ(t) - 1| = {:.1e}", max_dtr);

        let formatted_data = obs
            .iter()
            .map(|(n, nn, fid)| format!("{:.16}, {:.16}, {:.16}", n, nn, fid))
            .collect::<Vec<String>>()
            .join(", ");
        writeln!(file_occupation, "{},{},{},{},{}", alpha, p.gp, p.gm, p.omega, formatted_data)?;
    }
    file_occupation.flush()?;
    eprintln!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}
