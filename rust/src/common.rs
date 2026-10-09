// Shared by the binaries, included textually:  include!("../../common.rs");
//
// Open PXP chain (periodic, L sites, constraint n_j n_{j+1} = 0) with PXP-dressed σ± jumps:
//
//   L[ρ] = -iΩ [H, ρ] + Σ_{γ∈{+,-}} γ Σ_j ( A^γ_j ρ A^γ_j† - ½ {A^γ_j† A^γ_j, ρ} )
//   H = Σ_j P_{j-1} X_j P_{j+1},   A^+_j = P_{j-1} σ^+_j P_{j+1},   A^-_j = P_{j-1} σ^-_j P_{j+1}
//
// Symmetries used (all exact):
//  1. Translations:  sectors ρ_Q = Σ_k P_k ρ P_{k-Q}, basis |n,k><m,k-Q| (Q = 0 and Q = L/2).
//  2. Reflection R: j -> -j mod L.
//  3. S: ρ -> C ρᵀ C with C = Π_j Z_j. Exact because H is real with CHC = -H and the jumps
//     are real with CAC = -A. It breaks if a detuning Δ Σ n_j or complex jumps are added.
//  4. Hermiticity Θ: ρ -> ρ† (antilinear): in a Hermitian operator basis every block is REAL.
// Each Q sector splits into the four blocks (σ, τ) = characters of {1, R, S, RS}.
//
// Conventions:
//   |r,k> = (1/√p) Σ_{d<p} e^{-i2πkd/L} T^d |r>,  T = cyclic shift left by one bit,
//   p = period of the representative r, k allowed iff k·p ≡ 0 (mod L).

use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

struct Params {
    gp: f64,
    gm: f64,
    omega: f64,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

// ============================================================================
// Single-chain structure: Lucas configurations, translation orbits, momenta
// ============================================================================

const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
const ONE: C64 = C64 { re: 1.0, im: 0.0 };
const IMAG: C64 = C64 { re: 0.0, im: 1.0 };
const NONE: u32 = u32::MAX;

struct Chain {
    l: usize,
    mask: u64,
    /// All PBC-constrained (Lucas) configurations, ascending.
    configs: Vec<u64>,
    /// config -> index in `configs` (NONE if not allowed). Size 2^L.
    #[allow(dead_code)]
    conf_index: Vec<u32>,
    /// config -> representative index and shift: config = T^shift rep. Size 2^L.
    conf_rep: Vec<u32>,
    conf_shift: Vec<u8>,
    /// Orbit representatives (smallest member), their periods and orbits (as config indices).
    reps: Vec<u64>,
    period: Vec<usize>,
    #[allow(dead_code)]
    orbit: Vec<Vec<u32>>,
    /// mom[k]: representatives allowed at momentum k; pos[k][rep]: position in mom[k].
    mom: Vec<Vec<u32>>,
    pos: Vec<Vec<u32>>,
    /// Offsets of the single-copy momentum basis (index of |r,k> = ms_off[k] + pos[k][r]).
    ms_off: Vec<usize>,
    /// Reflection j -> -j mod L:  R|rep> = T^{refl_shift} |refl_rep>.
    refl_rep: Vec<u32>,
    refl_shift: Vec<u8>,
    /// e^{i 2π j / L}
    phases: Vec<C64>,
}

impl Chain {
    fn new(l: usize) -> Self {
        assert!(l >= 4 && l % 2 == 0 && l <= 28, "L must be even, 4 <= L <= 28");
        let mask = (1u64 << l) - 1;
        let rot = |s: u64, d: usize| -> u64 {
            let d = d % l;
            if d == 0 { s } else { ((s << d) | (s >> (l - d))) & mask }
        };
        let nconf = 1usize << l;
        let mut configs = Vec::new();
        let mut conf_index = vec![NONE; nconf];
        for s in 0..nconf as u64 {
            let ok = s & (s >> 1) == 0 && !((s & 1) != 0 && (s >> (l - 1)) & 1 != 0);
            if ok {
                conf_index[s as usize] = configs.len() as u32;
                configs.push(s);
            }
        }
        let reps: Vec<u64> = configs
            .iter()
            .copied()
            .filter(|&s| (1..l).all(|d| rot(s, d) >= s))
            .collect();
        let mut conf_rep = vec![NONE; nconf];
        let mut conf_shift = vec![0u8; nconf];
        let mut period = Vec::with_capacity(reps.len());
        let mut orbit = Vec::with_capacity(reps.len());
        for (ri, &r) in reps.iter().enumerate() {
            let p = (1..=l).find(|&d| rot(r, d) == r).unwrap();
            period.push(p);
            let mut orb = Vec::with_capacity(p);
            for d in 0..p {
                let c = rot(r, d) as usize;
                conf_rep[c] = ri as u32;
                conf_shift[c] = d as u8;
                orb.push(conf_index[c]);
            }
            orbit.push(orb);
        }
        let mut mom = vec![Vec::new(); l];
        let mut pos = vec![vec![NONE; reps.len()]; l];
        for k in 0..l {
            for ri in 0..reps.len() {
                if (k * period[ri]) % l == 0 {
                    pos[k][ri] = mom[k].len() as u32;
                    mom[k].push(ri as u32);
                }
            }
        }
        let mut ms_off = vec![0usize; l + 1];
        for k in 0..l {
            ms_off[k + 1] = ms_off[k] + mom[k].len();
        }
        let reflect = |s: u64| -> u64 {
            let rev = s.reverse_bits() >> (64 - l); // j -> L-1-j
            ((rev << 1) | (rev >> (l - 1))) & mask // then j -> L-j (mod L)
        };
        let mut refl_rep = Vec::with_capacity(reps.len());
        let mut refl_shift = Vec::with_capacity(reps.len());
        for &r in &reps {
            let c = reflect(r) as usize;
            assert!(conf_rep[c] != NONE);
            refl_rep.push(conf_rep[c]);
            refl_shift.push(conf_shift[c]);
        }
        let phases = (0..l)
            .map(|j| C64::from_polar(1.0, 2.0 * PI * j as f64 / l as f64))
            .collect();
        Chain {
            l, mask, configs, conf_index, conf_rep, conf_shift, reps, period, orbit, mom, pos,
            ms_off, refl_rep, refl_shift, phases,
        }
    }

    #[inline]
    fn rot(&self, s: u64, d: usize) -> u64 {
        let d = d % self.l;
        if d == 0 { s } else { ((s << d) | (s >> (self.l - d))) & self.mask }
    }

    /// e^{i 2π x / L}
    #[inline]
    fn ph(&self, x: i64) -> C64 {
        self.phases[x.rem_euclid(self.l as i64) as usize]
    }

    #[inline]
    fn ms(&self, k: usize, rep: usize) -> usize {
        self.ms_off[k] + self.pos[k][rep] as usize
    }

    fn n_ms(&self) -> usize {
        self.ms_off[self.l]
    }
}

// ============================================================================
// Single-copy operators in the momentum basis
// ============================================================================

/// table[ms][k'] = list of (rep', <rep',k'| O_0 |ms>) for a site-0 operator O_0
/// that maps configurations to configurations.
type OpTable = Vec<Vec<Vec<(u32, C64)>>>;

fn site_op_table(ch: &Chain, f: impl Fn(u64) -> Option<u64>) -> OpTable {
    let l = ch.l;
    let mut table = Vec::with_capacity(ch.n_ms());
    for k in 0..l {
        for &ri in &ch.mom[k] {
            let r = ri as usize;
            let p = ch.period[r];
            let mut acc: HashMap<(usize, u32), C64> = HashMap::new();
            for d in 0..p {
                let c = ch.rot(ch.reps[r], d);
                if let Some(c2) = f(c) {
                    let r2 = ch.conf_rep[c2 as usize];
                    assert!(r2 != NONE, "operator left the constrained space");
                    let e = ch.conf_shift[c2 as usize] as i64;
                    let p2 = ch.period[r2 as usize];
                    let amp = ch.ph(-(k as i64) * d as i64) / ((p * p2) as f64).sqrt();
                    for k2 in 0..l {
                        if (k2 * p2) % l == 0 {
                            *acc.entry((k2, r2)).or_insert(ZERO) += amp * ch.ph(k2 as i64 * e);
                        }
                    }
                }
            }
            let mut by_k = vec![Vec::new(); l];
            for ((k2, r2), v) in acc {
                if v.norm() > 1e-13 {
                    by_k[k2].push((r2, v));
                }
            }
            for v in by_k.iter_mut() {
                v.sort_by_key(|x| x.0);
            }
            table.push(by_k);
        }
    }
    table
}

struct Model {
    /// h[ms] = (rep', <rep',k|H|ms>) with H = Σ_j PXP_j (same k only).
    h: Vec<Vec<(u32, C64)>>,
    /// Site-0 jump operators A^+_0, A^-_0 in the momentum basis.
    a_plus: OpTable,
    a_minus: OpTable,
    /// Σ_j A^±_j† A^±_j is diagonal: eigenvalue per representative.
    kappa_plus: Vec<f64>,
    kappa_minus: Vec<f64>,
}

impl Model {
    fn new(ch: &Chain) -> Self {
        let l = ch.l;
        let nb_empty = |c: u64| (c >> 1) & 1 == 0 && (c >> (l - 1)) & 1 == 0;
        let h_tab = site_op_table(ch, |c| if nb_empty(c) { Some(c ^ 1) } else { None });
        let a_plus = site_op_table(ch, |c| if nb_empty(c) && c & 1 == 0 { Some(c | 1) } else { None });
        let a_minus = site_op_table(ch, |c| if nb_empty(c) && c & 1 == 1 { Some(c & !1) } else { None });
        // H = Σ_j T^j h_0 T^-j  =>  <r',k|H|r,k> = L <r',k|h_0|r,k>
        let mut h = Vec::with_capacity(ch.n_ms());
        for k in 0..l {
            for &ri in &ch.mom[k] {
                let ms = ch.ms(k, ri as usize);
                h.push(h_tab[ms][k].iter().map(|&(r, v)| (r, v * l as f64)).collect());
            }
        }
        let bit = |c: u64, j: usize| (c >> (j % l)) & 1;
        let mut kappa_plus = Vec::new();
        let mut kappa_minus = Vec::new();
        for &r in &ch.reps {
            let mut kp = 0.0;
            let mut km = 0.0;
            for j in 0..l {
                let nb0 = bit(r, j + l - 1) == 0 && bit(r, j + 1) == 0;
                if nb0 && bit(r, j) == 0 { kp += 1.0; }
                if nb0 && bit(r, j) == 1 { km += 1.0; }
            }
            kappa_plus.push(kp);
            kappa_minus.push(km);
        }
        Model { h, a_plus, a_minus, kappa_plus, kappa_minus }
    }

    /// Applies the Lindbladian to the sector basis element `s` and reports every
    /// nonzero output component (sector index, value) through `out` (duplicates possible).
    #[inline]
    fn apply<F: FnMut(usize, C64)>(&self, ch: &Chain, sec: &Sector, s: usize, p: &Params, out: &mut F) {
        let l = ch.l;
        let q = sec.q;
        let (k, n, m) = sec.elems[s];
        let (k, n, m) = (k as usize, n as usize, m as usize);
        let kb = (k + l - q) % l;
        let msa = ch.ms(k, n);
        let msb = ch.ms(kb, m);

        // -iΩ (Hρ - ρH)
        let mi = C64::new(0.0, -p.omega);
        for &(r, v) in &self.h[msa] {
            out(sec.index(ch, k, r as usize, m), mi * v);
        }
        for &(r, v) in &self.h[msb] {
            out(sec.index(ch, k, n, r as usize), -mi * v.conj());
        }
        // -½ {K, ρ},  K = Σ_j (γ+ A+†A+ + γ- A-†A-)  (diagonal)
        let diag = -0.5
            * (p.gp * (self.kappa_plus[n] + self.kappa_plus[m])
                + p.gm * (self.kappa_minus[n] + self.kappa_minus[m]));
        out(s, C64::new(diag, 0.0));
        // Σ_j A_j ρ A_j† = L · P_Q[A_0 ρ A_0†]
        for (g, tab) in [(p.gp, &self.a_plus), (p.gm, &self.a_minus)] {
            if g == 0.0 {
                continue;
            }
            let pref = g * l as f64;
            for k1 in 0..l {
                let la = &tab[msa][k1];
                if la.is_empty() {
                    continue;
                }
                let k2 = (k1 + l - q) % l;
                let lb = &tab[msb][k2];
                if lb.is_empty() {
                    continue;
                }
                for &(r1, va) in la {
                    let base = sec.off[k1] + ch.pos[k1][r1 as usize] as usize * sec.nb[k1];
                    let pa = pref * va;
                    for &(r2, vb) in lb {
                        out(base + ch.pos[k2][r2 as usize] as usize, pa * vb.conj());
                    }
                }
            }
        }
    }
}

// ============================================================================
// Q sector: basis |n,k><m,k-Q|, reflection, S and Hermiticity partners
// ============================================================================

struct Sector {
    q: usize,
    dim: usize,
    off: Vec<usize>,
    nb: Vec<usize>,
    /// (k, ket rep n, bra rep m) for every sector index.
    elems: Vec<(u8, u32, u32)>,
    /// Reflection: R|s> = chi[s] |part[s]>.
    part: Vec<u32>,
    chi: Vec<C64>,
    /// S (ρ -> C ρᵀ C): S|s> = schi[s] |spart[s]>,
    /// S|n,k><m,k-Q| = (-1)^{|n|+|m|} |m,Q-k><n,-k|  (no momentum phase).
    spart: Vec<u32>,
    schi: Vec<f64>,
    /// Hermiticity: (|s>)† = |bar[s]>  (no phase).
    bar: Vec<u32>,
}

impl Sector {
    fn new(ch: &Chain, q: usize) -> Self {
        let l = ch.l;
        assert!((2 * q) % l == 0, "only Q = 0 and Q = L/2 are self-conjugate sectors");
        let mut off = vec![0usize; l + 1];
        let mut nb = vec![0usize; l];
        let mut elems = Vec::new();
        for k in 0..l {
            let kb = (k + l - q) % l;
            off[k] = elems.len();
            nb[k] = ch.mom[kb].len();
            for &n in &ch.mom[k] {
                for &m in &ch.mom[kb] {
                    elems.push((k as u8, n, m));
                }
            }
        }
        off[l] = elems.len();
        let dim = elems.len();
        let mut sec = Sector {
            q, dim, off, nb, elems, part: vec![0; dim], chi: vec![ZERO; dim],
            spart: vec![0; dim], schi: vec![0.0; dim], bar: vec![0; dim],
        };
        for s in 0..dim {
            let (k, n, m) = sec.elems[s];
            let (k, n, m) = (k as usize, n as usize, m as usize);
            let kb = (k + l - q) % l;
            let kr = (l - k) % l;
            sec.part[s] = sec.index(ch, kr, ch.refl_rep[n] as usize, ch.refl_rep[m] as usize) as u32;
            let x = k as i64 * ch.refl_shift[n] as i64 - kb as i64 * ch.refl_shift[m] as i64;
            sec.chi[s] = ch.ph(-x);
            let ks = (q + l - k) % l;
            sec.spart[s] = sec.index(ch, ks, m, n) as u32;
            let odd = (ch.reps[n].count_ones() + ch.reps[m].count_ones()) % 2 == 1;
            sec.schi[s] = if odd { -1.0 } else { 1.0 };
            sec.bar[s] = sec.index(ch, kb, m, n) as u32;
        }
        sec
    }

    #[inline]
    fn index(&self, ch: &Chain, k: usize, n: usize, m: usize) -> usize {
        let kb = (k + ch.l - self.q) % ch.l;
        self.off[k] + ch.pos[k][n] as usize * self.nb[k] + ch.pos[kb][m] as usize
    }
}

// ============================================================================
// Symmetry block (σ, τ) with a real (Hermitian) basis
// ============================================================================

struct Block {
    /// Characters of R and S.
    #[allow(dead_code)]
    sigma: i32,
    #[allow(dead_code)]
    tau: i32,
    /// Real basis vectors e_α as sparse combinations of sector basis elements.
    vecs: Vec<Vec<(u32, C64)>>,
    /// For each sector index t: the (≤2) real basis vectors containing it, with e_α[t].
    rm_len: Vec<u8>,
    rm_idx: Vec<[u32; 2]>,
    rm_val: Vec<[C64; 2]>,
    /// Real-basis representation of linear functionals / the initial state.
    tr: Vec<f64>,   // Tr(e_α)
    nocc: Vec<f64>, // Tr(n e_α),        n  = (1/L) Σ_j n_j
    nnn: Vec<f64>,  // Tr(nn e_α),       nn = (1/L) Σ_j n_{j-1} n_{j+1}
    rho0: Vec<f64>, // <<e_α|ρ0>>       (Néel)
}

impl Block {
    fn new(ch: &Chain, sec: &Sector, sigma: i32, tau: i32) -> Self {
        let dim = sec.dim;
        let (sg, tg) = (sigma as f64, tau as f64);
        // --- group {1, R, S, RS}:  g|s> = c |t>  for g = R^a S^b
        let act = |s: usize, a: bool, b: bool| -> (usize, C64) {
            let (mut t, mut c) = (s, ONE);
            if b {
                c *= sec.schi[t];
                t = sec.spart[t] as usize;
            }
            if a {
                c *= sec.chi[t];
                t = sec.part[t] as usize;
            }
            (t, c)
        };
        let group = [(false, false, 1.0), (true, false, sg), (false, true, tg), (true, true, sg * tg)];
        // --- symmetrized states |u,σ,τ> ∝ Σ_g χ(g) g|u>: one per group orbit that carries (σ, τ)
        let mut seen = vec![false; dim];
        let mut owner = vec![NONE; dim];
        let mut sym: Vec<Vec<(u32, C64)>> = Vec::new();
        for u in 0..dim {
            if seen[u] {
                continue;
            }
            let mut comp: Vec<(u32, C64)> = Vec::with_capacity(4);
            for &(a, b, chr) in &group {
                let (t, c) = act(u, a, b);
                seen[t] = true;
                match comp.iter_mut().find(|x| x.0 == t as u32) {
                    Some(x) => x.1 += chr * c,
                    None => comp.push((t as u32, chr * c)),
                }
            }
            let norm = comp.iter().map(|x| x.1.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-8 {
                continue;
            }
            for x in comp.iter_mut() {
                x.1 /= norm;
                owner[x.0 as usize] = sym.len() as u32;
            }
            sym.push(comp);
        }
        let coeff = |a: usize, t: u32| -> C64 { sym[a].iter().find(|x| x.0 == t).map_or(ZERO, |x| x.1) };
        // --- Θ (ρ -> ρ†) on the symmetrized states:  Θ|a> = φ |b>
        let mut vecs: Vec<Vec<(u32, C64)>> = Vec::with_capacity(sym.len());
        for a in 0..sym.len() {
            let (t0, c0) = sym[a][0];
            let tb = sec.bar[t0 as usize];
            assert!(owner[tb as usize] != NONE, "Hermiticity partner missing");
            let b = owner[tb as usize] as usize;
            let phi = c0.conj() / coeff(b, tb);
            for &(t, c) in &sym[a] {
                let d = c.conj() - phi * coeff(b, sec.bar[t as usize]);
                assert!(d.norm() < 1e-10, "Θ does not map the block onto itself");
            }
            if b < a {
                continue;
            }
            if b == a {
                let rot = C64::from_polar(1.0, 0.5 * phi.arg());
                vecs.push(sym[a].iter().map(|&(t, c)| (t, c * rot)).collect());
            } else {
                let mut er = Vec::with_capacity(8);
                let mut ei = Vec::with_capacity(8);
                for &(t, c) in &sym[a] {
                    er.push((t, c * FRAC_1_SQRT_2));
                    ei.push((t, IMAG * c * FRAC_1_SQRT_2));
                }
                for &(t, c) in &sym[b] {
                    er.push((t, phi * c * FRAC_1_SQRT_2));
                    ei.push((t, -IMAG * phi * c * FRAC_1_SQRT_2));
                }
                vecs.push(er);
                vecs.push(ei);
            }
        }
        assert_eq!(vecs.len(), sym.len());
        // --- row map
        let mut rm_len = vec![0u8; dim];
        let mut rm_idx = vec![[0u32; 2]; dim];
        let mut rm_val = vec![[ZERO; 2]; dim];
        for (a, e) in vecs.iter().enumerate() {
            for &(t, c) in e {
                let t = t as usize;
                let j = rm_len[t] as usize;
                assert!(j < 2);
                rm_idx[t][j] = a as u32;
                rm_val[t][j] = c;
                rm_len[t] += 1;
            }
        }
        // --- functionals on the sector basis
        let l = ch.l;
        let neel: u64 = (0..l / 2).map(|j| 1u64 << (2 * j)).sum();
        let neel_rep = ch.conf_rep[neel as usize];
        let neel_elems: Vec<usize> = (0..dim)
            .filter(|&s| sec.elems[s].1 == neel_rep && sec.elems[s].2 == neel_rep)
            .collect();
        let mut g = vec![0.0; dim];
        let mut f = vec![0.0; dim];
        let mut ff = vec![0.0; dim];
        let mut r0 = vec![0.0; dim];
        for s in 0..dim {
            let (_, n, m) = sec.elems[s];
            if sec.q == 0 && n == m {
                let r = ch.reps[n as usize];
                g[s] = 1.0;
                f[s] = r.count_ones() as f64 / l as f64;
                ff[s] = (0..l).filter(|&j| (r >> ((j + l - 1) % l)) & 1 == 1 && (r >> ((j + 1) % l)) & 1 == 1).count() as f64
                    / l as f64;
            }
        }
        for &s in &neel_elems {
            r0[s] = 1.0 / neel_elems.len() as f64;
        }
        let lin = |w: &[f64], conj: bool| -> Vec<f64> {
            vecs.iter()
                .map(|e| e.iter().map(|&(t, c)| if conj { c.conj() } else { c } * w[t as usize]).sum::<C64>().re)
                .collect()
        };
        let tr = lin(&g, false);
        let nocc = lin(&f, false);
        let nnn = lin(&ff, false);
        let rho0 = lin(&r0, true);
        Block { sigma, tau, vecs, rm_len, rm_idx, rm_val, tr, nocc, nnn, rho0 }
    }

    fn dim(&self) -> usize {
        self.vecs.len()
    }

    /// Fills `a` (column-major dim×dim) with the real Lindbladian block.
    /// Returns the largest discarded imaginary part (should be ~1e-15).
    #[allow(dead_code)]
    fn build_matrix(&self, ch: &Chain, model: &Model, sec: &Sector, p: &Params, a: &mut [f64]) -> f64 {
        let n = self.dim();
        a.par_chunks_mut(n)
            .enumerate()
            .map_init(
                || vec![ZERO; n],
                |acc, (beta, col)| {
                    for &(s, cs) in &self.vecs[beta] {
                        model.apply(ch, sec, s as usize, p, &mut |t, v| {
                            let w = cs * v;
                            for j in 0..self.rm_len[t] as usize {
                                acc[self.rm_idx[t][j] as usize] += self.rm_val[t][j].conj() * w;
                            }
                        });
                    }
                    let mut max_im: f64 = 0.0;
                    for (x, c) in col.iter_mut().zip(acc.iter_mut()) {
                        *x = c.re;
                        max_im = max_im.max(c.im.abs());
                        *c = ZERO;
                    }
                    max_im
                },
            )
            .reduce(|| 0.0, f64::max)
    }

    /// The same real Lindbladian block as a sparse matrix (entries below 1e-15 are dropped).
    /// Returns the matrix and the largest discarded imaginary part (should be ~1e-15).
    #[allow(dead_code)]
    fn build_csr(&self, ch: &Chain, model: &Model, sec: &Sector, p: &Params) -> (Csr, f64) {
        let n = self.dim();
        let cols: Vec<(Vec<(u32, f64)>, f64)> = (0..n)
            .into_par_iter()
            .map_init(
                || (vec![ZERO; n], vec![false; n], Vec::<u32>::new()),
                |(acc, hit, rows), beta| {
                    for &(s, cs) in &self.vecs[beta] {
                        model.apply(ch, sec, s as usize, p, &mut |t, v| {
                            let w = cs * v;
                            for j in 0..self.rm_len[t] as usize {
                                let a = self.rm_idx[t][j] as usize;
                                if !hit[a] {
                                    hit[a] = true;
                                    rows.push(a as u32);
                                }
                                acc[a] += self.rm_val[t][j].conj() * w;
                            }
                        });
                    }
                    rows.sort_unstable();
                    let mut col = Vec::with_capacity(rows.len());
                    let mut max_im: f64 = 0.0;
                    for &a in rows.iter() {
                        let c = acc[a as usize];
                        if c.re.abs() > 1e-15 {
                            col.push((a, c.re));
                        }
                        max_im = max_im.max(c.im.abs());
                        acc[a as usize] = ZERO;
                        hit[a as usize] = false;
                    }
                    rows.clear();
                    (col, max_im)
                },
            )
            .collect();
        let max_im = cols.iter().map(|c| c.1).fold(0.0, f64::max);
        let mut indptr = vec![0usize; n + 1];
        for (col, _) in &cols {
            for &(a, _) in col {
                indptr[a as usize + 1] += 1;
            }
        }
        for a in 0..n {
            indptr[a + 1] += indptr[a];
        }
        let mut next = indptr.clone();
        let mut indices = vec![0u32; indptr[n]];
        let mut data = vec![0.0; indptr[n]];
        for (beta, (col, _)) in cols.iter().enumerate() {
            for &(a, x) in col {
                let i = next[a as usize];
                indices[i] = beta as u32;
                data[i] = x;
                next[a as usize] += 1;
            }
        }
        (Csr { n, indptr, indices, data }, max_im)
    }

    /// Real-basis coordinates y -> sector-basis vector.
    #[allow(dead_code)]
    fn to_sector(&self, dim: usize, y: &[C64]) -> Vec<C64> {
        let mut v = vec![ZERO; dim];
        for (e, &ya) in self.vecs.iter().zip(y) {
            if ya == ZERO {
                continue;
            }
            for &(t, c) in e {
                v[t as usize] += ya * c;
            }
        }
        v
    }
}

// ============================================================================
// Real sparse matrix (CSR)
// ============================================================================

#[allow(dead_code)]
struct Csr {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<f64>,
}

#[allow(dead_code)]
impl Csr {
    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn trace(&self) -> f64 {
        (0..self.n)
            .map(|a| {
                let r = self.indptr[a]..self.indptr[a + 1];
                self.indices[r.clone()].iter().zip(&self.data[r]).filter(|x| *x.0 as usize == a).map(|x| *x.1).sum::<f64>()
            })
            .sum()
    }

    /// 1-norm (largest absolute column sum) of A - μ·1.
    fn norm1_shifted(&self, mu: f64) -> f64 {
        let mut sums = vec![0.0; self.n];
        let mut diag = vec![false; self.n];
        for a in 0..self.n {
            for i in self.indptr[a]..self.indptr[a + 1] {
                let b = self.indices[i] as usize;
                if a == b {
                    diag[a] = true;
                    sums[b] += (self.data[i] - mu).abs();
                } else {
                    sums[b] += self.data[i].abs();
                }
            }
        }
        for a in 0..self.n {
            if !diag[a] {
                sums[a] += mu.abs();
            }
        }
        sums.into_iter().fold(0.0, f64::max)
    }

    /// y = scale · (A - μ·1) x
    fn mul_shifted(&self, mu: f64, scale: f64, x: &[f64], y: &mut [f64]) {
        // rows per task such that a task holds >= ~2^16 entries (small matrices run sequentially)
        let min_len = ((self.n << 16) / self.nnz().max(1)).max(1);
        y.par_iter_mut().enumerate().with_min_len(min_len).for_each(|(a, ya)| {
            let r = self.indptr[a]..self.indptr[a + 1];
            let mut acc = -mu * x[a];
            for (&b, &v) in self.indices[r.clone()].iter().zip(&self.data[r]) {
                acc += v * x[b as usize];
            }
            *ya = scale * acc;
        });
    }
}
