use ndarray::{Array1, Array2};
use ndarray_linalg::{Eig, Eigh, SVD};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use sprs::{CsMat, TriMat};
use std::error::Error;

const THRESHOLD: f64 = 1e-10;

#[derive(Debug, Clone)]
struct Basis {
    rep_states: Vec<u64>,
    rep_index: HashMap<u64, Vec<u64>>,
}

// Replaced with full basis: alpha < 1 allows adjacent excitations
fn full_basis(l: usize) -> Vec<u64> {
    (0..(1u64 << l)).collect()
}

fn translationally_invariant_basis(l: usize) -> Basis {
    let states = full_basis(l);
    let mut rep_states = Vec::new();
    let mut rep_index = HashMap::new();
    let mut basis_set: HashSet<u64> = states.into_iter().collect();

    while !basis_set.is_empty() {
        let state = *basis_set.iter().next().unwrap();
        
        let mut shifted_states = HashSet::new();
        for i in 0..l {
            let shifted = ((state << i) | (state >> (l - i))) & ((1u64 << l) - 1);
            shifted_states.insert(shifted);
        }
        
        let min_state = *shifted_states.iter().min().unwrap();
        basis_set = basis_set.difference(&shifted_states).cloned().collect();
        
        let translations: Vec<u64> = (0..l)
            .map(|i| ((min_state << i) | (min_state >> (l - i))) & ((1u64 << l) - 1))
            .collect();
        
        rep_states.push(min_state);
        rep_index.insert(min_state, translations);
    }

    Basis { rep_states, rep_index }
}

fn normalization_factor(l: usize, states: &[u64], k: i64, phases: &[Complex64]) -> f64 {
    let mut norm2 = Complex64::new(0.0, 0.0);
    for j in 0..l {
        for i in 0..l {
            if states[j] == states[i] {
                let idx = (k * j as i64 - k * i as i64).rem_euclid(l as i64);
                norm2 += phases[idx as usize];
            }
        }
    }
    if norm2.norm() <= THRESHOLD {
        0.0
    } else {
        norm2.sqrt().re
    }
}

fn inner_product(
    a_in: &[u64],
    b_out: &[u64],
    k_in: i64,
    k_out: i64,
    l: usize,
    norm_a: f64,
    norm_b: f64,
    phases: &[Complex64],
) -> Complex64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    
    let mut ip = Complex64::new(0.0, 0.0);
    for i in 0..l {
        for j in 0..l {
            if a_in[i] == b_out[j] {
                let idx = (k_out * j as i64 - k_in * i as i64).rem_euclid(l as i64);
                ip += phases[idx as usize];
            }
        }
    }
    
    ip / (norm_a * norm_b)
}

#[derive(Clone)]
struct BasisState {
    states_a: Vec<u64>,
    states_b: Vec<u64>,
    k: i64,
    norm_a: f64,
    norm_b: f64,
}

fn build_lindbladian(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    omega: f64,
    gamma_plus: f64,
    gamma_minus: f64,
    alpha: f64,
    phases: &[Complex64],
) -> CsMat<Complex64> {
    let dim = basis_states.len();
    
    let c_norm = 1.0 / (1.0 + alpha.abs());
    let calc_factor = |state: u64, site_idx: usize| -> f64 {
        if (state & (1u64 << (site_idx % l))) != 0 {
            c_norm * (1.0 - alpha)
        } else {
            c_norm * (1.0 + alpha)
        }
    };

    let triplets: Vec<(usize, usize, Complex64)> = (0..dim)
        .into_iter()
        .flat_map(|j| {
            let mut local_triplets = Vec::new();
            let state_out = &basis_states[j];
            
            for i in 0..dim {
                let state_in = &basis_states[i];
                let mut elem = Complex64::new(0.0, 0.0);
                
                let inner_b = inner_product(
                    &state_out.states_b, &state_in.states_b,
                    state_out.k - q_sector, state_in.k - q_sector,
                    l, state_out.norm_b, state_in.norm_b, phases
                );
                
                let inner_a = inner_product(
                    &state_in.states_a, &state_out.states_a,
                    state_in.k, state_out.k,
                    l, state_in.norm_a, state_out.norm_a, phases
                );

                for site in 0..l {
                    let states_a_shifted: Vec<u64> = state_in.states_a.iter().map(|&s| s ^ (1u64 << site)).collect();
                    let states_b_shifted: Vec<u64> = state_in.states_b.iter().map(|&s| s ^ (1u64 << site)).collect();
                    
                    let states_a_in_shifted = states_a_shifted;
                    let states_b_in_shifted = states_b_shifted;
                    let states_b_out_shifted: Vec<u64> = state_out.states_b.iter().map(|&s| s ^ (1u64 << site)).collect();

                    // ===========================
                    // HAMILTONIAN CONTRIBUTION (Maintains Strict Blockade)
                    // ===========================
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_in_shifted[ii] == state_out.states_a[jj];
                            let c2 = (states_a_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            
                            if c1 && c2 && c3 {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_prod += phases[idx as usize];
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_in_shifted[ii] == state_out.states_b[jj];
                            let c2 = (states_b_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_b_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            
                            if c1 && c2 && c3 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * jj as i64 + k_diff_in * ii as i64).rem_euclid(l as i64);
                                b_prod += phases[idx as usize];
                            }
                        }
                    }
                    b_prod /= state_in.norm_b * state_out.norm_b;
                    elem += Complex64::new(0.0, -omega) * (a_prod * inner_b - inner_a * b_prod);

                    // ===========================
                    // DISSIPATION: L_PLUS
                    // ===========================
                    let mut a_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_in_shifted[ii] == state_out.states_a[jj];
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(states_a_in_shifted[ii], site + l - 1) * calc_factor(states_a_in_shifted[ii], site + 1);
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_plus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    a_plus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_out_shifted[ii] == state_in.states_b[jj];
                            let c4 = (states_b_out_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(states_b_out_shifted[ii], site + l - 1) * calc_factor(states_b_out_shifted[ii], site + 1);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                b_plus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    b_plus /= state_in.norm_b * state_out.norm_b;
                    
                    let mut aa_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(state_in.states_a[ii], site + l - 1).powi(2) * calc_factor(state_in.states_a[ii], site + 1).powi(2);
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                aa_plus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    aa_plus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut bb_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_out.states_b[ii] == state_in.states_b[jj];
                            let c4 = (state_out.states_b[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(state_out.states_b[ii], site + l - 1).powi(2) * calc_factor(state_out.states_b[ii], site + 1).powi(2);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                bb_plus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    bb_plus /= state_in.norm_b * state_out.norm_b;
                    
                    elem += gamma_plus * (a_plus * b_plus - 0.5 * aa_plus * inner_b - 0.5 * inner_a * bb_plus);

                    // ===========================
                    // DISSIPATION: L_MINUS
                    // ===========================
                    let mut a_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_in_shifted[ii] == state_out.states_a[jj];
                            let c4 = (states_a_in_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(states_a_in_shifted[ii], site + l - 1) * calc_factor(states_a_in_shifted[ii], site + 1);
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_minus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    a_minus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_out_shifted[ii] == state_in.states_b[jj];
                            let c4 = (state_out.states_b[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(states_b_out_shifted[ii], site + l - 1) * calc_factor(states_b_out_shifted[ii], site + 1);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                b_minus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    b_minus /= state_in.norm_b * state_out.norm_b;
                    
                    let mut aa_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let c4 = (states_a_in_shifted[ii] & (1 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(state_in.states_a[ii], site + l - 1).powi(2) * calc_factor(state_in.states_a[ii], site + 1).powi(2);
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                aa_minus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    aa_minus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut bb_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_out.states_b[ii] == state_in.states_b[jj];
                            let c4 = (states_b_out_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c4 {
                                let p_factor = calc_factor(state_out.states_b[ii], site + l - 1).powi(2) * calc_factor(state_out.states_b[ii], site + 1).powi(2);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                bb_minus += Complex64::new(p_factor, 0.0) * phases[idx as usize];
                            }
                        }
                    }
                    bb_minus /= state_in.norm_b * state_out.norm_b;
                    
                    elem += gamma_minus * (a_minus * b_minus - 0.5 * aa_minus * inner_b - 0.5 * inner_a * bb_minus);
                }
                
                if elem.norm() > THRESHOLD {
                    local_triplets.push((j, i, elem));
                }
            }
            local_triplets
        })
        .collect();
        
    let mut tri_mat = TriMat::new((dim, dim));
    for (row, col, val) in triplets {
        tri_mat.add_triplet(row, col, val);
    }
    
    tri_mat.to_csr()
}

fn occupation_number(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    phases: &[Complex64],
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut n_cal = Array2::<Complex64>::zeros((dim, dim));
    
    let rows: Vec<_> = (0..dim)
        .into_iter()
        .map(|j| {
            let mut row = vec![Complex64::new(0.0, 0.0); dim];
            for i in 0..dim {
                let state_in = &basis_states[i];
                let state_out = &basis_states[j];
                
                let inner_b = inner_product(
                    &state_out.states_b, &state_in.states_b,
                    state_out.k - q_sector, state_in.k - q_sector,
                    l, state_out.norm_b, state_in.norm_b, phases
                );
                
                if inner_b.norm() <= THRESHOLD {
                    continue;
                }

                for site in 0..l {
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let c_occ = (state_in.states_a[ii] & (1u64 << (site % l))) != 0;
                            
                            if c1 && c_occ {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_prod += phases[idx as usize];
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    row[i] += a_prod * inner_b;
                }
            }
            row
        })
        .collect();
    
    for (j, row) in rows.into_iter().enumerate() {
        for (i, val) in row.into_iter().enumerate() {
            n_cal[[j, i]] = val;
        }
    }
    
    n_cal / l as f64
}

fn density_correlation_nnn(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    phases: &[Complex64],
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut corr_cal = Array2::<Complex64>::zeros((dim, dim));
    
    let rows: Vec<_> = (0..dim)
        .into_iter()
        .map(|j| {
            let mut row = vec![Complex64::new(0.0, 0.0); dim];
            for i in 0..dim {
                let state_in = &basis_states[i];
                let state_out = &basis_states[j];
                
                let inner_b = inner_product(
                    &state_out.states_b, &state_in.states_b,
                    state_out.k - q_sector, state_in.k - q_sector,
                    l, state_out.norm_b, state_in.norm_b, phases
                );
                
                if inner_b.norm() <= THRESHOLD {
                    continue;
                }

                for site in 0..l {
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    let idx_minus = if site == 0 { l - 1 } else { site - 1 };
                    let idx_plus = if site == l - 1 { 0 } else { site + 1 };

                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let mask = (1u64 << idx_minus) | (1u64 << idx_plus);
                            let c_corr = (state_in.states_a[ii] & mask) == mask;
                            
                            if c1 && c_corr {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_prod += phases[idx as usize];
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    row[i] += a_prod * inner_b;
                }
            }
            row
        })
        .collect();
    
    for (j, row) in rows.into_iter().enumerate() {
        for (i, val) in row.into_iter().enumerate() {
            corr_cal[[j, i]] = val;
        }
    }
    
    corr_cal / l as f64
}

fn compute_trace(
    l: usize,
    basis_states: &[BasisState],
    vectorized_op: &Array1<Complex64>,
    q_sector: i64,
    phases: &[Complex64],
) -> Complex64 {
    if q_sector != 0 {
        return Complex64::new(0.0, 0.0);
    }

    vectorized_op.as_slice().unwrap()
        .par_iter()
        .enumerate()
        .map(|(i, &coeff)| {
            if coeff.norm() <= THRESHOLD {
                return Complex64::new(0.0, 0.0);
            }
            let state = &basis_states[i];
            let overlap = inner_product(
                &state.states_a, &state.states_b,
                state.k, state.k, 
                l, state.norm_a, state.norm_b, phases
            );

            coeff * overlap
        })
        .sum()
}

fn compute_eigenstate_observable(
    eigenvector: &Array1<Complex64>, 
    basis_observables: &[f64], 
) -> f64 {
    let mut obs_sum = 0.0;
    let mut norm_sum = 0.0;
    
    for (i, &coeff) in eigenvector.iter().enumerate() {
        let prob = coeff.norm_sqr();
        obs_sum += prob * basis_observables[i];
        norm_sum += prob;
    }

    if norm_sum > 1e-16 {
        obs_sum / norm_sum
    } else {
        0.0
    }
}

fn steady_state_properties(
    l: usize,
    basis_states: &[BasisState],
    vectorized_op: &Array1<Complex64>,
) -> Vec<(f64, f64)> {
    
    let mut results = Vec::new();
    let mut current_offset = 0;
    let mut i = 0;

    while i < basis_states.len() {
        let current_k = basis_states[i].k;
        let mut block_len = 0;
        while i + block_len < basis_states.len() && basis_states[i + block_len].k == current_k {
            block_len += 1;
        }

        let dim = (block_len as f64).sqrt() as usize;

        if dim > 0 {
            let mut rho_block = Array2::<Complex64>::zeros((dim, dim));
            for row in 0..dim {
                for col in 0..dim {
                    let vec_idx = current_offset + (row * dim + col);
                    rho_block[[row, col]] = vectorized_op[vec_idx];
                }
            }

            if let Ok((eigvals, eigvecs)) = rho_block.eigh(ndarray_linalg::UPLO::Upper) {
                let basis_occupations: Vec<f64> = (0..dim).map(|n| {
                    let diag_idx = current_offset + (n * dim + n);
                    let state = &basis_states[diag_idx];
                    let particle_count = state.states_a[0].count_ones() as f64;
                    particle_count / (l as f64)
                }).collect();

                for (idx, &lambda) in eigvals.iter().enumerate() {
                    let eigenvector = eigvecs.column(idx).to_owned();
                    let occ = compute_eigenstate_observable(&eigenvector, &basis_occupations);
                    results.push((lambda, occ));
                }
            }
        }

        current_offset += block_len;
        i += block_len;
    }

    let trace: f64 = results.iter().map(|(lam, _)| lam).sum();
    if trace.abs() > 1e-15 {
        for (lam, _) in results.iter_mut() {
            *lam /= trace;
        }
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[derive(Debug, Clone)]
pub struct SpectralData {
    pub real_eigenvalue: f64,
    pub abs_imag_eigenvalue: f64,
    pub c_k: f64,
    pub s_k: f64,
    pub occ_c: f64,
    pub occ_s: f64,
    pub block_size: usize,
}

fn analyze_lindbladian(
    eigenvalues: &Array1<Complex64>,
    eigenvectors: &Array2<Complex64>,
    occ_op: &Array2<Complex64>,
    rho: &Array1<Complex64>,
    tol: f64,
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    phases: &[Complex64],
) -> Result<Vec<SpectralData>, Box<dyn Error>> {
    
    let evals = eigenvalues;
    let evecs = eigenvectors;
    
    let mut tagged_evals: Vec<(usize, Complex64)> = evals
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();

    tagged_evals.sort_by(|a, b| {
        b.1.re.partial_cmp(&a.1.re).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut results = Vec::new();
    let n = evecs.nrows();
    let mut i = 0;

    while i < tagged_evals.len() {
        let current_val = tagged_evals[i].1;
        let mut cluster_indices = vec![tagged_evals[i].0];
        
        let mut j = i + 1;
        while j < tagged_evals.len() {
            if (tagged_evals[j].1 - current_val).norm() < tol {
                cluster_indices.push(tagged_evals[j].0);
                j += 1;
            } else {
                break;
            }
        }
        
        let m_a = cluster_indices.len();

        if current_val.im < -tol {
            i = j;
            continue;
        }

        let mut raw_subspace = Array2::<Complex64>::zeros((n, m_a));
        for (col_idx, &eig_idx) in cluster_indices.iter().enumerate() {
            let vec = evecs.column(eig_idx);
            raw_subspace.column_mut(col_idx).assign(&vec);
        }

        let (_, sigma, _) = raw_subspace.svd(false, false)?;
        let rank_tol = 1e-5; 
        let m_g = sigma.iter().filter(|&&s| s > rank_tol).count();

        let (u_opt, _, _) = raw_subspace.svd(true, false)?;
        let u = u_opt.ok_or("SVD U calculation failed")?;

        if m_g == m_a {
            for k in 0..m_g {
                let u_k = u.column(k);
                
                let a_k = (&u_k).mapv(|x| x.conj()).dot(rho);
                let aux_vec = occ_op.dot(&u_k);
                let o_k = compute_trace(l, basis_states, &aux_vec, q_sector, phases);
                
                let c_k; let s_k; let occ_c; let occ_s;
                
                if current_val.im <= tol {
                    c_k = a_k.re;
                    s_k = 0.0;
                    occ_c = o_k.re;
                    occ_s = 0.0;
                } else {
                    c_k = 2.0 * a_k.re;
                    s_k = -2.0 * a_k.im;
                    occ_c = 2.0 * o_k.re;
                    occ_s = -2.0 * o_k.im;
                }
                
                results.push(SpectralData {
                    real_eigenvalue: current_val.re, abs_imag_eigenvalue: current_val.im.abs(),
                    c_k, s_k, occ_c, occ_s, block_size: 1,
                });
            }
        } else {
            let mut c_k_total = 0.0f64; let mut s_k_total = 0.0f64;
            let mut occ_c_total = 0.0f64; let mut occ_s_total = 0.0f64;
            
            for k in 0..m_g {
                let u_k = u.column(k);
                let a_k = (&u_k).mapv(|x| x.conj()).dot(rho);
                let aux_vec = occ_op.dot(&u_k);
                let o_k = compute_trace(l, basis_states, &aux_vec, q_sector, phases);
                
                if current_val.im <= tol {
                    c_k_total += a_k.re.powi(2);
                    occ_c_total += o_k.re.powi(2);
                } else {
                    c_k_total += (2.0 * a_k.re).powi(2);
                    s_k_total += (-2.0 * a_k.im).powi(2);
                    occ_c_total += (2.0 * o_k.re).powi(2);
                    occ_s_total += (-2.0 * o_k.im).powi(2);
                }
            }
            
            results.push(SpectralData {
                real_eigenvalue: current_val.re, abs_imag_eigenvalue: current_val.im.abs(),
                c_k: c_k_total.sqrt(), s_k: s_k_total.sqrt(),
                occ_c: occ_c_total.sqrt(), occ_s: occ_s_total.sqrt(), block_size: m_a,
            });
        }
        i = j;
    }
    Ok(results)
}

fn precompute_phases(l: usize) -> Vec<Complex64> {
    (0..l).map(|k| {
        let phase = 2.0 * PI * (k as f64) / (l as f64);
        Complex64::new(0.0, phase).exp()
    }).collect()
}

struct SimulationResult {
    occupation_str: String,
    std_eigenvalues_str: String,
    eigenvalues_str: String,
    decay_str: String,
    oee_str: String,
}

fn compute_rho_dagger_rho(
    expanded_state: &[(u64, u64, Complex64)]
) -> Vec<(u64, u64, Complex64)> {
    let mut a_groups: HashMap<u64, Vec<(u64, Complex64)>> = HashMap::new();
    for &(a, b, c) in expanded_state {
        a_groups.entry(a).or_default().push((b, c));
    }

    let mut rho_sq_map: HashMap<(u64, u64), Complex64> = HashMap::new();

    for b_list in a_groups.values() {
        for &(b, c_ab) in b_list {
            for &(b_prime, c_ab_prime) in b_list {
                let coeff = c_ab.conj() * c_ab_prime;
                *rho_sq_map.entry((b, b_prime)).or_insert(Complex64::new(0.0, 0.0)) += coeff;
            }
        }
    }

    rho_sq_map.into_iter()
        .filter(|(_, c)| c.norm() > 1e-12)
        .map(|((b, b_prime), c)| (b, b_prime, c))
        .collect()
}

fn expand_eigenmatrix(
    l: usize,
    basis_states: &[BasisState],
    eigenvector: &[Complex64],
    phases: &[Complex64],
) -> Vec<(u64, u64, Complex64)> {
    let mut expanded_map: HashMap<(u64, u64), Complex64> = HashMap::new();

    for (idx, &c) in eigenvector.iter().enumerate() {
        if c.norm() <= 1e-12 {
            continue;
        }
        
        let state = &basis_states[idx];
        let norm_factor = state.norm_a * state.norm_b;
        if norm_factor == 0.0 {
            continue;
        }

        for j in 0..l {
            let real_a = state.states_a[j];
            for j_prime in 0..l {
                let real_b = state.states_b[j_prime];

                let phase_idx = (state.k * j as i64 + state.k * j_prime as i64).rem_euclid(l as i64);
                let phase = phases[phase_idx as usize];

                let coeff = c * phase / norm_factor;
                
                *expanded_map.entry((real_a, real_b)).or_insert(Complex64::new(0.0, 0.0)) += coeff;
            }
        }
    }
    
    expanded_map.into_iter().map(|((a, b), c)| (a, b, c)).collect()
}

pub fn full_subsystem_basis(l: usize) -> Vec<u64> {
    (0..(1u64 << l)).collect()
}

pub fn split_state(state: u64, l: usize) -> (u64, u64) {
    let l_a = l / 2;
    let mask_a = (1u64 << l_a) - 1;
    
    let a = state & mask_a;
    let b = state >> l_a;
    
    (a, b)
}

pub fn operator_entanglement_entropy(
    expanded_state: &[(u64, u64, Complex64)],
    l: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let l_a = l / 2;
    let full_a = full_subsystem_basis(l_a);
    
    let mut a_idx_map = HashMap::new();
    for (i, &state) in full_a.iter().enumerate() {
        a_idx_map.insert(state, i);
    }
    
    let dim_a = full_a.len();
    let super_dim = dim_a * dim_a;
    let mut sigma_a = Array2::<Complex64>::zeros((super_dim, super_dim));
    
    let mut b_groups: HashMap<(u64, u64), Vec<((u64, u64), Complex64)>> = HashMap::new();
    
    for &(global_ket, global_bra, coeff) in expanded_state {
        let (a_ket, b_ket) = split_state(global_ket, l);
        let (a_bra, b_bra) = split_state(global_bra, l);
        
        b_groups.entry((b_ket, b_bra))
            .or_default()
            .push(((a_ket, a_bra), coeff));
    }
    
    for (_, terms) in b_groups {
        for &((a_ket1, a_bra1), c1) in &terms {
            for &((a_ket2, a_bra2), c2) in &terms {
                let row_idx = a_idx_map[&a_ket1] * dim_a + a_idx_map[&a_bra1];
                let col_idx = a_idx_map[&a_ket2] * dim_a + a_idx_map[&a_bra2];
                
                sigma_a[[row_idx, col_idx]] += c1 * c2.conj();
            }
        }
    }
    
    let mut trace = Complex64::new(0.0, 0.0);
    for i in 0..super_dim {
        trace += sigma_a[[i, i]];
    }
    
    if trace.norm() < 1e-12 {
        return Ok(0.0); 
    }
    sigma_a /= trace;
    
    let (eigenvalues, _) = sigma_a.eigh(ndarray_linalg::UPLO::Upper)?;
    
    let mut entropy = 0.0;
    for &lambda in eigenvalues.iter() {
        if lambda > 1e-12 {
            entropy -= lambda * lambda.ln();
        }
    }
    
    Ok(entropy)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let l = 8; 
    
    let basis = translationally_invariant_basis(l);
    let phases = precompute_phases(l);
    
    let mut basis_per_sector: HashMap<i64, Vec<u64>> = HashMap::new();
    for k_sector in 0..l as i64 {
        for &state in &basis.rep_states {
            if let Some(translations) = basis.rep_index.get(&state) {
                let norm = normalization_factor(l, translations, k_sector, &phases);
                if norm > 0.0 {
                    basis_per_sector.entry(k_sector).or_insert_with(Vec::new).push(state);
                }
            }
        }
    }
    
    let gp_values = Array1::linspace(0.001, 0.2, 2);
    let gm_values = Array1::linspace(0.001, 0.2, 2);
    let omega_values = Array1::linspace(1.0, 2.0, 1);
    let alpha_values = Array1::linspace(0.0, 1.0, 6);

    let mut parameters = Vec::new();
    for &gp in &gp_values {
        for &gm in &gm_values {
            for &omega in &omega_values {
                for &alpha in &alpha_values {
                    parameters.push((gp, gm, omega, alpha));
                }
            }
        }
    }

    let build_sector = |q_sector: i64| {
        let mut basis_states = Vec::new();
        for k_a in 0..l as i64 {
            let k_b = (k_a - q_sector).rem_euclid(l as i64);
            
            if let (Some(states_in_a), Some(states_in_b)) = (basis_per_sector.get(&k_a), basis_per_sector.get(&k_b)) {
                for &state_a in states_in_a {
                    for &state_b in states_in_b {
                        let states_a_vec = basis.rep_index.get(&state_a).unwrap().clone();
                        let states_b_vec = basis.rep_index.get(&state_b).unwrap().clone();
                        
                        let norm_a = normalization_factor(l, &states_a_vec, k_a, &phases);
                        let norm_b = normalization_factor(l, &states_b_vec, k_b, &phases);
                        
                        basis_states.push(BasisState {
                            states_a: states_a_vec,
                            states_b: states_b_vec,
                            k: k_a,
                            norm_a,
                            norm_b,
                        });
                    }
                }
            }
        }

        let neel_key: u64 = (4u64.pow(l as u32 / 2) - 1) / 3;
        let mut rho_vec_neel = Array1::<Complex64>::zeros(basis_states.len());
        
        let neel_indices: Vec<usize> = basis_states.iter().enumerate()
            .filter(|(_, s)| s.states_a.contains(&neel_key) && s.states_b.contains(&neel_key))
            .map(|(i, _)| i).collect();

        if !neel_indices.is_empty() {
            let initial_weight = 1.0 / neel_indices.len() as f64;
            for &idx in &neel_indices {
                rho_vec_neel[idx] = Complex64::new(initial_weight, 0.0);
            }
        }

        let n_matrix = occupation_number(l, &basis_states, q_sector, &phases);
        let corr_matrix = density_correlation_nnn(l, &basis_states, q_sector, &phases);

        (basis_states, rho_vec_neel, n_matrix, corr_matrix)
    };

    let q_pi = (l / 2) as i64;
    let (basis_0, rho_0, n_mat_0, corr_mat_0) = build_sector(0);
    let (basis_pi, rho_pi, n_mat_pi, corr_mat_pi) = build_sector(q_pi);

    println!("Starting parallel sweep over {} points...", parameters.len());

    let results: Vec<SimulationResult> = parameters
        .par_iter()
        .map(|&(gp, gm, omega, alpha)| {
            let mut res = SimulationResult {
                occupation_str: String::new(),
                std_eigenvalues_str: String::new(),
                eigenvalues_str: String::new(),
                decay_str: String::new(),
                oee_str: String::new(),
            };

            let sectors = [
                (0, &basis_0, &rho_0, &n_mat_0, &corr_mat_0),
                (q_pi, &basis_pi, &rho_pi, &n_mat_pi, &corr_mat_pi)
            ];

            for (q_sector, basis_states, rho_vec_neel, n_matrix, corr_matrix) in sectors {
                let l_cal = build_lindbladian(l, basis_states, q_sector, omega, gp, gm, alpha, &phases);
                
                let l_cal_dense = {
                    let mut dense = Array2::<Complex64>::zeros((basis_states.len(), basis_states.len()));
                    for (val, (row, col)) in l_cal.iter() {
                        dense[[row, col]] = *val;
                    }
                    dense
                };

                if let Ok((evals, evecs)) = l_cal_dense.eig() {
                    
                    res.oee_str.push_str(&format!("{},{},{},{},{}", q_sector, gp, gm, omega, alpha));
                    for (&lambda, vec_view) in evals.iter().zip(evecs.columns()) {
                        let vec_contiguous = vec_view.to_vec();
                        
                        let expanded = expand_eigenmatrix(l, basis_states, &vec_contiguous, &phases);
                        let rho_sq = compute_rho_dagger_rho(&expanded);
                        let entropy = operator_entanglement_entropy(&rho_sq, l).unwrap_or(0.0);
                        
                        res.oee_str.push_str(&format!(",{:.10},{:.10},{:.6}", lambda.re, lambda.im, entropy));
                    }
                    res.oee_str.push('\n');

                    if let Ok(analysis) = analyze_lindbladian(
                        &evals, &evecs, n_matrix, rho_vec_neel, 1e-6, 
                        l, &basis_states, q_sector, &phases
                    ) {
                        res.decay_str.push_str(&format!("{},{},{},{},{}", q_sector, gp, gm, omega, alpha)); 
                        for data in analysis {
                            res.decay_str.push_str(&format!(",{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{}", 
                                data.real_eigenvalue, 
                                data.abs_imag_eigenvalue, 
                                data.c_k, 
                                data.s_k, 
                                data.occ_c, 
                                data.occ_s, 
                                data.block_size
                            ));
                        }
                        res.decay_str.push('\n');
                    }

                    res.eigenvalues_str.push_str(&format!("{},{},{},{},{}", q_sector, gp, gm, omega, alpha));
                    for eval in evals.iter() {
                        res.eigenvalues_str.push_str(&format!(",{},{}", eval.re, eval.im));
                    }
                    res.eigenvalues_str.push('\n');

                    for (i, eval) in evals.iter().enumerate() {
                        if eval.re.abs() < 1e-8 && eval.im.abs() < 1e-8 {
                            let rho_vec = evecs.column(i).to_owned();
                            let tr_rho = compute_trace(l, basis_states, &rho_vec, q_sector, &phases);
                            
                            if tr_rho.norm() > 1e-10 {
                                let spectrum_data = steady_state_properties(l, basis_states, &rho_vec);
                                let formatted_data = spectrum_data.iter()
                                    .map(|(p, n)| format!("{:.16}, {:.6}", p, n))
                                    .collect::<Vec<String>>()
                                    .join(", ");
                                res.std_eigenvalues_str.push_str(&format!("{},{},{},{},{},{}\n", q_sector, gp, gm, omega, alpha, formatted_data));

                                let n_rho = n_matrix.dot(&rho_vec);
                                let nn_rho = corr_matrix.dot(&rho_vec);
                                
                                let tr_n = compute_trace(l, basis_states, &n_rho, q_sector, &phases);
                                let tr_nn = compute_trace(l, basis_states, &nn_rho, q_sector, &phases);

                                let exp_n = (tr_n / tr_rho).re;
                                let exp_nn = (tr_nn / tr_rho).re;
                                res.occupation_str.push_str(&format!("{},{},{},{},{},{},{}\n", q_sector, gp, gm, omega, alpha, exp_n, exp_nn));
                            }
                        }
                    }
                }
            }
            res
        })
        .collect();

    let mut file_occupation = File::create("occupation_alpha.csv")?;
    writeln!(file_occupation, "q_sector,gp,gm,omega,alpha,n,nn")?;
    
    let mut file_std = File::create("std_eigenvalues_alpha.csv")?;
    let mut file_evals = File::create("eigenvalues_alpha.csv")?;
    let mut file_decay = File::create("decay_alpha.csv")?;
    let mut file_oee = File::create("oee_alpha.csv")?;

    for res in results {
        file_occupation.write_all(res.occupation_str.as_bytes())?;
        file_std.write_all(res.std_eigenvalues_str.as_bytes())?;
        file_evals.write_all(res.eigenvalues_str.as_bytes())?;
        file_oee.write_all(res.oee_str.as_bytes())?;
        file_decay.write_all(res.decay_str.as_bytes())?;
    }

    Ok(())
}