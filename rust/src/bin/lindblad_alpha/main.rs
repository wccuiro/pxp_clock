use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use sprs::{CsMat, TriMat};

const THRESHOLD: f64 = 1e-10;

#[derive(Debug, Clone)]
struct Basis {
    rep_states: Vec<u64>,
    rep_index: HashMap<u64, Vec<u64>>,
}

// Replaced fibonacci_basis with full computational basis
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

fn normalization_factor(l: usize, states: &[u64], k: i64) -> f64 {
    let mut norm2 = Complex64::new(0.0, 0.0);
    for j in 0..l {
        for i in 0..l {
            if states[j] == states[i] {
                let phase = 2.0 * PI * (k as f64) * ((j as i64 - i as i64) as f64) / (l as f64);
                norm2 += Complex64::new(0.0, phase).exp();
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
) -> Complex64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    
    let mut ip = Complex64::new(0.0, 0.0);
    for i in 0..l {
        for j in 0..l {
            if a_in[i] == b_out[j] {
                let phase = 2.0 * PI * ((k_out * j as i64 - k_in * i as i64) as f64) / (l as f64);
                ip += Complex64::new(0.0, phase).exp();
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
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
                );
                
                let inner_a = inner_product(
                    &state_in.states_a,
                    &state_out.states_a,
                    state_in.k,
                    state_out.k,
                    l,
                    state_in.norm_a,
                    state_out.norm_a,
                );

                for site in 0..l {
                    let states_a_shifted: Vec<u64> = state_in.states_a
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    let states_b_shifted: Vec<u64> = state_in.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    
                    let states_a_in_shifted = states_a_shifted;
                    let states_b_in_shifted = states_b_shifted;
                    let states_b_out_shifted: Vec<u64> = state_out.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();

                    // ===========================
                    // HAMILTONIAN CONTRIBUTION (Strict Blockade)
                    // ===========================
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_in_shifted[ii] == state_out.states_a[jj];
                            let c2 = (states_a_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            
                            if c1 && c2 && c3 { // Strict boolean check restored
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_prod += Complex64::new(0.0, phase).exp();
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
                            
                            if c1 && c2 && c3 { // Strict boolean check restored
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * jj as i64 + k_diff_in * ii as i64) as f64) / (l as f64);
                                b_prod += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_plus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_plus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let p_factor = calc_factor(state_in.states_a[ii], site + l - 1) * calc_factor(state_in.states_a[ii], site + 1);
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_plus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let p_factor = calc_factor(state_out.states_b[ii], site + l - 1) * calc_factor(state_out.states_b[ii], site + 1);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_plus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_minus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_minus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let p_factor = calc_factor(state_in.states_a[ii], site + l - 1) * calc_factor(state_in.states_a[ii], site + 1);
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_minus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                                let p_factor = calc_factor(state_out.states_b[ii], site + l - 1) * calc_factor(state_out.states_b[ii], site + 1);
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_minus += Complex64::new(p_factor, 0.0) * Complex64::new(0.0, phase).exp();
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
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_prod += Complex64::new(0.0, phase).exp();
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
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_prod += Complex64::new(0.0, phase).exp();
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
) -> Complex64 {
    if q_sector != 0 {
        return Complex64::new(0.0, 0.0);
    }

    vectorized_op.as_slice().unwrap()
        .iter()
        .enumerate()
        .map(|(i, &coeff)| {
            if coeff.norm() <= THRESHOLD {
                return Complex64::new(0.0, 0.0);
            }

            let state = &basis_states[i];
            let overlap = inner_product(
                &state.states_a,
                &state.states_b,
                state.k,
                state.k, 
                l,
                state.norm_a,
                state.norm_b,
            );

            coeff * overlap
        })
        .sum()
}

fn obs_evolution(
    l: usize,
    n_matrix: &Array2<Complex64>,
    corr_matrix: &Array2<Complex64>,
    basis_states: &[BasisState],
    q_sector: i64,
    initial: &Array1<Complex64>,
    lindbladian: &Array2<Complex64>,
    neel_indices: &[usize], 
    dt: f64,
    t_final: f64,
) -> Array1<(f64, f64, f64)> {
    let num_steps = (t_final / dt).round() as usize;
    let mut observables = Vec::with_capacity(num_steps + 1);
    let mut current_rho = initial.clone();

    for _step in 0..=num_steps {
        let obs_state = n_matrix.dot(&current_rho);
        let expectation_val = compute_trace(l, basis_states, &obs_state, q_sector);
        
        let corr_state = corr_matrix.dot(&current_rho);
        let corr_val = compute_trace(l, basis_states, &corr_state, q_sector);

        let fidelity_val: f64 = neel_indices.iter()
            .map(|&idx| initial[idx].re * current_rho[idx].re)
            .sum();

        observables.push((expectation_val.re, corr_val.re, fidelity_val));

        let derivative = lindbladian.dot(&current_rho);
        current_rho = current_rho + &derivative * dt;
        
        let trace = compute_trace(l, basis_states, &current_rho, q_sector);
        if trace.norm() > 1e-15 {
            current_rho /= trace;
        }
    }

    Array1::from(observables)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let l = 8;
    let _q_sector = 0;

    let dt = 0.001;
    let t_final = 10.0;
    
    let basis = translationally_invariant_basis(l);    
    
    let mut basis_per_sector: HashMap<i64, Vec<u64>> = HashMap::new();
    for k_sector in 0..l as i64 {
        for &state in &basis.rep_states {
            if let Some(translations) = basis.rep_index.get(&state) {
                let norm = normalization_factor(l, translations, k_sector);
                if norm > 0.0 {
                    basis_per_sector.entry(k_sector).or_insert_with(Vec::new).push(state);
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
                        let norm_a = normalization_factor(l, &states_a_vec, k_a);
                        let norm_b = normalization_factor(l, &states_b_vec, k_b);
                        
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
        let mut rho_vec = Array1::<Complex64>::zeros(basis_states.len());
        
        let neel_indices: Vec<usize> = basis_states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.states_a.contains(&neel_key) && s.states_b.contains(&neel_key))
            .map(|(i, _)| i)
            .collect();

        if !neel_indices.is_empty() {
            let initial_weight = 1.0 / neel_indices.len() as f64;
            for &idx in &neel_indices {
                rho_vec[idx] = Complex64::new(initial_weight, 0.0);
            }
        }

        let n_matrix = occupation_number(l, &basis_states, q_sector);
        let corr_matrix = density_correlation_nnn(l, &basis_states, q_sector);

        (basis_states, n_matrix, corr_matrix, rho_vec, neel_indices)
    };

    let (basis_0, n_mat_0, corr_mat_0, rho_0, neel_idx_0) = build_sector(0);
    let q_pi = (l / 2) as i64;
    let (basis_pi, n_mat_pi, corr_mat_pi, rho_pi, neel_idx_pi) = build_sector(q_pi);

    let gp_values = Array1::linspace(0.001, 0.2, 2);
    let gm_values = Array1::linspace(0.001, 0.2, 2);
    let omega_values = Array1::linspace(1.0, 2.0, 1);
    let alpha_values = Array1::linspace(0.0, 1.0, 3); 

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

    let results: Vec<String> = parameters.par_iter()
        .map(|&(gp, gm, omega, alpha)| {
            let gamma_minus = gm;
            let gamma_plus = gp;

            let l_cal_0 = build_lindbladian(l, &basis_0, 0, omega, gamma_plus, gamma_minus, alpha);
            let l_cal_dense_0 = {
                let mut dense = Array2::<Complex64>::zeros((basis_0.len(), basis_0.len()));
                for (val, (row, col)) in l_cal_0.iter() { dense[[row, col]] = *val; }
                dense
            };
            let obs_0 = obs_evolution(l, &n_mat_0, &corr_mat_0, &basis_0, 0, &rho_0, &l_cal_dense_0, &neel_idx_0, dt, t_final);

            let l_cal_pi = build_lindbladian(l, &basis_pi, q_pi, omega, gamma_plus, gamma_minus, alpha);
            let l_cal_dense_pi = {
                let mut dense = Array2::<Complex64>::zeros((basis_pi.len(), basis_pi.len()));
                for (val, (row, col)) in l_cal_pi.iter() { dense[[row, col]] = *val; }
                dense
            };
            let obs_pi = obs_evolution(l, &n_mat_pi, &corr_mat_pi, &basis_pi, q_pi, &rho_pi, &l_cal_dense_pi, &neel_idx_pi, dt, t_final);

            let total_obs: Vec<(f64, f64, f64)> = obs_0.iter().zip(obs_pi.iter())
                .map(|(o0, opi)| {
                    (o0.0 + opi.0, o0.1 + opi.1, o0.2 + opi.2)
                }).collect();

            let formatted_data = total_obs.iter()
                .map(|(n, nn, fid)| format!("{:.16}, {:.16}, {:.16}", n, nn, fid))
                .collect::<Vec<String>>()
                .join(", ");

            format!("{},{},{},{},{}", alpha, gp, gm, omega, formatted_data)
        })
        .collect();

    let mut file_occupation = File::create("occupation_time_alpha.csv")?;

    for line in results {
        writeln!(file_occupation, "{}", line)?;
    }

    Ok(())
}
