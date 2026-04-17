use ndarray::{Array1,Array2,s};
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

fn lucas_basis(l: usize) -> Vec<u64> {
    let mut states = Vec::new();
    for i in 0..(1u64 << l) {
        if i & (i >> 1) == 0 {
            let first_bit = (1u64 << 0) & i != 0;
            let last_bit = (1u64 << (l - 1)) & i != 0;
            if !(first_bit && last_bit) {
                states.push(i);
            }
        }
    }
    states
}

fn translationally_invariant_basis(l: usize) -> Basis {
    let states = lucas_basis(l);
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
    phases: &[Complex64],
) -> CsMat<Complex64> {
    let dim = basis_states.len();
    
    // 1. Allocate the single final matrix
    // let mut l_cal = Array2::<Complex64>::zeros((dim, dim));
    
    // // 2. Parallelize directly over the mutable memory slice of the matrix
    // //    Each chunk corresponds to one row 'j'
    // l_cal.as_slice_mut()
    //     .unwrap()
    //     .par_chunks_mut(dim)
    //     .enumerate()
    //     .for_each(|(j, row_slice)| {
    let triplets: Vec<(usize, usize, Complex64)> = (0..dim)
        .into_iter()
        .flat_map(|j| {
            let mut local_triplets = Vec::new();

            let state_out = &basis_states[j];
            
            for i in 0..dim {
                
                let state_in = &basis_states[i];
                let mut elem = Complex64::new(0.0, 0.0);
                
                // Pre-calculate Inner Products (Common to both H and D parts)
                let inner_b = inner_product(
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
                    phases,
                );
                
                let inner_a = inner_product(
                    &state_in.states_a,
                    &state_out.states_a,
                    state_in.k,
                    state_out.k,
                    l,
                    state_in.norm_a,
                    state_out.norm_a,
                    phases,
                );

                for site in 0..l {
                    // Pre-calculate shifted states (Used by both H and D logic)
                    // Note: 'h_in' logic from your H-builder and 'l_in' from D-builder are identical (XOR)
                    let states_a_shifted: Vec<u64> = state_in.states_a
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    let states_b_shifted: Vec<u64> = state_in.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site)) // Note: Check if D used state_out or state_in here. 
                                                      // D uses state_out.states_b for B-operators. H uses state_in.
                        .collect();                   // We will compute specific shifts inside if they differ.

                    // To match your original logic strictly:
                    // H uses: state_in ^ site for both A and B.
                    // D uses: state_in ^ site for A (L_in), state_out ^ site for B (L_out).
                    
                    let states_a_in_shifted = states_a_shifted; // construct created above
                    let states_b_in_shifted = states_b_shifted; // construct created above
                    let states_b_out_shifted: Vec<u64> = state_out.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();

                    // ===========================
                    // HAMILTONIAN CONTRIBUTION
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
                                // let phase = 2.0 * PI * ((-k_diff_out * jj as i64 + k_diff_in * ii as i64) as f64) / (l as f64);
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
                            let c2 = (states_a_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_plus += phases[idx as usize];
                            }
                        }
                    }
                    a_plus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_out_shifted[ii] == state_in.states_b[jj];
                            let c2 = (states_b_out_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_b_out_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_b_out_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                b_plus += phases[idx as usize];
                            }
                        }
                    }
                    b_plus /= state_in.norm_b * state_out.norm_b;
                    
                    let mut aa_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let c2 = (state_in.states_a[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (state_in.states_a[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                aa_plus += phases[idx as usize];
                            }
                        }
                    }
                    aa_plus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut bb_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_out.states_b[ii] == state_in.states_b[jj];
                            let c2 = (state_out.states_b[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (state_out.states_b[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_out.states_b[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                bb_plus += phases[idx as usize];
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
                            let c2 = (states_a_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_a_in_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_minus += phases[idx as usize];
                            }
                        }
                    }
                    a_minus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_out_shifted[ii] == state_in.states_b[jj];
                            let c2 = (states_b_out_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_b_out_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_out.states_b[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                b_minus += phases[idx as usize];
                            }
                        }
                    }
                    b_minus /= state_in.norm_b * state_out.norm_b;
                    
                    let mut aa_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            let c2 = (state_in.states_a[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (state_in.states_a[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_a_in_shifted[ii] & (1 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                aa_minus += phases[idx as usize];
                            }
                        }
                    }
                    aa_minus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut bb_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = state_out.states_b[ii] == state_in.states_b[jj];
                            let c2 = (state_out.states_b[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (state_out.states_b[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_b_out_shifted[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let idx = (-k_diff_out * ii as i64 + k_diff_in * jj as i64).rem_euclid(l as i64);
                                bb_minus += phases[idx as usize];
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
                // row_slice[i] = elem;
    let mut tri_mat = TriMat::new((dim, dim));
    for (row, col, val) in triplets {
        tri_mat.add_triplet(row, col, val);
    }
    
    tri_mat.to_csr()
    //         }
    //     });
    
    // l_cal
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
                
                // 1. Compute the overlap of the B-part (Bra)
                // This is identical to 'inner_b' in your original code
                let inner_b = inner_product(
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
                    phases,
                );
                
                // Optimization: If B-states are orthogonal, the whole term is 0.
                if inner_b.norm() <= THRESHOLD {
                    continue;
                }

                // 2. Compute the A-part (Ket): Sum over all sites
                for site in 0..l {
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    
                    // Loop over translations (same logic as your Hamiltonian)
                    for ii in 0..l {
                        for jj in 0..l {
                            // Condition 1: States must match (Diagonal in real space)
                            // We do NOT flip bits (unlike Hamiltonian/Dissipation)
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            
                            // Condition 2: Is the particle there? 
                            // (We removed the P constraints c2/c3, kept only occupancy check)
                            let c_occ = (state_in.states_a[ii] & (1u64 << (site % l))) != 0;
                            
                            if c1 && c_occ {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_prod += phases[idx as usize];
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    
                    // Add contribution for this site
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
    
    n_cal/l as f64
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
                
                // 1. Compute the overlap of the B-part (Bra)
                let inner_b = inner_product(
                    &state_out.states_b,
                    &state_in.states_b,
                    state_out.k - q_sector,
                    state_in.k - q_sector,
                    l,
                    state_out.norm_b,
                    state_in.norm_b,
                    phases,
                );
                
                if inner_b.norm() <= THRESHOLD {
                    continue;
                }

                // 2. Compute the A-part (Ket): Sum over all sites
                // This corresponds to the sum over j in <n_{j-1} n_{j+1}>
                for site in 0..l {
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    
                    // Identify neighbors with Periodic Boundary Conditions
                    // (site - 1) wrapping around
                    let idx_minus = if site == 0 { l - 1 } else { site - 1 };
                    // (site + 1) wrapping around
                    let idx_plus = if site == l - 1 { 0 } else { site + 1 };

                    // Loop over translations
                    for ii in 0..l {
                        for jj in 0..l {
                            // Condition 1: States must match (Diagonal in real space)
                            let c1 = state_in.states_a[ii] == state_out.states_a[jj];
                            
                            // Condition 2: Check correlations
                            // We need occupancy at BOTH (site-1) AND (site+1)
                            // We create a mask for these two positions
                            let mask = (1u64 << idx_minus) | (1u64 << idx_plus);
                            
                            // Check if the current configuration has bits set at both positions
                            let c_corr = (state_in.states_a[ii] & mask) == mask;
                            
                            if c1 && c_corr {
                                let idx = (state_out.k * jj as i64 - state_in.k * ii as i64).rem_euclid(l as i64);
                                a_prod += phases[idx as usize];
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    
                    // Add contribution for this specific site pair
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
    
    // Normalize by L to get the average correlation per site
    corr_cal / l as f64
}

fn compute_trace(
    l: usize,
    basis_states: &[BasisState],
    vectorized_op: &Array1<Complex64>, // This is your aux_vec
    q_sector: i64,
    phases: &[Complex64],
) -> Complex64 {
    // If Q != 0, trace is 0
    if q_sector != 0 {
        return Complex64::new(0.0, 0.0);
    }

    // Sum_j (coeff_j * <B_j | A_j>)
    vectorized_op.as_slice().unwrap()
        .par_iter()
        .enumerate()
        .map(|(i, &coeff)| {
            if coeff.norm() <= THRESHOLD {
                return Complex64::new(0.0, 0.0);
            }

            let state = &basis_states[i];
            
            // Calculate overlap <B|A>
            let overlap = inner_product(
                &state.states_a,
                &state.states_b,
                state.k,
                state.k, 
                l,
                state.norm_a,
                state.norm_b,
                phases,
            );

            coeff * overlap
        })
        .sum()
}


/// Computes <O> = Sum(|c_j|^2 * o_j) / Sum(|c_j|^2) for a given eigenvector
fn compute_eigenstate_observable(
    eigenvector: &Array1<Complex64>, 
    basis_observables: &[f64], 
) -> f64 {
    let mut obs_sum = 0.0;
    let mut norm_sum = 0.0;
    
    // Iterate over the coefficients
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
                
                // Pre-compute Occupation Number
                let basis_occupations: Vec<f64> = (0..dim).map(|n| {
                    let diag_idx = current_offset + (n * dim + n);
                    let state = &basis_states[diag_idx];
                    let particle_count = state.states_a[0].count_ones() as f64;
                    particle_count / (l as f64)
                }).collect();

                // Compute for each Eigenvector
                for (idx, &lambda) in eigvals.iter().enumerate() {
                    // CHANGE: .to_owned() creates a standard Array1 from the column view
                    let eigenvector = eigvecs.column(idx).to_owned();
                    
                    // Now we pass &Array1, which is compatible with the new signature
                    let occ = compute_eigenstate_observable(&eigenvector, &basis_occupations);
                    
                    results.push((lambda, occ));
                }
            }
        }

        current_offset += block_len;
        i += block_len;
    }

    // Normalize Spectrum
    let trace: f64 = results.iter().map(|(lam, _)| lam).sum();
    if trace.abs() > 1e-15 {
        for (lam, _) in results.iter_mut() {
            *lam /= trace;
        }
    }

    // Sort Descending
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    results
}



#[derive(Debug, Clone)]
pub struct SpectralData {
    pub real_eigenvalue: f64,
    pub imag_eigenvalue: f64,
    pub overlap: f64,
    pub occupation: f64,
    pub block_size: usize, 
}

pub fn analyze_lindbladian(
    eigenvalues: &Array1<Complex64>,
    eigenvectors: &Array2<Complex64>,
    occ_op: &Array2<Complex64>,
    rho: &Array1<Complex64>,
    tol: f64,
) -> Result<Vec<SpectralData>, Box<dyn Error>> {
    
    // 1. Standard Eigen Decomposition
    // Note: For defective matrices, LAPACK returns "parallel" eigenvectors
    // for the Jordan chain. We detect this using SVD below.
    let evals = eigenvalues;
    let evecs = eigenvectors;
    
    // 2. Pair eigenvalues with indices and Sort by Real part (Decay Rate)
    let mut tagged_evals: Vec<(usize, Complex64)> = evals
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();

    // Sort descending by real part (closest to 0 first)
    tagged_evals.sort_by(|a, b| {
        b.1.re.partial_cmp(&a.1.re).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut results = Vec::new();
    let n = evecs.nrows();
    let mut i = 0;

    // 3. Loop through eigenvalues and Cluster them
    while i < tagged_evals.len() {
        let current_val = tagged_evals[i].1;
        let mut cluster_indices = vec![tagged_evals[i].0];
        
        // Find all subsequent eigenvalues that are within 'tol' distance
        let mut j = i + 1;
        while j < tagged_evals.len() {
            if (tagged_evals[j].1 - current_val).norm() < tol {
                cluster_indices.push(tagged_evals[j].0);
                j += 1;
            } else {
                break;
            }
        }
        
        // m_a: Algebraic Multiplicity (How many eigenvalues are identical)
        let m_a = cluster_indices.len();

        // 4. Extract Eigenvectors for this cluster
        // We create a matrix of shape (N x m_a)
        let mut raw_subspace = Array2::<Complex64>::zeros((n, m_a));
        for (col_idx, &eig_idx) in cluster_indices.iter().enumerate() {
            let vec = evecs.column(eig_idx);
            raw_subspace.column_mut(col_idx).assign(&vec);
        }

        // 5. The Jordan Test: Compute Rank via SVD
        let (_, sigma, _) = raw_subspace.svd(false, false)?;
        let rank_tol = 1e-5; 
        let m_g = sigma.iter().filter(|&&s| s > rank_tol).count();

        // [RESTORED] Compute the orthonormal basis `u`
        // This is required for both branches below to calculate overlaps.
        let (u_opt, _, _) = raw_subspace.svd(true, false)?;
        let u = u_opt.ok_or("SVD U calculation failed")?;

        // 6-8: Branch based on diagonalizability
        if m_g == m_a {
            // Fully diagonalizable (Jordan blocks of size 1)
            for k in 0..m_g {
                let u_k = u.column(k);
                let overlap = (&u_k).mapv(|x| x.conj()).dot(rho).norm();

                let occupation = (&u_k).mapv(|x| x.conj()).dot(occ_op).dot(&u_k).norm();                
                
                results.push(SpectralData {
                    real_eigenvalue: current_val.re,
                    imag_eigenvalue: current_val.im,
                    overlap,
                    occupation,
                    block_size: 1,
                });
            }
        } else {
            // Defective subspace (Jordan blocks > 1)
            // [FIXED] Explicitly designate 0.0 as an f64 to resolve the ambiguous float error.
            let mut overlap_sq = 0.0f64;
            let mut occupation_sq = 0.0f64;
            for k in 0..m_g {
                let u_k = u.column(k);
                overlap_sq += (&u_k).mapv(|x| x.conj()).dot(rho).norm_sqr();

                occupation_sq += (&u_k).mapv(|x| x.conj()).dot(occ_op).dot(&u_k).norm_sqr();
            }
            
            results.push(SpectralData {
                real_eigenvalue: current_val.re,
                imag_eigenvalue: current_val.im,
                overlap: overlap_sq.sqrt(),
                occupation: occupation_sq.sqrt(),
                block_size: m_a, // Note: This still assumes a single block of size m_a
            });
        }

        // Advance the outer loop index past this cluster
        i = j;    }

    Ok(results)
}


fn precompute_phases(l: usize) -> Vec<Complex64> {
    (0..l).map(|k| {
        let phase = 2.0 * PI * (k as f64) / (l as f64);
        Complex64::new(0.0, phase).exp()
    }).collect()
}

struct SimulationResult {
    // We store the ready-to-write strings here
    occupation_str: String,
    std_eigenvalues_str: String,
    eigenvalues_str: String,
    decay_str: String,
    oee_str: String,
}




// ==========================================
// ENTANGLEMENT ENTROPY FUNCTIONS
// ==========================================

/// Expands a Q=0 eigenmatrix into the doubled real-space basis.
/// CRITICAL: Implements the exp(i*k*(j + j')) phase required for OEE vectorization.
fn expand_eigenmatrix(
    l: usize,
    basis_states: &[BasisState],
    eigenvector: &[Complex64],
    phases: &[Complex64],
) -> Vec<(u64, u64, Complex64)> {
    // Use a HashMap to combine coefficients for identical real-space states.
    // This prevents catastrophic O(N^2) performance degradation in the trace computation
    // when states have orbits smaller than L (e.g., Néel states).
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

                // The ket and the bra are both treated as kets in the doubled space.
                // Hence, the forward phase is applied to both: k*(j + j_prime)
                let phase_idx = (state.k * j as i64 + state.k * j_prime as i64).rem_euclid(l as i64);
                let phase = phases[phase_idx as usize];

                let coeff = c * phase / norm_factor;
                
                // Combine identical terms immediately
                *expanded_map.entry((real_a, real_b)).or_insert(Complex64::new(0.0, 0.0)) += coeff;
            }
        }
    }
    
    // Convert back to a flat vector for the entropy function
    expanded_map.into_iter().map(|((a, b), c)| (a, b, c)).collect()
}

/// Generates the Fibonacci cube (open boundary conditions) for a given length.
pub fn fibonacci_basis(l: usize) -> Vec<u64> {
    (0..(1u64 << l))
        .filter(|&state| state & (state >> 1) == 0)
        .collect()
}

/// Splits a global state into Subsystem A (lower half) and Subsystem B (upper half).
pub fn split_state(state: u64, l: usize) -> (u64, u64) {
    let l_a = l / 2;
    let mask_a = (1u64 << l_a) - 1;
    
    let a = state & mask_a;
    let b = state >> l_a;
    
    (a, b)
}

/// Computes the Operator Entanglement Entropy of an expanded eigenmatrix.
/// `expanded_state` is a vector of (Global Ket, Global Bra, Coefficient).
pub fn operator_entanglement_entropy(
    expanded_state: &[(u64, u64, Complex64)],
    l: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let l_a = l / 2;
    let fib_a = fibonacci_basis(l_a);
    
    // Map Fibonacci states to integer indices for matrix construction
    let mut a_idx_map = HashMap::new();
    for (i, &state) in fib_a.iter().enumerate() {
        a_idx_map.insert(state, i);
    }
    
    let dim_a = fib_a.len();
    let super_dim = dim_a * dim_a;
    let mut sigma_a = Array2::<Complex64>::zeros((super_dim, super_dim));
    
    // 1. Group terms by their Subsystem B configuration (b_ket, b_bra)
    let mut b_groups: HashMap<(u64, u64), Vec<((u64, u64), Complex64)>> = HashMap::new();
    
    for &(global_ket, global_bra, coeff) in expanded_state {
        let (a_ket, b_ket) = split_state(global_ket, l);
        let (a_bra, b_bra) = split_state(global_bra, l);
        
        b_groups.entry((b_ket, b_bra))
            .or_default()
            .push(((a_ket, a_bra), coeff));
    }
    
    // 2. Perform the partial trace over the doubled B space
    for (_, terms) in b_groups {
        for &((a_ket1, a_bra1), c1) in &terms {
            for &((a_ket2, a_bra2), c2) in &terms {
                let row_idx = a_idx_map[&a_ket1] * dim_a + a_idx_map[&a_bra1];
                let col_idx = a_idx_map[&a_ket2] * dim_a + a_idx_map[&a_bra2];
                
                sigma_a[[row_idx, col_idx]] += c1 * c2.conj();
            }
        }
    }
    
    // 3. Normalize the reduced matrix
    let mut trace = Complex64::new(0.0, 0.0);
    for i in 0..super_dim {
        trace += sigma_a[[i, i]];
    }
    
    if trace.norm() < 1e-12 {
        return Ok(0.0); 
    }
    sigma_a /= trace;
    
    // 4. Diagonalize and compute von Neumann entropy
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
    let l = 12;
    let q_sector = 0;
    
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
    
    let mut basis_states = Vec::new();
    for k in 0..l as i64 {
        if let Some(states_in_sector) = basis_per_sector.get(&k) {
            for &state_a in states_in_sector {
                for &state_b in states_in_sector {
                    let states_a = basis.rep_index.get(&state_a).unwrap().clone();
                    let states_b = basis.rep_index.get(&state_b).unwrap().clone();
                    let norm_a = normalization_factor(l, &states_a, k, &phases);
                    let norm_b = normalization_factor(l, &states_b, k, &phases);
                    
                    basis_states.push(BasisState {
                        states_a,
                        states_b,
                        k,
                        norm_a,
                        norm_b,
                    });
                }
            }
        }
    }

    println!("Total basis states in Q=0 sector: {}", basis_states.len());
    
    let gp_values = Array1::linspace(0.001, 0.2, 2);
    let gm_values = Array1::linspace(0.001, 0.2, 2);
    let omega_values = Array1::linspace(1.0, 2.0, 1);

    // let raw_space = Array1::linspace(0.5, 1.0, 3);
    // let lower_segment = raw_space.slice(s![..-1]);
    // let mut result = lower_segment.to_vec(); // Convert to Vec
    // result.push(1.0);                        // Add center
    // for &g in lower_segment.iter().rev() {
    //     result.push(1.0 / g);
    // }
    // let g_values = Array1::from(result);

    let mut parameters = Vec::new();
    for &gp in &gp_values {
        for &gm in &gm_values {
            for &omega in &omega_values {
                parameters.push((gp, gm, omega));
            }
        }
    }


    // --- Initial State (Neel x Neel) ---
    let neel_key: u64 = (4u64.pow(l as u32 / 2) - 1) / 3;
    let mut rho_vec_neel = Array1::<Complex64>::zeros(basis_states.len());

    if let Some(idx) = basis_states.iter().position(|s| 
        s.states_a.contains(&neel_key) && s.states_b.contains(&neel_key)
    ) {
        rho_vec_neel[idx] = Complex64::new(1.0, 0.0);
        println!("Vectorized |Neel>x|Neel> placed at index: {}", idx);
    } else {
        println!("Néel state not found. Check if the k-sector allows it.");
    }

    let n_matrix = occupation_number(l, &basis_states, q_sector, &phases);
    let corr_matrix = density_correlation_nnn(l, &basis_states, q_sector, &phases);


    println!("Starting parallel sweep over {} points...", parameters.len());

    // 1. RUN PARALLEL SIMULATIONS
    let results: Vec<SimulationResult> = parameters
        .par_iter()
        .map(|&(gp, gm, omega)| {
            // A. Setup Buffers for Output
            let mut res = SimulationResult {
                occupation_str: String::new(),
                std_eigenvalues_str: String::new(),
                eigenvalues_str: String::new(),
                decay_str: String::new(),
                oee_str: String::new(),
            };

            // B. Build Matrix (Directly Dense)
            // Allocates ~2.2GB for L=12

            let gamma_minus = gm;
            let gamma_plus = gp;
            // let mut l_mat = Array2::<Complex64>::zeros((basis_states.len(), basis_states.len()));
            
            // // Call the SERIAL builder
            // build_lindbladian_serial(l, &basis_states, &mut l_mat, q_sector, omega, gamma_plus, gamma_minus, &phases);

            let l_cal = build_lindbladian(l, &basis_states, q_sector, omega, gamma_plus, gamma_minus, &phases);

            let l_cal_dense = {
                let mut dense = Array2::<Complex64>::zeros((basis_states.len(), basis_states.len()));
                for (val, (row, col)) in l_cal.iter() {
                    dense[[row, col]] = *val;
                }
                dense
            };

            // C. Diagonalize
            if let Ok((evals, evecs)) = l_cal_dense.eig() {
                
                res.oee_str.push_str(&format!("{},{},{}", gp, gm, omega));

                for (&lambda, vec_view) in evals.iter().zip(evecs.columns()) {
                    // Safely extract the contiguous vector
                    let vec_contiguous = vec_view.to_vec();
                    
                    // Expand and compute
                    let expanded = expand_eigenmatrix(l, &basis_states, &vec_contiguous, &phases);
                    let entropy = operator_entanglement_entropy(&expanded, l).unwrap_or(0.0);
                    
                    // Append the triplet (Re[eval], Im[eval], OEE) continuously to the same line
                    res.oee_str.push_str(&format!(",{:.10},{:.10},{:.6}", lambda.re, lambda.im, entropy));
                }
                // Terminate the row for this parameter combination
                res.oee_str.push('\n');

                // --- 1. Decay Analysis ---
                if let Ok(analysis) = analyze_lindbladian(&evals, &evecs, &n_matrix, &rho_vec_neel, 1e-6) {
                    res.decay_str.push_str(&format!("{},{},{}", gp, gm, omega)); 
                    for data in analysis {
                        res.decay_str.push_str(&format!(",{:.10},{:.10},{:.10},{},{}", 
                            data.real_eigenvalue, data.imag_eigenvalue, data.overlap, data.occupation, data.block_size));
                    }
                    res.decay_str.push('\n');
                }                

                // --- 2. Eigenvalues Dump ---
                res.eigenvalues_str.push_str(&format!("{},{},{}", gp, gm, omega));
                for (_i, eval) in evals.iter().enumerate() {
                    res.eigenvalues_str.push_str(&format!(",{},{}", eval.re, eval.im));
                }
                res.eigenvalues_str.push('\n');

                // --- 3. Steady State Analysis ---
                for (i, eval) in evals.iter().enumerate() {
                    if eval.re.abs() < 1e-8 && eval.im.abs() < 1e-8 {
                        let rho_vec = evecs.column(i).to_owned();

                        // Std Eigenvalues
                        let spectrum_data = steady_state_properties(l, &basis_states, &rho_vec);
                        let formatted_data = spectrum_data.iter()
                            .map(|(p, n)| format!("{:.16}, {:.6}", p, n))
                            .collect::<Vec<String>>()
                            .join(", ");
                        res.std_eigenvalues_str.push_str(&format!("{},{},{},{}\n", gp, gm, omega, formatted_data));

                        // Occupation
                        let tr_rho = compute_trace(l, &basis_states, &rho_vec, q_sector, &phases);
                        let n_rho = n_matrix.dot(&rho_vec);
                        let nn_rho = corr_matrix.dot(&rho_vec); // Using corr_matrix from outer scope is fine (read-only)
                        
                        let tr_n = compute_trace(l, &basis_states, &n_rho, q_sector, &phases);
                        let tr_nn = compute_trace(l, &basis_states, &nn_rho, q_sector, &phases);

                        if tr_rho.norm() > 1e-10 {
                            let exp_n = (tr_n / tr_rho).re;
                            let exp_nn = (tr_nn / tr_rho).re;
                            res.occupation_str.push_str(&format!("{},{},{},{},{}\n", gp, gm, omega, exp_n, exp_nn));
                        }
                    }
                }
            }
            // l_mat is dropped here, freeing RAM
            res
        })
        .collect();

    // 2. WRITE TO FILES (After loop, safe and fast)
    let mut file_occupation = File::create("occupation.csv")?;
    writeln!(file_occupation, "gp,gm,omega,n,nn")?;
    
    let mut file_std = File::create("std_eigenvalues.csv")?;
    let mut file_evals = File::create("eigenvalues.csv")?;
    let mut file_decay = File::create("decay.csv")?;
    let mut file_oee = File::create("oee.csv")?;

    for res in results {
        file_occupation.write_all(res.occupation_str.as_bytes())?;
        file_std.write_all(res.std_eigenvalues_str.as_bytes())?;
        file_evals.write_all(res.eigenvalues_str.as_bytes())?;
        file_oee.write_all(res.oee_str.as_bytes())?;
        file_decay.write_all(res.decay_str.as_bytes())?;
    }

    Ok(())
}
