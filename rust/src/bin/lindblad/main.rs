use ndarray::Array2;
use ndarray_linalg::Eig;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

const THRESHOLD: f64 = 1e-10;

#[derive(Debug, Clone)]
struct Basis {
    rep_states: Vec<u64>,
    rep_index: HashMap<u64, Vec<u64>>,
}

fn fibonacci_basis(l: usize) -> Vec<u64> {
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
    let states = fibonacci_basis(l);
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

fn build_hamiltonian_parallel(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    omega: f64,
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut h_cal = Array2::<Complex64>::zeros((dim, dim));
    
    let rows: Vec<_> = (0..dim)
        .into_par_iter()
        .map(|j| {
            let mut row = vec![Complex64::new(0.0, 0.0); dim];
            
            for i in 0..dim {
                let state_in = &basis_states[i];
                let state_out = &basis_states[j];
                
                for site in 0..l {
                    let states_a_h_in: Vec<u64> = state_in.states_a
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    let states_b_h_in: Vec<u64> = state_in.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    
                    let mut a_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let cond1 = states_a_h_in[ii] == state_out.states_a[jj];
                            let cond2 = (states_a_h_in[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let cond3 = (states_a_h_in[ii] & (1u64 << ((site + 1) % l))) == 0;
                            
                            if cond1 && cond2 && cond3 {
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_prod += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    a_prod /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_prod = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let cond1 = states_b_h_in[ii] == state_out.states_b[jj];
                            let cond2 = (states_b_h_in[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let cond3 = (states_b_h_in[ii] & (1u64 << ((site + 1) % l))) == 0;
                            
                            if cond1 && cond2 && cond3 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * jj as i64 + k_diff_in * ii as i64) as f64) / (l as f64);
                                b_prod += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    b_prod /= state_in.norm_b * state_out.norm_b;
                    
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
                    
                    row[i] += Complex64::new(0.0, -omega) * (a_prod * inner_b - inner_a * b_prod);
                }
            }
            row
        })
        .collect();
    
    for (j, row) in rows.into_iter().enumerate() {
        for (i, val) in row.into_iter().enumerate() {
            h_cal[[j, i]] = val;
        }
    }
    
    h_cal
}

fn build_dissipation_parallel(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    gamma_plus: f64,
    gamma_minus: f64,
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut d_cal = Array2::<Complex64>::zeros((dim, dim));
    
    let rows: Vec<_> = (0..dim)
        .into_par_iter()
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
                    let states_a_l_in: Vec<u64> = state_in.states_a
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    let states_b_l_out: Vec<u64> = state_out.states_b
                        .iter()
                        .map(|&s| s ^ (1u64 << site))
                        .collect();
                    
                    // L_PLUS operators
                    let mut a_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_l_in[ii] == state_out.states_a[jj];
                            let c2 = (states_a_l_in[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_l_in[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_plus += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    a_plus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_plus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_l_out[ii] == state_in.states_b[jj];
                            let c2 = (states_b_l_out[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_b_l_out[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_b_l_out[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_plus += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    bb_plus /= state_in.norm_b * state_out.norm_b;
                    
                    row[i] += gamma_plus * (a_plus * b_plus - 0.5 * aa_plus * inner_b - 0.5 * inner_a * bb_plus);
                    
                    // L_MINUS operators
                    let mut a_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_a_l_in[ii] == state_out.states_a[jj];
                            let c2 = (states_a_l_in[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_l_in[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (states_a_l_in[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_minus += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    a_minus /= state_in.norm_a * state_out.norm_a;
                    
                    let mut b_minus = Complex64::new(0.0, 0.0);
                    for ii in 0..l {
                        for jj in 0..l {
                            let c1 = states_b_l_out[ii] == state_in.states_b[jj];
                            let c2 = (states_b_l_out[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_b_l_out[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_out.states_b[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_minus += Complex64::new(0.0, phase).exp();
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
                            let c4 = (states_a_l_in[ii] & (1 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_minus += Complex64::new(0.0, phase).exp();
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
                            let c4 = (states_b_l_out[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let k_diff_out = state_out.k - q_sector;
                                let k_diff_in = state_in.k - q_sector;
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_minus += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    bb_minus /= state_in.norm_b * state_out.norm_b;
                    
                    row[i] += gamma_minus * (a_minus * b_minus - 0.5 * aa_minus * inner_b - 0.5 * inner_a * bb_minus);
                }
            }
            row
        })
        .collect();
    
    for (j, row) in rows.into_iter().enumerate() {
        for (i, val) in row.into_iter().enumerate() {
            d_cal[[j, i]] = val;
        }
    }
    
    d_cal
}

fn build_lindbladian(
    l: usize,
    basis_states: &[BasisState],
    q_sector: i64,
    omega: f64,
    gamma_plus: f64,
    gamma_minus: f64,
) -> Array2<Complex64> {
    let dim = basis_states.len();
    
    // 1. Allocate the single final matrix
    let mut l_cal = Array2::<Complex64>::zeros((dim, dim));
    
    // 2. Parallelize directly over the mutable memory slice of the matrix
    //    Each chunk corresponds to one row 'j'
    l_cal.as_slice_mut()
        .unwrap()
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(j, row_slice)| {
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
                            
                            if c1 && c2 && c3 {
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
                            let c2 = (states_a_in_shifted[ii] & (1u64 << ((site + l - 1) % l))) == 0;
                            let c3 = (states_a_in_shifted[ii] & (1u64 << ((site + 1) % l))) == 0;
                            let c4 = (state_in.states_a[ii] & (1u64 << (site % l))) == 0;
                            if c1 && c2 && c3 && c4 {
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_plus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                a_minus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                b_minus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((state_out.k * jj as i64 - state_in.k * ii as i64) as f64) / (l as f64);
                                aa_minus += Complex64::new(0.0, phase).exp();
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
                                let phase = 2.0 * PI * ((-k_diff_out * ii as i64 + k_diff_in * jj as i64) as f64) / (l as f64);
                                bb_minus += Complex64::new(0.0, phase).exp();
                            }
                        }
                    }
                    bb_minus /= state_in.norm_b * state_out.norm_b;
                    
                    elem += gamma_minus * (a_minus * b_minus - 0.5 * aa_minus * inner_b - 0.5 * inner_a * bb_minus);
                }
                
                row_slice[i] = elem;
            }
        });
    
    l_cal
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let l = 10;
    let q_sector = 0;
    let omega = 2.0;
    let gamma_plus = 1.0;
    let gamma_minus = 0.5;
    
    println!("=== Lindbladian Solver ===");
    println!("L = {}, Q_sector = {}", l, q_sector);
    println!("Omega = {}, γ+ = {}, γ- = {}", omega, gamma_plus, gamma_minus);
    println!("\nGenerating basis...");
    
    let basis = translationally_invariant_basis(l);
    println!("Representative states: {}", basis.rep_states.len());
    
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
    
    let mut basis_states = Vec::new();
    for k in 0..l as i64 {
        if let Some(states_in_sector) = basis_per_sector.get(&k) {
            for &state_a in states_in_sector {
                for &state_b in states_in_sector {
                    let states_a = basis.rep_index.get(&state_a).unwrap().clone();
                    let states_b = basis.rep_index.get(&state_b).unwrap().clone();
                    let norm_a = normalization_factor(l, &states_a, k);
                    let norm_b = normalization_factor(l, &states_b, k);
                    
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
    
    let dimension = basis_states.len();
    println!("Total dimension: {}\n", dimension);
    
    // println!("Building Hamiltonian matrix...");
    // let h_cal = build_hamiltonian_parallel(l, &basis_states, q_sector, omega);
    // println!("✓ Hamiltonian complete!");
    
    // println!("\nBuilding Dissipation matrix...");
    // let d_cal = build_dissipation_parallel(l, &basis_states, q_sector, gamma_plus, gamma_minus);
    // println!("✓ Dissipation complete!");
    
    println!("\nBuilding Lindbladian...");
    let l_cal = build_lindbladian(l, &basis_states, q_sector, omega, gamma_plus, gamma_minus);
    // let l_cal = h_cal + d_cal;
    println!("✓ Lindbladian complete!");

    
    // println!("\nComputing eigenvalues...");
    // match l_cal.eig() {
    //     Ok((eigenvalues, _eigenvectors)) => {
    //         println!("✓ Eigenvalues computed ({} values)\n", eigenvalues.len());
            
    //         // Write to CSV
    //         let mut file = File::create("eigenvalues.csv")?;
    //         writeln!(file, "real,imaginary")?;
    //         for eval in eigenvalues.iter() {
    //             writeln!(file, "{},{}", eval.re, eval.im)?;
    //         }
    //         println!("✓ Eigenvalues saved to eigenvalues.csv");
            
    //         // Print first 10 eigenvalues
    //         println!("\nFirst 10 eigenvalues:");
    //         for (i, eval) in eigenvalues.iter().take(10).enumerate() {
    //             println!("  λ_{:<2} = {:>12.6} + {:>12.6}i", i, eval.re, eval.im);
    //         }
    //     }
    //     Err(e) => {
    //         eprintln!("Error: Eigenvalue computation failed: {:?}", e);
    //     }
    // }
    
    println!("\n=== Done! ===");
    Ok(())
}
