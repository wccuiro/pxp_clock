use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

const THRESHOLD: f64 = 1e-10;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct BasisState {
    a: u64,
    b: u64,
}

// Generates the real-space Lucas states (no adjacent excitations, PBC checked)
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

// Builds the Lindbladian in the full Doubled space with Asymmetric Dissipation
fn build_lindbladian(
    l: usize,
    basis_states: &[BasisState],
    omega: f64,
    gamma_plus: f64,
    gamma_minus: f64,
) -> CsMat<Complex64> {
    let dim = basis_states.len();
    
    let mut state_to_idx = HashMap::new();
    for (i, state) in basis_states.iter().enumerate() {
        state_to_idx.insert((state.a, state.b), i);
    }

    let triplets: Vec<(usize, usize, Complex64)> = (0..dim)
        .into_par_iter()
        .flat_map(|i| {
            let mut local_triplets = Vec::new();
            let state_in = &basis_states[i];
            let a = state_in.a;
            let b = state_in.b;
            let mut diag_val = Complex64::new(0.0, 0.0);

            for site in 0..l {
                let p_prev = 1u64 << ((site + l - 1) % l);
                let p_next = 1u64 << ((site + 1) % l);
                let mask = 1u64 << site;

                let a_can_flip = (a & p_prev) == 0 && (a & p_next) == 0;
                let b_can_flip = (b & p_prev) == 0 && (b & p_next) == 0;

                // ===========================
                // HAMILTONIAN CONTRIBUTION (-i [H, rho])
                // ===========================
                if a_can_flip {
                    let a_out = a ^ mask;
                    if let Some(&j) = state_to_idx.get(&(a_out, b)) {
                        local_triplets.push((j, i, Complex64::new(0.0, -omega)));
                    }
                }
                if b_can_flip {
                    let b_out = b ^ mask;
                    if let Some(&j) = state_to_idx.get(&(a, b_out)) {
                        local_triplets.push((j, i, Complex64::new(0.0, omega)));
                    }
                }

                // ===========================
                // DISSIPATION: ASSYMETRIC
                // ===========================
                if site % 2 == 0 {
                    // EVEN SITES: ONLY EMISSION (gamma_minus)
                    if a_can_flip && b_can_flip && (a & mask) != 0 && (b & mask) != 0 {
                        let a_out = a ^ mask;
                        let b_out = b ^ mask;
                        if let Some(&j) = state_to_idx.get(&(a_out, b_out)) {
                            local_triplets.push((j, i, Complex64::new(gamma_minus, 0.0)));
                        }
                    }
                    if a_can_flip && (a & mask) != 0 {
                        diag_val -= Complex64::new(0.5 * gamma_minus, 0.0);
                    }
                    if b_can_flip && (b & mask) != 0 {
                        diag_val -= Complex64::new(0.5 * gamma_minus, 0.0);
                    }
                } else {
                    // ODD SITES: ONLY ABSORPTION (gamma_plus)
                    if a_can_flip && b_can_flip && (a & mask) == 0 && (b & mask) == 0 {
                        let a_out = a ^ mask;
                        let b_out = b ^ mask;
                        if let Some(&j) = state_to_idx.get(&(a_out, b_out)) {
                            local_triplets.push((j, i, Complex64::new(gamma_plus, 0.0)));
                        }
                    }
                    if a_can_flip && (a & mask) == 0 {
                        diag_val -= Complex64::new(0.5 * gamma_plus, 0.0);
                    }
                    if b_can_flip && (b & mask) == 0 {
                        diag_val -= Complex64::new(0.5 * gamma_plus, 0.0);
                    }
                }
            }
            
            if diag_val.norm() > THRESHOLD {
                local_triplets.push((i, i, diag_val));
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

// Optimized observables: We only care about diagonal operators in real space
// So we extract the diagonal values instead of doing full Matrix dot products.
fn occupation_number_diag(l: usize, basis_states: &[BasisState]) -> Array1<f64> {
    let mut diag = Array1::<f64>::zeros(basis_states.len());
    for (i, state) in basis_states.iter().enumerate() {
        if state.a == state.b { // Traces only matter where a == b
            let count = state.a.count_ones() as f64;
            diag[i] = count / l as f64;
        }
    }
    diag
}

fn density_correlation_nnn_diag(l: usize, basis_states: &[BasisState]) -> Array1<f64> {
    let mut diag = Array1::<f64>::zeros(basis_states.len());
    for (i, state) in basis_states.iter().enumerate() {
        if state.a == state.b {
            let mut count = 0;
            for site in 0..l {
                let idx_minus = (site + l - 1) % l;
                let idx_plus = (site + 1) % l;
                if (state.a & (1 << idx_minus)) != 0 && (state.a & (1 << idx_plus)) != 0 {
                    count += 1;
                }
            }
            diag[i] = count as f64 / l as f64;
        }
    }
    diag
}

fn compute_trace(basis_states: &[BasisState], rho: &Array1<Complex64>) -> Complex64 {
    let mut tr = Complex64::new(0.0, 0.0);
    for (i, state) in basis_states.iter().enumerate() {
        if state.a == state.b {
            tr += rho[i];
        }
    }
    tr
}

fn compute_observable_expectation(
    basis_states: &[BasisState], 
    rho: &Array1<Complex64>, 
    obs_diag: &Array1<f64>
) -> Complex64 {
    let mut val = Complex64::new(0.0, 0.0);
    for (i, state) in basis_states.iter().enumerate() {
        if state.a == state.b {
            val += rho[i] * obs_diag[i];
        }
    }
    val
}

fn obs_evolution(
    n_diag: &Array1<f64>,
    corr_diag: &Array1<f64>,
    basis_states: &[BasisState],
    initial: &Array1<Complex64>,
    lindbladian: &Array2<Complex64>,
    neel_idx: usize, 
    dt: f64,
    t_final: f64,
) -> Vec<(f64, f64, f64)> {
    let num_steps = (t_final / dt).round() as usize;
    let mut observables = Vec::with_capacity(num_steps + 1);
    let mut current_rho = initial.clone();

    for _step in 0..=num_steps {
        // Fast O(N) evaluation of physical observables without dense dot products
        let expectation_val = compute_observable_expectation(basis_states, &current_rho, n_diag);
        let corr_val = compute_observable_expectation(basis_states, &current_rho, corr_diag);
        
        // FIDELITY: the overlap with the purely initial single Neel state
        let fidelity_val = current_rho[neel_idx].re;

        observables.push((expectation_val.re, corr_val.re, fidelity_val));

        // Advance time (Euler)
        let derivative = lindbladian.dot(&current_rho);
        current_rho = current_rho + derivative * Complex64::new(dt, 0.0);
        
        // Normalize
        let trace = compute_trace(basis_states, &current_rho);
        if trace.norm() > 1e-15 {
            current_rho /= trace;
        }
    }

    observables
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setting L to 8 to manage dense real space arrays (Matrix Dim: 2209 x 2209)
    let l = 8; 
    
    // --- Basis Construction ---
    let states_1d = lucas_basis(l);
    let mut basis_states = Vec::new();
    for &a in &states_1d {
        for &b in &states_1d {
            basis_states.push(BasisState { a, b });
        }
    }
    
    // The pure Néel state
    let neel_key: u64 = (4u64.pow(l as u32 / 2) - 1) / 3;
    let neel_idx = basis_states.iter()
        .position(|s| s.a == neel_key && s.b == neel_key)
        .expect("Néel state missing from basis");

    let mut rho_0 = Array1::<Complex64>::zeros(basis_states.len());
    rho_0[neel_idx] = Complex64::new(1.0, 0.0);

    // Precompute diagonals
    let n_diag = occupation_number_diag(l, &basis_states);
    let corr_diag = density_correlation_nnn_diag(l, &basis_states);

    // ----------------------------------------------------------------
    // 2. PARAMETER SWEEP PREPARATION
    // ----------------------------------------------------------------
    let gp_values = Array1::linspace(0.001, 0.2, 2);
    let gm_values = Array1::linspace(0.001, 0.2, 2);
    let omega_values = Array1::linspace(1.0, 2.0, 1);

    let mut parameters = Vec::new();
    for &gp in &gp_values {
        for &gm in &gm_values {
            for &omega in &omega_values {
                parameters.push((gp, gm, omega));
            }
        }
    }

    println!("Starting parallel evolution sweep over {} points (Dim: {})...", parameters.len(), basis_states.len());

    // ----------------------------------------------------------------
    // 3. PARALLEL EVOLUTION
    // ----------------------------------------------------------------
    let results: Vec<String> = parameters.par_iter()
        .map(|&(gp, gm, omega)| {
            let l_cal_sparse = build_lindbladian(l, &basis_states, omega, gp, gm);
            
            // Convert to dense matrix for Time Evolution
            let mut l_cal_dense = Array2::<Complex64>::zeros((basis_states.len(), basis_states.len()));
            for (val, (row, col)) in l_cal_sparse.iter() { 
                l_cal_dense[[row, col]] = *val; 
            }
            
            // Simulate directly across the complete coupled dynamics space
            let total_obs = obs_evolution(&n_diag, &corr_diag, &basis_states, &rho_0, &l_cal_dense, neel_idx, 1e-3, 20.0);

            let formatted_data = total_obs.iter()
                .map(|(n, nn, fid)| format!("{:.16}, {:.16}, {:.16}", n, nn, fid))
                .collect::<Vec<String>>()
                .join(", ");

            format!("{},{},{},{}", gp, gm, omega, formatted_data)
        })
        .collect();

    // ----------------------------------------------------------------
    // 4. FILE OUTPUT
    // ----------------------------------------------------------------
    let mut file_occupation = File::create("occupation_time_asymmetric.csv")?;

    for line in results {
        writeln!(file_occupation, "{}", line)?;
    }
    
    println!("Simulation complete. Output written to occupation_time_asymmetric.csv");

    Ok(())
}