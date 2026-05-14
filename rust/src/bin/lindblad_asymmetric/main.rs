use ndarray::{Array1, Array2};
use ndarray_linalg::{Eig, Eigh, SVD};
use num_complex::Complex64;
use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::error::Error;

const THRESHOLD: f64 = 1e-10;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct BasisState {
    a: u64,
    b: u64,
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

fn occupation_number(
    l: usize,
    basis_states: &[BasisState],
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut n_cal = Array2::<Complex64>::zeros((dim, dim));
    
    for (i, state) in basis_states.iter().enumerate() {
        let mut count = 0;
        for j in 0..l {
            if (state.a & (1 << j)) != 0 {
                count += 1;
            }
        }
        n_cal[[i, i]] = Complex64::new(count as f64 / l as f64, 0.0);
    }
    
    n_cal
}

fn density_correlation_nnn(
    l: usize,
    basis_states: &[BasisState],
) -> Array2<Complex64> {
    let dim = basis_states.len();
    let mut corr_cal = Array2::<Complex64>::zeros((dim, dim));
    
    for (i, state) in basis_states.iter().enumerate() {
        let mut count = 0;
        for site in 0..l {
            let idx_minus = (site + l - 1) % l;
            let idx_plus = (site + 1) % l;
            if (state.a & (1 << idx_minus)) != 0 && (state.a & (1 << idx_plus)) != 0 {
                count += 1;
            }
        }
        corr_cal[[i, i]] = Complex64::new(count as f64 / l as f64, 0.0);
    }
    
    corr_cal
}

fn compute_trace(
    basis_states: &[BasisState],
    vectorized_op: &Array1<Complex64>,
) -> Complex64 {
    let mut trace = Complex64::new(0.0, 0.0);
    for (i, state) in basis_states.iter().enumerate() {
        if state.a == state.b {
            trace += vectorized_op[i];
        }
    }
    trace
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
    
    let mut states_1d = lucas_basis(l);
    states_1d.sort();
    let d = states_1d.len();
    
    let mut state_idx = HashMap::new();
    for (i, &s) in states_1d.iter().enumerate() {
        state_idx.insert(s, i);
    }
    
    let mut rho_mat = Array2::<Complex64>::zeros((d, d));
    for (i, state) in basis_states.iter().enumerate() {
        let row = state_idx[&state.a];
        let col = state_idx[&state.b];
        rho_mat[[row, col]] = vectorized_op[i];
    }

    // Symmetrize to fix minor numerical errors
    for row in 0..d {
        for col in 0..row {
            let avg = (rho_mat[[row, col]] + rho_mat[[col, row]].conj()) * 0.5;
            rho_mat[[row, col]] = avg;
            rho_mat[[col, row]] = avg.conj();
        }
        rho_mat[[row, row]] = Complex64::new(rho_mat[[row, row]].re, 0.0);
    }

    let mut results = Vec::new();
    if let Ok((eigvals, eigvecs)) = rho_mat.eigh(ndarray_linalg::UPLO::Upper) {
        let basis_occupations: Vec<f64> = states_1d.iter().map(|&a| {
            let particle_count = a.count_ones() as f64;
            particle_count / (l as f64)
        }).collect();

        for (idx, &lambda) in eigvals.iter().enumerate() {
            let eigenvector = eigvecs.column(idx).to_owned();
            let occ = compute_eigenstate_observable(&eigenvector, &basis_occupations);
            results.push((lambda, occ));
        }
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

pub struct SpectralData {
    pub real_eval: f64,
    pub imag_eval: f64,
    pub c_k: Complex64,
    pub o_k: Complex64,
    pub occupation_c_k: Complex64,
    pub occupation_o_k: Complex64,
    pub z_k: Complex64, 
}

fn analyze_lindbladian(
    right_evals: &Array1<Complex64>,
    left_evals_raw: &Array1<Complex64>,
    right_evecs: &Array2<Complex64>,
    left_evecs: &Array2<Complex64>,     
    occ_op: &Array2<Complex64>,
    rho: &Array1<Complex64>,
    basis_states: &[BasisState], 
) -> Result<Vec<SpectralData>, Box<dyn Error>> {
    
    let n_evals = right_evals.len();
    let mut results = Vec::with_capacity(n_evals);

    let mut r_tagged: Vec<(usize, Complex64)> = right_evals.iter().cloned().enumerate().collect();
    let mut l_tagged: Vec<(usize, Complex64)> = left_evals_raw
        .iter()
        .map(|e| e.conj())
        .enumerate()
        .collect();

    let sort_cmp = |a: &(usize, Complex64), b: &(usize, Complex64)| {
        b.1.re.partial_cmp(&a.1.re).unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.im.partial_cmp(&a.1.im).unwrap_or(std::cmp::Ordering::Equal))
    };

    r_tagged.sort_by(sort_cmp);
    l_tagged.sort_by(sort_cmp);

    for k in 0..n_evals {
        let (r_idx, current_val) = r_tagged[k]; 
        let (l_idx, _) = l_tagged[k];      
        
        let right_vec = right_evecs.column(r_idx);
        let left_vec = left_evecs.column(l_idx);

        let z_k = left_vec.mapv(|x| x.conj()).dot(&right_vec);

        let raw_c_k = left_vec.mapv(|x| x.conj()).dot(rho);
        let o_k = right_vec.mapv(|x| x.conj()).dot(rho);

        let c_k = raw_c_k / z_k;

        let aux_left = occ_op.dot(&left_vec);
        let raw_occupation_c_k = compute_trace(basis_states, &aux_left);
        let occupation_c_k = raw_occupation_c_k / z_k;

        let aux_right = occ_op.dot(&right_vec);
        let occupation_o_k = compute_trace(basis_states, &aux_right);

        results.push(SpectralData {
            real_eval: current_val.re,
            imag_eval: current_val.im,
            c_k,
            o_k,
            occupation_c_k,
            occupation_o_k,
            z_k,
        });
    }

    Ok(results)
}

struct SimulationResult {
    occupation_str: String,
    std_eigenvalues_str: String,
    eigenvalues_str: String,
    decay_str: String,
    oee_str: String,
}

// ==========================================
// ENTANGLEMENT ENTROPY FUNCTIONS
// ==========================================

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
    basis_states: &[BasisState],
    eigenvector: &[Complex64],
) -> Vec<(u64, u64, Complex64)> {
    let mut expanded = Vec::new();
    for (idx, &c) in eigenvector.iter().enumerate() {
        if c.norm() > 1e-12 {
            let state = &basis_states[idx];
            expanded.push((state.a, state.b, c));
        }
    }
    expanded
}

pub fn fibonacci_basis(l: usize) -> Vec<u64> {
    (0..(1u64 << l))
        .filter(|&state| state & (state >> 1) == 0)
        .collect()
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
    let fib_a = fibonacci_basis(l_a);
    
    let mut a_idx_map = HashMap::new();
    for (i, &state) in fib_a.iter().enumerate() {
        a_idx_map.insert(state, i);
    }
    
    let dim_a = fib_a.len();
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

fn is_diagonalizable(matrix: &Array2<Complex64>) -> bool {
    let (rows, cols) = matrix.dim();
    if rows != cols { return false; }
    
    match matrix.eig() {
        Ok((_, eigenvectors)) => {
            match eigenvectors.svd(false, false) {
                Ok((_, s, _)) => {
                    let max_sv = s[0];
                    let min_sv = s[s.len() - 1];
                    if min_sv == 0.0 { return false; }
                    let condition_number = max_sv / min_sv;
                    condition_number < 1e12 
                },
                Err(_) => false,
            }
        },
        Err(_) => false,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Changed l from 10 to 8: since the code no longer breaks down by Q sector,
    // the dense matrix dimension is the full D^2 x D^2 of the block.
    let l = 8; 

    let states_1d = lucas_basis(l);
    let mut basis_states = Vec::new();
    for &a in &states_1d {
        for &b in &states_1d {
            basis_states.push(BasisState { a, b });
        }
    }
    
    let neel_key: u64 = (4u64.pow(l as u32 / 2) - 1) / 3;
    let mut rho_vec_neel = Array1::<Complex64>::zeros(basis_states.len());
    if let Some(idx) = basis_states.iter().position(|s| s.a == neel_key && s.b == neel_key) {
        rho_vec_neel[idx] = Complex64::new(1.0, 0.0);
    }

    let n_matrix = occupation_number(l, &basis_states);
    let corr_matrix = density_correlation_nnn(l, &basis_states);
    
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

    println!("Starting parallel sweep over {} points (Dim: {})...", parameters.len(), basis_states.len());

    let results: Vec<SimulationResult> = parameters
        .par_iter()
        .map(|&(gp, gm, omega)| {
            let mut res = SimulationResult {
                occupation_str: String::new(),
                std_eigenvalues_str: String::new(),
                eigenvalues_str: String::new(),
                decay_str: String::new(),
                oee_str: String::new(),
            };

            let l_cal = build_lindbladian(l, &basis_states, omega, gp, gm);
            
            let l_cal_dense = {
                let mut dense = Array2::<Complex64>::zeros((basis_states.len(), basis_states.len()));
                for (val, (row, col)) in l_cal.iter() {
                    dense[[row, col]] = *val;
                }
                dense
            };

            if is_diagonalizable(&l_cal_dense) {
                println!("Diagonalizability: omega {}, gamma - {}, gamma + {} IS diagonalizable.", omega, gm, gp);
            } else {
                println!("Diagonalizability: omega {}, gamma - {}, gamma + {} is NOT diagonalizable.", omega, gm, gp);
            }

            if let Ok((evals, evecs)) = l_cal_dense.eig() {
                let l_cal_dag = l_cal_dense.t().mapv(|c| c.conj());

                if let Ok((evalsdag, evecsdag)) = l_cal_dag.eig() {
                    if let Ok(analysis) = analyze_lindbladian(
                        &evals, &evalsdag, &evecs, &evecsdag, &n_matrix, &rho_vec_neel, &basis_states
                    ) {
                        res.decay_str.push_str(&format!("{},{},{}", gp, gm, omega)); 
                        for data in analysis {
                            res.decay_str.push_str(&format!(",{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}", 
                                data.real_eval, 
                                data.imag_eval, 
                                data.c_k.re,
                                data.c_k.im, 
                                data.o_k.re,
                                data.o_k.im, 
                                data.occupation_c_k.re, 
                                data.occupation_c_k.im
                            ));
                        }
                        res.decay_str.push('\n');
                    }
                }

                res.oee_str.push_str(&format!("{},{},{}", gp, gm, omega));
                for (&lambda, vec_view) in evals.iter().zip(evecs.columns()) {
                    let vec_contiguous = vec_view.to_vec();
                    let expanded = expand_eigenmatrix(&basis_states, &vec_contiguous);
                    let rho_sq = compute_rho_dagger_rho(&expanded);
                    let entropy = operator_entanglement_entropy(&rho_sq, l).unwrap_or(0.0);
                    
                    res.oee_str.push_str(&format!(",{:.10},{:.10},{:.6}", lambda.re, lambda.im, entropy));
                }
                res.oee_str.push('\n');

                res.eigenvalues_str.push_str(&format!("{},{},{}", gp, gm, omega));
                for eval in evals.iter() {
                    res.eigenvalues_str.push_str(&format!(",{},{}", eval.re, eval.im));
                }
                res.eigenvalues_str.push('\n');

                for (i, eval) in evals.iter().enumerate() {
                    if eval.re.abs() < 1e-8 && eval.im.abs() < 1e-8 {
                        let rho_vec = evecs.column(i).to_owned();
                        let tr_rho = compute_trace(&basis_states, &rho_vec);
                        
                        if tr_rho.norm() > 1e-10 {
                            let spectrum_data = steady_state_properties(l, &basis_states, &rho_vec);
                            let formatted_data = spectrum_data.iter()
                                .map(|(p, n)| format!("{:.16}, {:.6}", p, n))
                                .collect::<Vec<String>>()
                                .join(", ");
                            res.std_eigenvalues_str.push_str(&format!("{},{},{},{}\n", gp, gm, omega, formatted_data));

                            let n_rho = n_matrix.dot(&rho_vec);
                            let nn_rho = corr_matrix.dot(&rho_vec);
                            
                            let tr_n = compute_trace(&basis_states, &n_rho);
                            let tr_nn = compute_trace(&basis_states, &nn_rho);

                            let exp_n = (tr_n / tr_rho).re;
                            let exp_nn = (tr_nn / tr_rho).re;
                            res.occupation_str.push_str(&format!("{},{},{},{},{}\n", gp, gm, omega, exp_n, exp_nn));
                        }
                    }
                }
            }

            res
        })
        .collect();

    let mut file_occupation = File::create("occupation_asymmetric.csv")?;
    writeln!(file_occupation, "gp,gm,omega,n,nn")?;
    
    let mut file_std = File::create("std_eigenvalues_asymmetric.csv")?;
    let mut file_evals = File::create("eigenvalues_asymmetric.csv")?;
    let mut file_decay = File::create("decay_asymmetric.csv")?;
    let mut file_oee = File::create("oee_asymmetric.csv")?;

    for res in results {
        file_occupation.write_all(res.occupation_str.as_bytes())?;
        file_std.write_all(res.std_eigenvalues_str.as_bytes())?;
        file_evals.write_all(res.eigenvalues_str.as_bytes())?;
        file_oee.write_all(res.oee_str.as_bytes())?;
        file_decay.write_all(res.decay_str.as_bytes())?;
    }

    Ok(())
}