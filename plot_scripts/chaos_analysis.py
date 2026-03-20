import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def classify_1d_dynamics(r_avg):
    """Classify 1D real spectra against Poisson and GOE benchmarks."""
    integrable = 0.386
    chaotic = 0.530
    
    dist_to_integrable = abs(r_avg - integrable)
    dist_to_chaotic = abs(r_avg - chaotic)
    
    if dist_to_integrable < dist_to_chaotic:
        return "Integrable (1D Poisson)"
    else:
        return "Chaotic (1D GOE)"

def classify_2d_dynamics(r_avg, cos_theta_avg):
    """Classify 2D complex spectra against Poisson and Ginibre benchmarks."""
    integrable = np.array([0.667, 0.0])
    chaotic = np.array([0.738, -0.24])
    
    actual = np.array([r_avg, cos_theta_avg])
    
    dist_to_integrable = np.linalg.norm(actual - integrable)
    dist_to_chaotic = np.linalg.norm(actual - chaotic)
    
    if dist_to_integrable < dist_to_chaotic:
        return "Integrable (2D Poisson)"
    else:
        return "Chaotic (2D Ginibre)"

def process_1d_spectrum(g, omega, reals):
    """Pipeline for purely real eigenvalues (1D limit)."""
    # Step 1: Filter to isolate the bulk
    mean_E = np.mean(reals)
    std_E = np.std(reals)
    bulk_mask = np.abs(reals - mean_E) <= 2 * std_E
    bulk_E = reals[bulk_mask]
    
    if len(bulk_E) < 4:
        return pd.Series({'g': g, 'omega': omega, 'r_avg': np.nan, 'cos_theta_avg': 0.0, 'classification': 'Insufficient Data'})
        
    # Sort the bulk real spectrum
    E_sorted = np.sort(bulk_E)
    
    # Step 2: Calculate consecutive level spacings
    s = np.diff(E_sorted)
    
    # Filter out exact numerical degeneracies to avoid division by zero
    s = s[s > 1e-12]
    
    if len(s) < 2:
        return pd.Series({'g': g, 'omega': omega, 'r_avg': np.nan, 'cos_theta_avg': 0.0, 'classification': 'Insufficient Data'})

    # Step 3: Compute the 1D ratios
    r_alpha = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    
    # Step 4: Compute the average and classify
    r_avg = np.mean(r_alpha)
    classification = classify_1d_dynamics(r_avg)
    
    return pd.Series({
        'g': g, 
        'omega': omega, 
        'r_avg': r_avg, 
        'cos_theta_avg': 0.0,  # Explicitly returning 0.0 for 1D real spectra
        'classification': classification
    })

def process_2d_spectrum(g, omega, complex_evals):
    """Pipeline for complex eigenvalues (2D Non-Hermitian regime)."""
    # Step 1: Filter to isolate the bulk
    nonzero_mask = ~np.isclose(complex_evals, 0.0, atol=1e-10)
    filtered_evals = complex_evals[nonzero_mask]
    
    if len(filtered_evals) < 3:
        return pd.Series({'g': g, 'omega': omega, 'r_avg': np.nan, 'cos_theta_avg': np.nan, 'classification': 'Insufficient Data'})
        
    mean_X = np.mean(np.real(filtered_evals))
    std_X = np.std(np.real(filtered_evals))
    
    bulk_mask = np.abs(np.real(filtered_evals) - mean_X) <= 2 * std_X
    bulk_evals = filtered_evals[bulk_mask]
    
    if len(bulk_evals) < 3:
        return pd.Series({'g': g, 'omega': omega, 'r_avg': np.nan, 'cos_theta_avg': np.nan, 'classification': 'Insufficient Data'})

    # Step 2: Find the nearest neighbors
    points = np.column_stack((np.real(bulk_evals), np.imag(bulk_evals)))
    tree = KDTree(points)
    
    distances, indices = tree.query(points, k=3)
    
    nn_indices = indices[:, 1]
    nnn_indices = indices[:, 2]
    
    lambda_k = bulk_evals
    lambda_NN = bulk_evals[nn_indices]
    lambda_NNN = bulk_evals[nnn_indices]
    
    # Step 3: Compute the Complex Spacing Ratios
    denom = lambda_NNN - lambda_k
    valid_ratios = np.abs(denom) > 1e-12
    
    z_k = (lambda_NN[valid_ratios] - lambda_k[valid_ratios]) / denom[valid_ratios]
    
    # Step 4: Compute the single-number statistical averages
    r_k = np.abs(z_k)
    cos_theta_k = np.real(z_k) / r_k 
    
    r_avg = np.mean(r_k)
    cos_theta_avg = np.mean(cos_theta_k)
    
    # Step 5: Classify the dynamics
    classification = classify_2d_dynamics(r_avg, cos_theta_avg)
    
    return pd.Series({
        'g': g, 
        'omega': omega, 
        'r_avg': r_avg, 
        'cos_theta_avg': cos_theta_avg, 
        'classification': classification
    })

def process_spectrum(row):
    """Main router function to handle both 1D and 2D spectra."""
    g = row.iloc[0]
    omega = row.iloc[1]
    
    vals = row.iloc[2:].values.astype(float)
    reals = vals[0::2]
    imags = vals[1::2]
    
    valid_mask = ~np.isnan(reals) & ~np.isnan(imags)
    reals = reals[valid_mask]
    imags = imags[valid_mask]
    
    # Route to 1D pipeline if omega is zero OR if all imaginary parts are effectively zero
    if np.isclose(omega, 0.0, atol=1e-10) or np.allclose(imags, 0.0, atol=1e-10):
        return process_1d_spectrum(g, omega, reals)
    else:
        complex_evals = reals + 1j * imags
        return process_2d_spectrum(g, omega, complex_evals)


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv('../rust/eigenvalues_10.csv', header=None) 
    results_df = df.apply(process_spectrum, axis=1)
    print_full(results_df)
    # pass