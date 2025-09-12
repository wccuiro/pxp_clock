import numpy as np

# optional scipy sparse + eigs
try:
    from scipy.sparse import coo_matrix, csr_matrix, issparse
    from scipy.sparse.linalg import eigs
    SCIPY = True
except Exception:
    SCIPY = False
    def issparse(x): return False

# 1) basis
def fibonacci_basis(L):
    """Return (states, index) where states are ints with no adjacent 1s."""
    states = [s for s in range(1 << L) if (s & (s << 1)) == 0]
    index = {s: i for i, s in enumerate(states)}
    return states, index

# 2) build W (interior flips only: j=1..L-2)
def build_W_interior(L, states, index, gamma_plus, gamma_minus, sparse=True):
    """
    Return W and (states, index). W is csr sparse if SCIPY and sparse requested,
    otherwise a dense numpy array. Columns are source states (column-stochastic generator).
    """
    N = len(states)
    rows, cols, vals = [], [], []

    for m in states:
        m_idx = index[m]
        exit_rate = 0.0
        for j in range(1, L - 1):
            # neighbor check
            if ((m >> (j - 1)) & 1) or ((m >> (j + 1)) & 1):
                continue
            bit = (m >> j) & 1
            if bit == 0:
                target = m | (1 << j); rate = gamma_plus
            else:
                target = m & ~(1 << j); rate = gamma_minus
            if target in index:
                rows.append(index[target]); cols.append(m_idx); vals.append(rate)
                exit_rate += rate
        if exit_rate != 0.0:
            rows.append(m_idx); cols.append(m_idx); vals.append(-exit_rate)

    if SCIPY and sparse:
        W = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        return W, states, index
    else:
        Wd = np.zeros((N, N), dtype=float)
        for r, c, v in zip(rows, cols, vals):
            Wd[r, c] += v
        return Wd, states, index

# 3) find eigenpairs with |lambda| < threshold
def find_near_zero_eigs(W, threshold=1e-10, k_sparse=None):
    """
    Return (eigvals_selected, eigvecs_selected) where eigvecs are columns.
    Tries scipy.sparse.linalg.eigs once if W is sparse; otherwise dense np.linalg.eig.
    """
    N = W.shape[0]
    # sparse attempt
    if SCIPY and issparse(W):
        k = k_sparse if k_sparse is not None else min(6, max(1, N - 1))
        k = min(k, N - 1) if N > 1 else 1
        try:
            vals, vecs = eigs(W, k=k, sigma=0.0)
            mask = np.abs(vals) < threshold
            if np.any(mask):
                return np.array(vals[mask]), np.column_stack([vecs[:, i] for i in np.where(mask)[0]])
        except Exception:
            pass  # fall through to dense
    # dense fallback
    if SCIPY and issparse(W):
        Wd = W.toarray()
    else:
        Wd = np.array(W, dtype=float)
    vals_all, vecs_all = np.linalg.eig(Wd)
    mask = np.abs(vals_all) < threshold
    if not np.any(mask):
        return np.array([]), np.empty((W.shape[0], 0), dtype=complex)
    vals_sel = np.array(vals_all[mask])
    vecs_sel = np.column_stack([vecs_all[:, i] for i in np.where(mask)[0]])
    return vals_sel, vecs_sel

# 4) compute overlaps with the two Neel patterns
def neel_patterns(L):
    a = sum((1 << j) for j in range(0, L, 2))
    b = sum((1 << j) for j in range(1, L, 2))
    return a, b

def compute_neel_overlap(eigvecs, index, L):
    """
    eigvecs: matrix with eigenvectors as columns (complex or real).
    index: map from state int -> index in basis.
    Returns list of dicts with eig vector info per column.
    """
    neel_a, neel_b = neel_patterns(L)
    in_a = neel_a in index; in_b = neel_b in index
    ia = index[neel_a] if in_a else None
    ib = index[neel_b] if in_b else None

    overlaps = []
    for col in range(eigvecs.shape[1]):
        v = eigvecs[:, col]
        # prefer real if effectively real
        if np.max(np.abs(v.imag)) < 1e-12:
            v = v.real.astype(float)
        norm = np.linalg.norm(v)
        vnorm = v / norm if norm != 0 else v
        amp_a = float(vnorm[ia]) if in_a else None
        amp_b = float(vnorm[ib]) if in_b else None
        overlaps.append({
            "amp_neel_a": amp_a,
            "amp_neel_b": amp_b,
            "prob_neel_a": None if amp_a is None else float(abs(amp_a)**2),
            "prob_neel_b": None if amp_b is None else float(abs(amp_b)**2),
            "l2_norm_before": float(norm)
        })
    return overlaps

# -------------------------
# Example minimal workflow
# -------------------------
if __name__ == "__main__":
    L = 20
    gamma_plus = 1.0
    gamma_minus = 0.5
    threshold = 1e-10

    # 1) basis
    states, index = fibonacci_basis(L)

    # 2) build W
    W, _, _ = build_W_interior(L, states, index, gamma_plus, gamma_minus, sparse=True)

    # 3) find near-zero eigenpairs
    eigvals, eigvecs = find_near_zero_eigs(W, threshold=threshold)

    # 4) compute overlaps
    overlaps = compute_neel_overlap(eigvecs, index, L)

    # compact print
    print(f"L={L}, N_states={len(states)}, found {eigvecs.shape[1]} modes with |lambda|<{threshold}")
    for i, v in enumerate(eigvals):
        prob_a = overlaps[i]["prob_neel_a"] if overlaps else None
        prob_b = overlaps[i]["prob_neel_b"] if overlaps else None
        print(f" mode {i}: eig={v:.3e}, prob(NeelA)={prob_a}, prob(NeelB)={prob_b}")
