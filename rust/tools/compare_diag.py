"""
Compare the output of the Rust diagonalization with a reference written by reference_diag.py.

    python3 tools/compare_diag.py ref.json run_dir

run_dir holds eigenvalues*.csv, decay*.csv and (optional) oee*.csv, occupation*.csv,
std_eigenvalues*.csv, either with the plain names or with the _L.._gp.._gm.._omega.. suffix.
The rows with the q, gp, gm, omega of the reference are used.

Eigenvalues are matched by assignment (Hungarian), never by sorting. c_k and o_k carry the
arbitrary phase of the eigenvector, so |c|, |o|, c conj(o) and w are compared. Inside a group
of degenerate eigenvalues the eigenvectors are not unique: there only the sums of c conj(o)
and of w over the group are compared, and |c|, |o| and the OEE are skipped.

The same holds for the steady-state spectrum: the occupation is compared as the mean over
each group of degenerate weights p.

Exit status 1 if any difference is above its tolerance.
"""
import sys
import glob
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# decay.csv is printed with 10 decimals, oee.csv with 6
TOL = {"eigenvalues": 1e-9, "|c|": 1e-8, "|o|": 1e-8, "c conj(o)": 1e-8, "w": 1e-8,
       "OEE": 5e-6, "steady n, nn": 1e-10, "steady spectrum p": 1e-10, "steady spectrum n": 5e-6}
TOL_DEGENERATE = 1e-8

def read_rows(path, width):
  """List of ([q, gp, gm, omega], array(-1, width)); header lines are skipped."""
  rows = []
  with open(path, 'r') as f:
    for line in f:
      vals = line.strip().split(',')
      try:
        head = [float(x) for x in vals[:4]]
        data = np.array(vals[4:], dtype=float)
      except ValueError:
        continue
      if len(head) < 4 or len(data) % width != 0:
        continue
      rows.append((head, data.reshape(-1, width)))
  return rows

def select_row(rows, q, params):
  for head, data in rows:
    if np.allclose(head, [q] + params, rtol=1e-12, atol=1e-12):
      return data
  return None

def find_run(run_dir, ref):
  """Suffix of the newest eigenvalues*.csv that has both sectors of the reference."""
  params = [ref["gp"], ref["gm"], ref["omega"]]
  found = []
  for path in glob.glob(os.path.join(run_dir, "eigenvalues*.csv")):
    rows = read_rows(path, 2)
    ok = True
    for q, sec in ref["sectors"].items():
      data = select_row(rows, int(q), params)
      ok = ok and data is not None and len(data) == sec["dim"]
    if ok:
      found.append(path)
  if not found:
    sys.exit(f"no eigenvalues*.csv in {run_dir} with L={ref['L']} gp={params[0]} gm={params[1]} omega={params[2]}")
  path = max(found, key=os.path.getmtime)
  return os.path.basename(path)[len("eigenvalues"):]

def assign(a, b):
  """Permutation perm with a[i] <-> b[perm[i]] minimizing sum |a - b|."""
  cost = np.abs(a[:, None] - b[None, :])
  i, perm = linear_sum_assignment(cost)
  return perm, cost[i, perm]

def degenerate_groups(lam):
  near = np.abs(lam[:, None] - lam[None, :]) < TOL_DEGENERATE
  _, label = connected_components(csr_matrix(near), directed=False)
  return label

def main():
  if len(sys.argv) != 3:
    sys.exit("usage: compare_diag.py ref.json run_dir")
  with open(sys.argv[1], 'r') as f:
    ref = json.load(f)
  run_dir = sys.argv[2]
  params = [ref["gp"], ref["gm"], ref["omega"]]

  suffix = find_run(run_dir, ref)
  path = lambda name: os.path.join(run_dir, name + suffix)
  print(f"reference: L={ref['L']} gp={params[0]} gm={params[1]} omega={params[2]}")
  print(f"run:       {path('eigenvalues')}")

  worst = {}
  def record(name, value):
    worst[name] = max(worst.get(name, 0.0), float(value))

  have_decay = os.path.exists(path("decay"))
  have_oee = os.path.exists(path("oee"))

  for q, sec in ref["sectors"].items():
    q = int(q)
    lam = np.array(sec["lambda_re"]) + 1j * np.array(sec["lambda_im"])
    c = np.array(sec["c_re"]) + 1j * np.array(sec["c_im"])
    o = np.array(sec["o_re"]) + 1j * np.array(sec["o_im"])
    w = np.array(sec["w_re"]) + 1j * np.array(sec["w_im"])
    oee = np.array(sec["oee"])
    label = degenerate_groups(lam)
    size = np.bincount(label)
    single = size[label] == 1
    line = f"  Q={q}: dim {len(lam)}, {np.sum(~single)} modes in degenerate groups"

    # eigenvalues (full precision file)
    ev = select_row(read_rows(path("eigenvalues"), 2), q, params)
    _, dist = assign(lam, ev[:, 0] + 1j * ev[:, 1])
    record("eigenvalues", dist.max())
    line += f" | eigenvalues {dist.max():.1e}"

    # Neel coefficients
    if have_decay:
      d = select_row(read_rows(path("decay"), 8), q, params)
      if d is None or len(d) != len(lam):
        sys.exit(f"{path('decay')}: no row for Q={q} with {len(lam)} modes of 8 columns")
      perm, _ = assign(lam, d[:, 0] + 1j * d[:, 1])
      c_r = (d[:, 2] + 1j * d[:, 3])[perm]
      o_r = (d[:, 4] + 1j * d[:, 5])[perm]
      w_r = (d[:, 6] + 1j * d[:, 7])[perm]
      if np.any(np.isnan(c_r)):
        sys.exit(f"{path('decay')}: NaN coefficients in Q={q}")
      d_c = np.abs(np.abs(c) - np.abs(c_r))[single].max(initial=0.0)
      d_o = np.abs(np.abs(o) - np.abs(o_r))[single].max(initial=0.0)
      d_co = np.abs(np.bincount(label, (c * np.conj(o) - c_r * np.conj(o_r)).real)
                    + 1j * np.bincount(label, (c * np.conj(o) - c_r * np.conj(o_r)).imag)).max()
      d_w = np.abs(np.bincount(label, (w - w_r).real) + 1j * np.bincount(label, (w - w_r).imag)).max()
      record("|c|", d_c); record("|o|", d_o); record("c conj(o)", d_co); record("w", d_w)
      line += f" | |c| {d_c:.1e}  |o| {d_o:.1e}  c conj(o) {d_co:.1e}  w {d_w:.1e}"

    # operator entanglement
    if have_oee:
      e = select_row(read_rows(path("oee"), 3), q, params)
      if e is None or len(e) != len(lam):
        sys.exit(f"{path('oee')}: no row for Q={q} with {len(lam)} modes of 3 columns")
      perm, _ = assign(lam, e[:, 0] + 1j * e[:, 1])
      d_e = np.abs(oee - e[perm, 2])[single].max(initial=0.0)
      record("OEE", d_e)
      line += f" | OEE {d_e:.1e}"

    print(line)

  # steady state
  if ref["steady"] is not None and os.path.exists(path("occupation")):
    occ = select_row(read_rows(path("occupation"), 2), 0, params)
    if occ is not None:
      record("steady n, nn", max(abs(occ[0, 0] - ref["steady"]["n"]), abs(occ[0, 1] - ref["steady"]["nn"])))
  if ref["steady"] is not None and os.path.exists(path("std_eigenvalues")):
    spec = select_row(read_rows(path("std_eigenvalues"), 2), 0, params)
    p, n = np.array(ref["steady"]["spectrum_p"]), np.array(ref["steady"]["spectrum_n"])
    if spec is not None and len(spec) == len(p):
      # the eigenvectors of a degenerate p are not unique: compare the mean occupation of each group
      perm, _ = assign(p, spec[:, 0])
      label = degenerate_groups(p)
      d_n = np.bincount(label, n - spec[perm, 1]) / np.bincount(label)
      record("steady spectrum p", np.abs(p - spec[perm, 0]).max())
      record("steady spectrum n", np.abs(d_n).max())

  print()
  failed = False
  for name, tol in TOL.items():
    if name not in worst:
      print(f"  {name:20s} not available in the run")
      continue
    ok = worst[name] <= tol
    failed = failed or not ok
    print(f"  {name:20s} max difference {worst[name]:.2e}   (tolerance {tol:.0e})   {'ok' if ok else 'FAIL'}")
  sys.exit(1 if failed else 0)

if __name__ == "__main__":
  main()
