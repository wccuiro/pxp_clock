"""
Compare the output of the Rust time evolution with a reference written by reference_evolution.py.

    python3 tools/compare_evolution.py ref.npy occupation_time.csv [gp gm omega [alpha]]

occupation_time.csv is the file written by lindblad_evol, lindblad_asymmetric_evol or lindblad_alpha_evol
(occupation_time[_staggered|_alpha]_evol_L.._gp.._gm.._omega..[_alpha..].csv). It has one line per
parameter point: gp, gm, omega, then n, nn, F at t = i dt; the partial projection file starts with
alpha, gp, gm, omega instead.
With several lines, give gp gm omega (and alpha) to choose the one to compare.

Exit status 1 if any difference is above the tolerance.
"""
import sys

import numpy as np

TOL = 1e-10

def read_lines(path):
  """List of ([gp, gm, omega] or [gp, gm, omega, alpha], array(-1, 3))."""
  lines = []
  with open(path, 'r') as f:
    for line in f:
      vals = line.strip().split(',')
      try:
        vals = np.array(vals, dtype=float)
      except ValueError:
        continue
      if len(vals) >= 6 and (len(vals) - 3) % 3 == 0:
        lines.append((list(vals[:3]), vals[3:].reshape(-1, 3)))
      elif len(vals) >= 7 and (len(vals) - 4) % 3 == 0:
        lines.append((list(vals[1:4]) + [vals[0]], vals[4:].reshape(-1, 3)))
  return lines

def describe(head):
  return f"gp={head[0]:g} gm={head[1]:g} omega={head[2]:g}" + (f" alpha={head[3]:g}" if len(head) == 4 else "")

def main():
  if len(sys.argv) not in (3, 6, 7):
    sys.exit("usage: compare_evolution.py ref.npy occupation_time.csv [gp gm omega [alpha]]")
  ref = np.load(sys.argv[1])
  times, ref = ref[:, 0], ref[:, 1:]
  lines = read_lines(sys.argv[2])
  if not lines:
    sys.exit(f"{sys.argv[2]}: no line of the form [alpha,]gp,gm,omega,(n,nn,F)*")

  if len(sys.argv) > 3:
    params = [float(x) for x in sys.argv[3:]]
    lines = [(head, data) for head, data in lines
             if len(head) == len(params) and np.allclose(head, params, rtol=1e-9, atol=1e-12)]
    if not lines:
      sys.exit(f"{sys.argv[2]}: no line with {describe(params)}")
  if len(lines) > 1:
    listing = "\n".join("  " + describe(h) for h, _ in lines)
    sys.exit(f"{sys.argv[2]} has {len(lines)} matching lines, give gp gm omega [alpha] to choose one:\n{listing}")

  head, data = lines[0]
  if len(data) != len(ref):
    sys.exit(f"different number of time points: reference {len(ref)}, run {len(data)} (same T and dt?)")

  print(f"run: {describe(head)}, {len(data)} time points up to t = {times[-1]:g}")
  diff = np.abs(data - ref)
  failed = False
  for j, name in enumerate(("<n>", "<nn>", "F")):
    i = np.argmax(diff[:, j])
    ok = diff[i, j] <= TOL
    failed = failed or not ok
    print(f"  {name:5s} max difference {diff[i, j]:.2e} at t = {times[i]:g}   (tolerance {TOL:.0e})   {'ok' if ok else 'FAIL'}")
  sys.exit(1 if failed else 0)

if __name__ == "__main__":
  main()
