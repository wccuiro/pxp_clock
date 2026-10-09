"""
Reference time evolution of the Neel state under the PXP Lindbladian (independent of the Rust code).

    python3 tools/reference_evolution.py L T dt gp gm omega out.npy [alpha] [plus=0|1]

With alpha it is the partial projection model of src/bin/lindblad_alpha (full 2^L basis,
see pxp_reference.py); without it the constrained model. With plus=0 or plus=1 it is the staggered
model of src/bin/lindblad_asymmetric (sigma^+ only on the sites j = plus mod 2, sigma^- on the others).

rho(t) is propagated in the translation sectors Q = 0 and Q = L/2 (staggered: the sector Q = 0 of
the translation by two sites) with the dense matrix
exponential P = expm(dt L_Q) (scipy.linalg.expm, Pade + scaling and squaring), rho(t+dt) = P rho(t).
No Euler steps and no trace renormalization.

out.npy has one row per time t = i dt, i = 0 ... round(T/dt), with columns
    t, <n>, <nn>, F          n = (1/L) sum_j n_j,  nn = (1/L) sum_j n_{j-1} n_{j+1},  F = <Neel|rho(t)|Neel>

Printed checks: trace conservation, and the same observables from the eigen-expansion
sum_k c_k e^{lambda_k t} r_k of the dense diagonalization (a second, independent exact method;
its accuracy is limited by the conditioning of the eigenvectors).
"""
import sys
import time

import numpy as np
from scipy.linalg import expm

import pxp_reference as ref

def main():
  plus_site, step = ref.plus_site_argument(sys.argv)
  if len(sys.argv) not in (8, 9):
    sys.exit("usage: reference_evolution.py L T dt gp gm omega out.npy [alpha] [plus=0|1]")
  L = int(sys.argv[1])
  T, dt, gamma_plus, gamma_minus, omega = (float(x) for x in sys.argv[2:7])
  out_file = sys.argv[7]
  alpha = float(sys.argv[8]) if len(sys.argv) == 9 else None

  steps = int(round(T / dt))
  times = np.arange(steps + 1) * dt

  t0 = time.time()
  states, index, Lind = ref.build_model(L, gamma_plus, gamma_minus, omega, alpha, plus_site)
  f = ref.trace_vectors(L, states, index)
  model = "constrained" if alpha is None else f"alpha = {alpha}"
  if plus_site is not None:
    model += f", staggered (sigma+ on sites j = {plus_site} mod 2)"
  print(f"L = {L}, {model}: {len(states)} configurations, {steps} steps of dt = {dt}  (built in {time.time()-t0:.1f}s)")

  names = ('n', 'nn', 'rho0')
  obs = np.zeros((steps + 1, 3))
  obs_eig = np.zeros((steps + 1, 3))
  trace = np.zeros(steps + 1)

  for Q in ref.sectors(L, step):
    t0 = time.time()
    B = ref.sector_basis(L, states, index, Q, step)
    L_Q, residual = ref.reduce_to_sector(Lind, B)
    f_Q = {name: B.T @ vec for name, vec in f.items()}

    # propagator
    P = expm(L_Q * dt)
    rho = f_Q['rho0'].astype(complex)
    for i in range(steps + 1):
      for j, name in enumerate(names):
        obs[i, j] += np.real(f_Q[name] @ rho)
      trace[i] += np.real(f_Q['tr'] @ rho)
      rho = P @ rho

    # eigen-expansion
    eigvals, V = np.linalg.eig(L_Q)
    c = np.linalg.solve(V, f_Q['rho0'])
    phases = np.exp(np.outer(times, eigvals))
    for j, name in enumerate(names):
      obs_eig[:, j] += np.real(phases @ (c * (f_Q[name] @ V)))

    print(f"  Q={Q}: dim {L_Q.shape[0]}  sector residual {residual:.1e}  ({time.time()-t0:.1f}s)")

  print(f"  max |Tr rho(t) - 1| = {np.abs(trace - 1).max():.1e}")
  diff = np.abs(obs - obs_eig).max(axis=0)
  print(f"  propagator vs eigen-expansion: max |dn| = {diff[0]:.1e}  |dnn| = {diff[1]:.1e}  |dF| = {diff[2]:.1e}")
  print(f"  t = {times[-1]:g}: <n> = {obs[-1, 0]:.12f}  <nn> = {obs[-1, 1]:.12f}  F = {obs[-1, 2]:.12f}")

  np.save(out_file, np.column_stack([times, obs]))
  print(f"written {out_file}")

if __name__ == "__main__":
  main()
