"""Shared by the studies: the spike's Q1 heat matrices (dense, free nodes) and a fixed-count Jacobi-PCG."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spike.problem import Problem, element_matrices, mesh  # noqa: E402

jax.config.update("jax_enable_x64", True)


def setup(nx, ny):
    p = Problem(nx=nx, ny=ny)
    coords, cells, free = mesh(p)
    ke, me = element_matrices(p.Lx / p.nx, p.Ly / p.ny)
    n, nf = len(coords), len(free)
    P = np.zeros((n, nf)); P[free, np.arange(nf)] = 1
    Kc = np.zeros((len(cells), nf, nf)); M = np.zeros((n, n))
    for e, c in enumerate(cells):
        Kf = np.zeros((n, n)); Kf[np.ix_(c, c)] = ke; Kc[e] = P.T @ Kf @ P
        M[np.ix_(c, c)] += me
    M = P.T @ M @ P
    x, y = coords[free, 0], coords[free, 1]
    cx = np.array([coords[c, 0].mean() for c in cells]); cy = np.array([coords[c, 1].mean() for c in cells])
    return p, jnp.asarray(Kc), jnp.asarray(M), x, y, cx, cy


def pcg(A, b, x0, dinv, iters):
    xk = x0; r = b - A @ xk; z = dinv * r; pk = z; rz = r @ z
    for _ in range(iters):
        Ap = A @ pk; a = rz / (pk @ Ap); xk = xk + a * pk; r = r - a * Ap
        z = dinv * r; rzn = r @ z; pk = z + (rzn / rz) * pk; rz = rzn
    return xk
