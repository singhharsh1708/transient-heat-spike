"""Every cell's gradient from the JAX backend against its closed form, not only their sum.

    python spike/check_full_gradient.py

The initial temperature s is an exact eigenvector of the discrete scheme and the adjoint stays in the same mode, so
at uniform density dJ/dk_e = (s' K_e s / s' K s) dJ/deps: each cell's share of the mode's stiffness energy times
the exact derivative along uniform conductivity (discrete_exact). Prints the largest cell error over the largest
cell gradient for each mesh, step count and strategy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backends", "jax"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from heat import build  # noqa: E402

from spike.problem import Problem, discrete_exact, element_matrices, initial_temperature, mesh  # noqa: E402

for nx, ny in ((32, 16), (64, 32), (128, 64)):
    p = Problem(nx=nx, ny=ny)
    coords, cells, free = mesh(p)
    ke, _ = element_matrices(p.Lx / p.nx, p.Ly / p.ny)
    s = np.zeros(len(coords))
    s[free] = initial_temperature(coords, p)[free]
    share = np.array([s[c] @ ke @ s[c] for c in cells])
    share /= share.sum()
    for steps in (10, 100, 1000, 10000):
        for strategy in ("tape", "checkpoint"):
            g = np.asarray(jax.jit(jax.grad(build(p, steps, strategy)))(jnp.ones(p.n_cells)))
            exact = share * discrete_exact(p, steps)[1]
            err = np.max(np.abs(g - exact)) / np.max(np.abs(exact))
            print(f"{nx} x {ny}, {steps} steps, {strategy}: max cell error / max cell gradient {err:.1e}", flush=True)
