"""spike/problem.py's discrete_exact against the same closed form at 40 significant digits (mpmath).

    python spike/check_reference_mp.py      (needs mpmath, e.g. mpmath==1.3.0)

The closed form (see discrete_exact's docstring) is J_h = Ly |f0|_M^2 (1 + dt lam1)^-2N and
dJ_h/deps = -2N dt lam1 J_h / (1 + dt lam1). Here every quantity, including the 1D mass matrix product and lam1, is
evaluated with mpmath at 40 digits, on the meshes and step counts of the CI sweep, and discrete_exact must agree to
1e-15 relative.
"""

import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spike.problem import Problem, discrete_exact  # noqa: E402

mp.mp.dps = 40
worst = 0.0
for nx, ny in ((32, 16), (64, 32), (128, 64)):
    p = Problem(nx=nx, ny=ny)
    Lx, Ly, k0 = (mp.mpf(repr(v)) for v in (p.Lx, p.Ly, p.k0))
    t_end = mp.mpf(repr(p.lam_t_end)) / (k0 * (mp.pi / (2 * Lx)) ** 2)  # derived here, not taken from float64
    h = Lx / nx
    f = [mp.sin(mp.pi * (i * h) / (2 * Lx)) for i in range(1, nx + 1)]  # x = 0 held at zero
    # f' Mx f for the P1 mass matrix with the left node removed: h/6 [2 1; 1 2] per element.
    fmf = mp.mpf(0)
    for e in range(nx):
        a = f[e - 1] if e > 0 else mp.mpf(0)
        b = f[e]
        fmf += h / 6 * (2 * a * a + 2 * a * b + 2 * b * b)
    th = mp.pi / (2 * nx)
    lam1 = 6 * k0 / h**2 * (1 - mp.cos(th)) / (2 + mp.cos(th))
    for steps in (10, 100, 1000, 10000, 100000):
        a = t_end / steps * lam1
        J = Ly * fmf * (1 + a) ** (-2 * steps)
        dJ = -2 * steps * a * J / (1 + a)
        J64, dJ64 = discrete_exact(p, steps)
        err = max(abs((mp.mpf(J64) - J) / J), abs((mp.mpf(dJ64) - dJ) / dJ))
        worst = max(worst, float(err))
        print(f"{nx}x{ny} {steps:>6} steps: J {mp.nstr(J, 20)}  dJ {mp.nstr(dJ, 20)}  rel diff {float(err):.1e}")
print(f"worst relative difference {worst:.1e}")
assert worst < 1e-15
