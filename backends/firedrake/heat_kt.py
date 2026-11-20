"""mosaic#375 idea 4 follow-on on Firedrake: temperature-dependent conductivity, a Newton solve per step.

    python3 backends/firedrake/heat_kt.py --nx 16 --ny 8 --steps 50 --newton 3 --diff unrolled --out result.json

The scheme of backends/jax/heat_kt.py: Q1 elements, k_e = k0 rho_e (1 + beta Tbar_e) with Tbar_e the cell average
(cell_avg; on a bilinear rectangle that is the mean of the four nodal values, as in the JAX backend), and per
backward Euler step `--newton` Newton iterations from T_n on
    R(T) = (T - T_n, v) + dt (k(rho, T) grad T, grad v) = 0,   T = 0 at x = 0.

Differentiation of that solve (`--diff`), by pyadjoint:
  unrolled  the Newton iterations written out as linear solves on the tape, so the gradient is exact for the iterate
            reached and carries the Newton error
  implicit  a NonlinearVariationalSolver that takes exactly `--newton` iterations (convergence test skipped);
            pyadjoint differentiates it by the implicit function theorem at the iterate reached, which is exact at
            a converged root and approximate otherwise
`--reference` computes the implicit gradient at 30 iterations and checks it against central differences.
"""

import argparse
import json
import os
import sys

import numpy as np
from firedrake import (
    Constant, DirichletBC, Function, LinearVariationalProblem, LinearVariationalSolver, NonlinearVariationalProblem,
    NonlinearVariationalSolver, SpatialCoordinate, TestFunction, TrialFunction, VectorFunctionSpace, assemble,
    cell_avg, derivative, dx, grad, inner,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, get_working_tape, pause_annotation, stop_annotating,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backends.firedrake.heat import Heat  # noqa: E402
from spike.problem import Problem  # noqa: E402

BETA = 1.0
LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def run(h: Heat, rho: Function, steps: int, n_newton: int, diff: str) -> float:
    """Backward Euler with a fixed number of Newton iterations per step; returns J = integral T^2."""
    V, p = h.V, h.p
    dt = Constant(p.t_end / steps)
    T_old = Function(V).interpolate(h.T0_expr)
    T = Function(V)
    v = TestFunction(V)

    def residual(u):
        k = p.k0 * rho * (1 + BETA * cell_avg(u))
        return (u - T_old) * v * dx + dt * k * inner(grad(u), grad(v)) * dx

    if diff == "implicit":
        F = residual(T)
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, T, bcs=h.bc),
            solver_parameters={**LU, "snes_type": "newtonls", "snes_linesearch_type": "basic",
                               "snes_max_it": n_newton, "snes_convergence_test": "skip"},
        )
        for _ in range(steps):
            T.assign(T_old)  # Newton starts from the previous step
            solver.solve()
            T_old.assign(T)
    else:
        dT = Function(V)
        F = residual(T)  # assembled at the current T on every solve
        newton_step = LinearVariationalSolver(
            LinearVariationalProblem(derivative(F, T, TrialFunction(V)), F, dT, bcs=DirichletBC(V, 0.0, 1)),
            solver_parameters=LU,
        )
        for _ in range(steps):
            T.assign(T_old)
            for _ in range(n_newton):
                newton_step.solve()
                T.assign(T - dT)
            T_old.assign(T)
    return assemble(T_old * T_old * dx)


def jax_direction(h: Heat) -> np.ndarray:
    """The random direction of backends/jax/heat_kt.py, in Firedrake's cell order, so grad_dir compares."""
    p = h.p
    c = Function(VectorFunctionSpace(h.mesh, "DQ", 0)).interpolate(SpatialCoordinate(h.mesh)).dat.data_ro
    jax_cell = (c[:, 1] // (p.Ly / p.ny)).astype(int) * p.nx + (c[:, 0] // (p.Lx / p.nx)).astype(int)
    d = np.random.default_rng(0).standard_normal(p.n_cells)
    return (d / np.linalg.norm(d))[jax_cell]


def gradient(h: Heat, steps: int, n_newton: int, diff: str):
    tape = get_working_tape()
    tape.clear_tape()
    rho = Function(h.D).assign(1.0)
    continue_annotation()
    J = run(h, rho, steps, n_newton, diff)
    pause_annotation()
    g = ReducedFunctional(J, Control(rho)).derivative()
    out = float(J), g.dat.data_ro.copy()
    tape.clear_tape()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--newton", type=int, default=3)
    ap.add_argument("--diff", default="unrolled", choices=["unrolled", "implicit"])
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    h = Heat(Problem(nx=a.nx, ny=a.ny))
    res = {"backend": "firedrake", "problem": "k(T)", "beta": BETA, "diff": a.diff, "newton": a.newton,
           "nx": a.nx, "ny": a.ny, "steps": a.steps}
    try:
        d = jax_direction(h)
        if a.reference:
            res.update(diff="reference", newton=30)
            res["J"], g = gradient(h, a.steps, 30, "implicit")
            res["dJ_uniform"] = float(g.sum())
            eps = 1e-5
            with stop_annotating():
                fp, fm = Function(h.D), Function(h.D)
                fp.dat.data[:] = 1.0 + eps * d
                fm.dat.data[:] = 1.0 - eps * d
                res["fd_dir"] = (run(h, fp, a.steps, 30, "implicit") - run(h, fm, a.steps, 30, "implicit")) / (2 * eps)
            res["grad_dir"] = float(g @ d)
        else:
            res["J"], g = gradient(h, a.steps, a.newton, a.diff)
            res["dJ_uniform"] = float(g.sum())
            res["grad_dir"] = float(g @ d)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
