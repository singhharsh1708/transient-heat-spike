"""Firedrake backend: the same Q1 FEM and backward Euler as spike/problem.py, gradients by pyadjoint.

    python3 backends/firedrake/heat.py --nx 32 --ny 16 --steps 1000 --strategy tape --out result.json

Strategies (pyadjoint's discrete adjoint):
  tape        the default tape: every step's solution is kept for the reverse sweep
  checkpoint  checkpoint_schedules' MultistageCheckpointSchedule (Revolve) with ceil(sqrt(steps)) snapshots in
              memory: forward steps are recomputed from the snapshots on the way back
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, LinearVariationalProblem, LinearVariationalSolver,
    RectangleMesh, SpatialCoordinate, TestFunction, TrialFunction, assemble, dx, grad, inner, pi, sin,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, get_working_tape, pause_annotation, stop_annotating,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from spike.problem import Problem  # noqa: E402

SOLVER = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


class Heat:
    def __init__(self, p: Problem):
        self.p = p
        self.mesh = RectangleMesh(p.nx, p.ny, p.Lx, p.Ly, quadrilateral=True)
        self.V = FunctionSpace(self.mesh, "Q", 1)
        self.D = FunctionSpace(self.mesh, "DQ", 0)
        x, _ = SpatialCoordinate(self.mesh)
        self.T0_expr = sin(pi * x / (2 * p.Lx))
        self.bc = DirichletBC(self.V, 0.0, 1)  # boundary 1 of RectangleMesh is x = 0

    def run(self, rho: Function, steps: int, timestepper=None) -> float:
        """Backward Euler to t_end; returns J = integral T^2. Annotated when annotation is on."""
        V, p = self.V, self.p
        dt = Constant(p.t_end / steps)
        T_old = Function(V).interpolate(self.T0_expr)
        T = Function(V)
        u, v = TrialFunction(V), TestFunction(V)
        a = u * v * dx + dt * p.k0 * rho * inner(grad(u), grad(v)) * dx
        L = T_old * v * dx
        solver = LinearVariationalSolver(
            LinearVariationalProblem(a, L, T, bcs=self.bc, constant_jacobian=True), solver_parameters=SOLVER
        )
        it = range(steps) if timestepper is None else timestepper(iter(range(steps)))
        for _ in it:
            solver.solve()
            T_old.assign(T)
        return assemble(T_old * T_old * dx)


def gradient(h: Heat, steps: int, strategy: str):
    tape = get_working_tape()
    tape.clear_tape()
    rho = Function(h.D).assign(1.0)
    if strategy == "checkpoint":
        from checkpoint_schedules import MultistageCheckpointSchedule

        tape.enable_checkpointing(MultistageCheckpointSchedule(steps, max(1, math.ceil(math.sqrt(steps))), 0))
        timestepper = tape.timestepper
    elif strategy == "tape":
        timestepper = None
    else:
        raise ValueError(strategy)
    continue_annotation()
    J = h.run(rho, steps, timestepper)
    pause_annotation()
    dJ = ReducedFunctional(J, Control(rho)).derivative()
    g = dJ.dat.data_ro.copy()
    tape.clear_tape()
    return float(J), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--strategy", default="tape", choices=["tape", "checkpoint"])
    ap.add_argument("--mode", default="grad", choices=["grad", "forward"])
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    p = Problem(nx=a.nx, ny=a.ny)
    h = Heat(p)
    res = {"backend": "firedrake", "strategy": a.strategy, "mode": a.mode, "nx": a.nx, "ny": a.ny, "steps": a.steps,
           "n_dofs": h.V.dim() - (p.ny + 1), "exact_J": p.exact_J(), "exact_dJ_uniform": p.exact_dJ_uniform()}
    try:
        # Warm-up at 10 steps; its peak is the baseline (see the JAX backend).
        t0 = time.perf_counter()
        if a.mode == "grad":
            gradient(h, 10, a.strategy)
        else:
            with stop_annotating():
                h.run(Function(h.D).assign(1.0), 10)
        res["compile_s"] = time.perf_counter() - t0
        res["baseline_rss_mb"] = peak_rss_mb()
        t0 = time.perf_counter()
        if a.mode == "forward":
            with stop_annotating():
                res["J"] = h.run(Function(h.D).assign(1.0), a.steps)
        else:
            res["J"], g = gradient(h, a.steps, a.strategy)
            res["dJ_uniform"] = float(g.sum())
        res["warm_s"] = time.perf_counter() - t0
        if a.mode == "grad" and a.fd:
            d = np.random.default_rng(0).standard_normal(len(g))
            d /= np.linalg.norm(d)
            eps = 1e-4
            with stop_annotating():
                fp, fm = Function(h.D), Function(h.D)
                fp.dat.data[:] = 1.0 + eps * d
                fm.dat.data[:] = 1.0 - eps * d
                res["fd_dir"] = (h.run(fp, a.steps) - h.run(fm, a.steps)) / (2 * eps)
            res["grad_dir"] = float(g @ d)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    res["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
