"""Same-tolerance FD check through the served torch-fem thermal-transient backend.

For fresh lognormal density fields, both ``adjoint_mode`` values, ``cg_rtol`` in
{1e-3, 1e-6} and two workloads (one step of t_end/100, and the 50-step
``gradient/fd_check`` workload of thermal_transient/config.py), this records

- E, the relative L2 error of the full served gradient of mosaic's loss sum(J^2)
  against an exact discrete adjoint (dense solves of the same Q1 / backward Euler
  scheme, assembled here without torch-fem), and
- the FD-check verdict mosaic would give: mosaic's own ``_fd_vjp_arrays`` (central
  FD of the served forward at the same tolerance, the config's eps grid and
  direction count) scored by the problem's ``gradient/fd_check`` status checks.
  The peer-median check needs three solvers and is inactive with one.

The exact gradient itself is scored against the same FD values as a third
candidate. The tesseract is loaded in-process (``Tesseract.from_tesseract_api``),
so the calls go through the schema validation and the tesseract-jax VJP path of a
served backend without Docker. Needs torch and torch-fem (a CPU wheel is enough) and the
mosaic fork with the torch-fem transient backend (github.com/singhharsh1708/mosaic, branch
feat/thermal-transient-torch; the archived results come from commit 9ef61ca, the same
patch as 66f4ce6 after a rebase), installed in the
same environment, with ``MOSAIC_CHECKOUT`` pointing at that checkout::

    MOSAIC_CHECKOUT=<mosaic checkout> python studies/fd_served/fd_served.py [--fields 30]

writes results.csv (one row per field, workload, tolerance and candidate) and
summary.txt (the counts table).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from tesseract_core import Tesseract

from mosaic.benchmarks.core.experiment import random_direction
from mosaic.benchmarks.core.status import OK, Cell, _refine_fd_check
from mosaic.benchmarks.problems.shared.gradient import _fd_vjp_arrays, fd_eps_sweep
from mosaic.benchmarks.problems.thermal_transient.config import _SOLVERS, problem
from mosaic.benchmarks.problems.thermal_transient.references import (
    K_MIN_RATIO,
    P_EXP,
    default_t_end,
    discrete_dloss_drho_uniform,
    discrete_output,
)

_API = (
    Path(os.environ["MOSAIC_CHECKOUT"])
    / "mosaic/tesseracts/thermal-mesh/torch-fem-thermal/tesseract_api.py"
)
_SOLVER = _SOLVERS["torch_fem_thermal"].name
# gradient/fd_check in thermal_transient/config.py
_SLAB = {"nx": 8, "ny": None, "nz": 1, "Lx": 2.0, "Ly": 1.0, "rho_0": 0.5}
_EPS = [1e-1, 1e-2, 1e-3, 1e-4]
_N_DIRS = 6
_CHECKS = problem.status_checks["gradient/fd_check"]
_T_END = default_t_end(_SLAB["Lx"], _SLAB["rho_0"])
WORKLOADS = {
    "short_1step": {"steps": 1, "t_end": _T_END / 100},
    "fd_check_50step": {"steps": 50},
}
RTOLS = (1e-3, 1e-6)
MODES = ("implicit", "unrolled")
FIELD_SEEDS = range(2000, 2100)  # fresh: not used by the earlier studies


def lognormal_rho(seed: int, n_cells: int) -> np.ndarray:
    """Density 0.5 exp(0.3 z), clipped to [0.05, 0.95] so no cell sits at a clip."""
    z = np.random.default_rng(seed).standard_normal(n_cells)
    return np.clip(0.5 * np.exp(0.3 * z), 0.05, 0.95).astype(np.float32)


# ── Exact discrete adjoint (dense, independent of torch-fem) ─────────────────

_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=float,
)


def _element_matrices(xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-conductivity stiffness and unit-capacity mass of one trilinear hexahedron."""
    s = 2.0 * _CORNERS - 1.0
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    Ke, Me = np.zeros((8, 8)), np.zeros((8, 8))
    for q in np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T:
        N = np.prod(1.0 + s * q, axis=1) / 8.0
        dN = np.empty((8, 3))
        for d in range(3):
            o = [e for e in range(3) if e != d]
            dN[:, d] = s[:, d] * np.prod(1.0 + s[:, o] * q[o], axis=1) / 8.0
        J = dN.T @ xe  # d x / d xi
        dNx = dN @ np.linalg.inv(J).T
        w = abs(np.linalg.det(J))
        Ke += w * dNx @ dNx.T
        Me += w * np.outer(N, N)
    return Ke, Me


class DenseScheme:
    """Backward Euler of M dT/dt + K(rho) T = 0 with T = 0 on the Dirichlet nodes."""

    def __init__(self, inputs: dict) -> None:
        hm, bc = inputs["hex_mesh"], inputs["boundary_conditions"]
        pts = np.asarray(hm["points"][: hm["n_points"]], dtype=np.float64)
        cells = np.asarray(hm["faces"][: hm["n_faces"]], dtype=np.int64)
        assert not np.any(inputs["source"]), "the reference has no source term"
        assert not np.any(bc["neumann"]["values"]), "the reference has no heat flux"
        assert not np.any(bc["dirichlet"]["values"]), (
            "the reference has T = 0 on Gamma_D"
        )
        free = np.asarray(bc["dirichlet"]["mask"])[: len(pts)] == 0
        n = len(pts)
        self.free = np.nonzero(free)[0]
        P = np.zeros((n, len(self.free)))
        P[self.free, np.arange(len(self.free))] = 1.0
        self.Ke = np.zeros((len(cells), len(self.free), len(self.free)))
        M = np.zeros((n, n))
        for e, c in enumerate(cells):
            ke, me = _element_matrices(pts[c])
            Kf = np.zeros((n, n))
            Kf[np.ix_(c, c)] = ke
            self.Ke[e] = P.T @ Kf @ P
            M[np.ix_(c, c)] += me
        self.M = P.T @ M @ P
        self.T0 = np.asarray(inputs["initial_temperature"], dtype=np.float64)[self.free]
        self.dt = float(inputs["dt"][0])
        self.steps = int(inputs["steps"])

    def loss_and_grad(self, rho: np.ndarray) -> tuple[float, np.ndarray]:
        """J = sum_nodes T_N^2 and the gradient of J^2 with respect to rho."""
        rho = np.asarray(rho, dtype=np.float64)
        w = K_MIN_RATIO + (1 - K_MIN_RATIO) * rho**P_EXP  # k_max = 1
        A = self.M + self.dt * np.einsum("e,eij->ij", w, self.Ke)
        chol = cho_factor(A)
        states = [self.T0]
        for _ in range(self.steps):
            states.append(cho_solve(chol, self.M @ states[-1]))
        J = float(states[-1] @ states[-1])
        cot, g_w = 2.0 * states[-1], np.zeros_like(w)
        for i in range(self.steps, 0, -1):
            lam = cho_solve(chol, cot)  # A is symmetric
            g_w -= self.dt * np.einsum("i,eij,j->e", lam, self.Ke, states[i])
            cot = self.M @ lam
        dJ = g_w * P_EXP * (1 - K_MIN_RATIO) * rho ** (P_EXP - 1)
        return J, 2.0 * J * dJ


def check_reference() -> None:
    """The dense scheme reproduces the closed-form discrete reference at uniform rho."""
    rho = np.full(32, 0.5, dtype=np.float32)
    for steps in (1, 50):
        inputs = problem.make_inputs(_SOLVER, rho, **_SLAB, steps=steps)
        J, g = DenseScheme(inputs).loss_and_grad(rho)
        J_ref = float(discrete_output(None, _T_END, nx=8, steps=steps))
        d_ref = discrete_dloss_drho_uniform(None, _T_END, nx=8, steps=steps)
        # The served inputs carry dt in float32 (relative rounding up to 6e-8); the
        # closed form uses the float64 dt.
        assert abs(J / J_ref - 1) < 1e-6, (J, J_ref)
        assert abs(g.sum() / d_ref - 1) < 1e-6, (g.sum(), d_ref)


# ── mosaic's FD check through the served backend ─────────────────────────────


def verdict(vjp_arr: np.ndarray, fd_per_eps: dict) -> tuple[str, str, float, float]:
    """Mosaic's fd_check status for one solver: (status, reason, best rel_err, best cos)."""
    sweep = fd_eps_sweep(vjp_arr, fd_per_eps)
    cells = {_SOLVER: Cell(OK)}
    _refine_fd_check({"by_solver": {_SOLVER: {"eps_sweep": sweep}}}, cells, _CHECKS)
    best_rel = min(float(np.median(s["rel_error"])) for s in sweep.values())
    best_cos = max(float(s["cosine"]) for s in sweep.values())
    return cells[_SOLVER].status, cells[_SOLVER].reason, best_rel, best_cos


def run_case(
    t: Tesseract,
    rho: np.ndarray,
    workload: str,
    rtol: float,
    ref: tuple[float, np.ndarray],
) -> list[dict]:
    """Both adjoint modes and the exact gradient, scored by one FD sweep each."""
    physics = {**_SLAB, **WORKLOADS[workload]}
    rows = []
    fd_values = None
    for mode in MODES:

        def make_inputs(
            name: str, ic: np.ndarray, _mode: str = mode, **phys: Any
        ) -> dict:
            return {
                **problem.make_inputs(name, ic, **phys),
                "cg_rtol": rtol,
                "adjoint_mode": _mode,
            }

        ctx = SimpleNamespace(
            name=_SOLVER,
            ic=rho,
            phys=physics,
            ic_key="rho",
            output_key="identification_error",
            seed=0,
            make_inputs=make_inputs,
        )
        _, _, g, vjp_arr, fd_per_eps = _fd_vjp_arrays(t, ctx, _EPS, _N_DIRS)
        if fd_values is None:
            fd_values = fd_per_eps
        else:  # the forward does not depend on adjoint_mode
            for eps in _EPS:
                assert np.array_equal(fd_values[eps], fd_per_eps[eps])
        rows.append((mode, np.asarray(g, dtype=np.float64), vjp_arr))

    J_exact, g_exact = ref
    keys = jax.random.split(jax.random.PRNGKey(0), _N_DIRS)
    dirs = np.stack([np.asarray(random_direction(rho.shape, k)) for k in keys])
    rows.append(("exact", g_exact, dirs.astype(np.float64) @ g_exact))

    out = []
    for cand, g, vjp_arr in rows:
        status, reason, best_rel, best_cos = verdict(vjp_arr, fd_values)
        out.append(
            {
                "workload": workload,
                "cg_rtol": rtol,
                "candidate": cand,
                "E": float(np.linalg.norm(g - g_exact) / np.linalg.norm(g_exact)),
                "best_median_rel_err": best_rel,
                "best_cosine": best_cos,
                "status": status,
                "reason": reason,
                "J_exact": J_exact,
            }
        )
    return out


def summarise(rows: list[dict], n_fields: int) -> str:
    """The counts table, one line per workload, tolerance and candidate."""
    lines = [
        (
            f"{n_fields} lognormal density fields (seeds {FIELD_SEEDS.start}.."
            f"{FIELD_SEEDS.start + n_fields - 1}), nx=8, mosaic fd_check eps {_EPS}, "
            f"{_N_DIRS} directions, status checks of gradient/fd_check (peer check "
            "inactive)."
        ),
        "",
        (
            "| workload | cg_rtol | gradient | E min..max | E > 10% | E > 1% |"
            " FD passes | passes with E > 10% | fails with E < 1% |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for workload in WORKLOADS:
        for rtol in RTOLS:
            for cand in (*MODES, "exact"):
                sel = [
                    r
                    for r in rows
                    if r["workload"] == workload
                    and r["cg_rtol"] == rtol
                    and r["candidate"] == cand
                ]
                E = np.array([r["E"] for r in sel])
                ok = np.array([r["status"] == OK for r in sel])
                n = len(sel)
                lines.append(
                    f"| {workload} | {rtol:g} | {cand} | {E.min():.2g}..{E.max():.2g} |"
                    f" {int((E > 0.1).sum())}/{n} | {int((E > 0.01).sum())}/{n} |"
                    f" {int(ok.sum())}/{n} | {int((ok & (E > 0.1)).sum())}/{n} |"
                    f" {int((~ok & (E < 0.01)).sum())}/{n} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the study and write results.csv and summary.txt."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    check_reference()
    t = Tesseract.from_tesseract_api(_API)
    rows: list[dict] = []
    t0 = time.time()
    for seed in FIELD_SEEDS[: args.fields]:
        rho = lognormal_rho(seed, 32)
        for workload, physics in WORKLOADS.items():
            inputs = problem.make_inputs(_SOLVER, rho, **_SLAB, **physics)
            ref = DenseScheme(inputs).loss_and_grad(rho)
            for rtol in RTOLS:
                for row in run_case(t, rho, workload, rtol, ref):
                    rows.append({"seed": seed, **row})
        print(
            f"seed {seed} done, {time.time() - t0:.0f} s", file=sys.stderr, flush=True
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, args.fields)
    (args.out / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
