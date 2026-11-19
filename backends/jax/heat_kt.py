"""mosaic#375 idea 4 follow-on: temperature-dependent conductivity, a Newton solve per step (JAX).

    python backends/jax/heat_kt.py --steps 200 --newton 3 --diff unrolled --out result.json

Same mesh, Q1 elements and objective as spike/problem.py, with k_e = k0 rho_e (1 + beta Tbar_e), Tbar_e the mean
of the cell's nodal temperatures. Each backward Euler step solves
    R(T) = M (T - T_n) + dt K(rho, T) T = 0
by `--newton` Newton iterations from T_n (dense Jacobian from jax.jacfwd and LU for the Newton systems).

Differentiation of that solve (`--diff`):
  unrolled  reverse mode through the Newton iterations as run: the gradient is exact for the iterate reached,
            so it carries the Newton error
  implicit  jax.lax.custom_root: the implicit function theorem at the iterate reached, -dR/dT^-T dR/drho, which is
            exact at a converged root and approximate otherwise
The reference (`--reference`, its own process) is the implicit gradient at 30 Newton iterations, checked against
central differences. Newton systems are solved densely (LU of the jacfwd Jacobian), which is what the unrolled
tape stores per iteration; with mosaic's sparse or iterative solvers the memory ratio would differ.
"""

import argparse
import json
import os
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from spike.problem import Problem, element_matrices, initial_temperature, mesh  # noqa: E402

BETA = 1.0


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def build(p: Problem, steps: int, n_newton: int, diff: str):
    coords, cells, free = mesh(p)
    ke, me = (jnp.asarray(a) for a in element_matrices(p.Lx / p.nx, p.Ly / p.ny))
    cells = jnp.asarray(cells)
    n = len(coords)
    mask = jnp.zeros(n).at[jnp.asarray(free)].set(1.0)
    T0 = jnp.asarray(initial_temperature(coords, p)) * mask
    dt = p.t_end / steps

    def scatter(local):
        return jnp.zeros(n).at[cells.ravel()].add(local.ravel())

    def M_mv(v):
        return scatter((v * mask)[cells] @ me.T) * mask

    def R(rho, T_old, T):
        Tm = T * mask
        k = p.k0 * rho * (1 + BETA * Tm[cells].mean(axis=1))
        KT = scatter(k[:, None] * (Tm[cells] @ ke.T)) * mask
        return M_mv(T - T_old) + dt * KT + (1 - mask) * T  # Dirichlet rows: T = 0

    def linsolve(linear_map, b):
        # Dense: the k(T) runs use small meshes, and a dense LU keeps the Newton systems exact and differentiable.
        A = jax.jacfwd(linear_map)(jnp.zeros_like(b))
        return jnp.linalg.solve(A, b)

    def newton_solve(rho, T_old, T):
        def it(T, _):
            r = R(rho, T_old, T)
            dT = linsolve(lambda v: jax.jvp(lambda t: R(rho, T_old, t), (T,), (v,))[1], r)  # J_R(T) v
            return T - dT, None

        T, _ = jax.lax.scan(it, T, None, length=n_newton)
        return T

    def step(rho, T_old):
        if diff == "unrolled":
            return newton_solve(rho, T_old, T_old)
        # implicit: custom_root differentiates the root condition R(T) = 0 instead of the iterations
        return jax.lax.custom_root(
            lambda T: R(rho, T_old, T), T_old,
            solve=lambda f, x0: newton_solve(rho, T_old, x0),
            tangent_solve=lambda g, y: linsolve(g, y),  # g is already the linear map dR/dT
        )

    def objective(rho):
        T, _ = jax.lax.scan(lambda T, _: (step(rho, T), None), T0, None, length=steps)
        return T @ M_mv(T)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--newton", type=int, default=3)
    ap.add_argument("--diff", default="unrolled", choices=["unrolled", "implicit"])
    ap.add_argument("--reference", action="store_true",
                    help="compute only the 30-iteration implicit reference and its central-difference check")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Problem(nx=a.nx, ny=a.ny)
    rho = jnp.ones(p.n_cells)
    res = {"backend": "jax", "problem": "k(T)", "beta": BETA, "diff": a.diff, "newton": a.newton, "nx": a.nx,
           "ny": a.ny, "steps": a.steps}
    try:
        d = np.random.default_rng(0).standard_normal(p.n_cells)
        d /= np.linalg.norm(d)
        if a.reference:  # in its own process, so it does not count towards a case's memory
            res.update(diff="reference", newton=30)
            ref = jax.jit(jax.value_and_grad(build(p, a.steps, 30, "implicit")))
            rv, rg = ref(rho)
            rg = np.asarray(rg)
            res["J"], res["dJ_uniform"], res["grad_dir"] = float(rv), float(rg.sum()), float(rg @ d)
            f = jax.jit(build(p, a.steps, 30, "implicit"))
            eps = 1e-5
            res["fd_dir"] = (float(f(rho + eps * d)) - float(f(rho - eps * d))) / (2 * eps)
            res["ok"] = True
            raise StopIteration
        vg = jax.jit(jax.value_and_grad(build(p, a.steps, a.newton, a.diff)))
        t0 = time.perf_counter(); val, g = vg(rho); jax.block_until_ready(g); res["first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(vg(rho)); res["warm_s"] = time.perf_counter() - t0
        g = np.asarray(g)
        res["J"], res["dJ_uniform"] = float(val), float(g.sum())
        res["grad_dir"] = float(g @ d)
        res["ok"] = True
    except StopIteration:
        pass
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
