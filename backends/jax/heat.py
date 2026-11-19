"""JAX backend: Q1 FEM, backward Euler, gradient of J with respect to the per-cell conductivity field rho.

    python backends/jax/heat.py --nx 32 --ny 16 --steps 1000 --strategy tape --out result.json

Strategies (reverse mode through the time loop):
  tape        plain jax.grad through lax.scan: every step's state is kept, so memory grows with the step count
  checkpoint  two-level scan, the inner scan wrapped in jax.checkpoint: chunks of c steps, c the largest divisor
              of steps not above isqrt(steps); about steps / c + c states are kept and each chunk is recomputed
              once on the way back
Each step solves (M + dt K) T_{n+1} = M T_n by conjugate gradients (Jacobi preconditioner); jax.scipy's cg is a
custom_linear_solve, so its derivative is an implicit transposed solve, not a tape of CG iterations.
"""

import argparse
import json
import math
import os
import resource
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from spike.problem import Problem, element_matrices, initial_temperature, mesh  # noqa: E402

CG_TOL = 1e-12


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


def build(p: Problem, steps: int, strategy: str):
    coords, cells, free = mesh(p)
    ke, me = (jnp.asarray(a) for a in element_matrices(p.Lx / p.nx, p.Ly / p.ny))
    cells = jnp.asarray(cells)
    n = len(coords)
    mask = jnp.zeros(n).at[jnp.asarray(free)].set(1.0)  # Dirichlet nodes (x = 0) held at zero
    T0 = jnp.asarray(initial_temperature(coords, p)) * mask
    dt = p.t_end / steps

    def scatter(local):  # (n_cells, 4) -> (n,)
        return jnp.zeros(n).at[cells.ravel()].add(local.ravel())

    # Both sides masked, so the free-node block is symmetric (cg's derivative solves with the same operator).
    def M_mv(v):
        return scatter((v * mask)[cells] @ me.T) * mask

    def K_mv(rho, v):
        return scatter(p.k0 * rho[:, None] * ((v * mask)[cells] @ ke.T)) * mask

    # diagonal of A for a Jacobi preconditioner
    def A_diag(rho):
        d = scatter(jnp.broadcast_to(jnp.diag(me), (p.n_cells, 4)) + dt * p.k0 * rho[:, None] * jnp.diag(ke))
        return jnp.where(mask > 0, d, 1.0)

    def step(rho, dinv, T):
        def A_mv(v):
            return (M_mv(v) + dt * K_mv(rho, v)) * mask + (1 - mask) * v

        Tn, _ = jax.scipy.sparse.linalg.cg(A_mv, M_mv(T), x0=T, tol=CG_TOL, M=lambda v: dinv * v)
        return Tn

    def objective(rho):
        dinv = 1.0 / A_diag(rho)

        def body(T, _):
            return step(rho, dinv, T), None

        if strategy == "tape":
            T, _ = jax.lax.scan(body, T0, None, length=steps)
        elif strategy == "checkpoint":
            c = max(d for d in range(1, math.isqrt(steps) + 1) if steps % d == 0)

            @jax.checkpoint
            def chunk(T, _):
                T, _ = jax.lax.scan(body, T, None, length=c)
                return T, None

            T, _ = jax.lax.scan(chunk, T0, None, length=steps // c)
        else:
            raise ValueError(strategy)
        return T @ M_mv(T)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--strategy", default="tape", choices=["tape", "checkpoint"])
    ap.add_argument("--mode", default="grad", choices=["grad", "forward"])
    ap.add_argument("--fd", action="store_true", help="also check against central differences (two extra forward runs)")
    ap.add_argument("--out")
    a = ap.parse_args()

    p = Problem(nx=a.nx, ny=a.ny)
    rho = jnp.ones(p.n_cells)
    J = build(p, a.steps, a.strategy)
    # Warm-up at 10 steps; its peak is the baseline, so peak_rss_mb - baseline_rss_mb is what the measured run
    # needed on top of a process that has JAX loaded and has compiled a gradient of this mesh.
    t0 = time.perf_counter()
    warm = build(p, 10, a.strategy)
    jax.block_until_ready(jax.jit(jax.value_and_grad(warm) if a.mode == "grad" else warm)(rho))
    compile_s = time.perf_counter() - t0
    baseline = peak_rss_mb()
    res = {"backend": "jax", "strategy": a.strategy, "mode": a.mode, "nx": a.nx, "ny": a.ny, "steps": a.steps,
           "n_dofs": int(len(mesh(p)[2])), "exact_J": p.exact_J(), "exact_dJ_uniform": p.exact_dJ_uniform(),
           "compile_s": compile_s, "baseline_rss_mb": baseline}
    try:
        if a.mode == "forward":
            f = jax.jit(J)
            t0 = time.perf_counter(); val = float(f(rho)); res["first_s"] = time.perf_counter() - t0
            t0 = time.perf_counter(); f(rho).block_until_ready(); res["warm_s"] = time.perf_counter() - t0
            res["J"] = val
        else:
            vg = jax.jit(jax.value_and_grad(J))
            t0 = time.perf_counter(); val, g = vg(rho); g = np.asarray(g); res["first_s"] = time.perf_counter() - t0
            t0 = time.perf_counter(); jax.block_until_ready(vg(rho)); res["warm_s"] = time.perf_counter() - t0
            res["J"] = float(val)
            res["dJ_uniform"] = float(g.sum())
            if a.fd:
                d = np.random.default_rng(0).standard_normal(p.n_cells)
                d /= np.linalg.norm(d)
                fj = jax.jit(J)
                eps = 1e-4
                fd = (float(fj(rho + eps * d)) - float(fj(rho - eps * d))) / (2 * eps)
                res["fd_dir"] = fd
                res["grad_dir"] = float(g @ d)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 - record the failure (e.g. out of memory) as a result
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    res["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
