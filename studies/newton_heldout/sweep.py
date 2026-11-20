"""Held-out Newton study; criteria in PREREG.md. Needs optimistix==0.1.0 next to the spike's dependencies. python -B sweep.py > results.csv"""

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from scipy.linalg import solve

SPIKE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPIKE))
from spike.problem import Problem, assemble_dense, element_matrices, initial_temperature, mesh  # noqa: E402

jax.config.update("jax_enable_x64", True)

LAWS = {  # g(T), g'(T); numpy and jax share the expressions
    "linear": (lambda t, xp: 1 + t, lambda t, xp: xp.ones_like(t)),
    "exp": (lambda t, xp: xp.exp(t), lambda t, xp: xp.exp(t)),
    "quadratic": (lambda t, xp: 1 + 2 * t**2, lambda t, xp: 4 * t),
}


def discretization(p, field):
    coords, cells, free = mesh(p)
    mapping = np.full(len(coords), len(free))
    mapping[free] = np.arange(len(free))
    ke, _ = element_matrices(p.Lx / p.nx, p.Ly / p.ny)
    _, mass = assemble_dense(p, np.ones(p.n_cells))
    rng = np.random.default_rng(field)
    rho = np.exp(0.4 * rng.standard_normal(p.n_cells))
    t0 = initial_temperature(coords, p)[free]
    t0 = t0 / np.sqrt(t0 @ mass @ t0)
    noise = rng.standard_normal(t0.size)
    t0 = t0 + 0.3 * noise / np.sqrt(noise @ mass @ noise)
    return mapping[cells], ke, mass, t0, rho


def reference(ids, ke, mass, start, rho, dt, steps, law):
    g, dg = LAWS[law]
    n = len(start) + 1
    pairs = (n * ids[:, :, None] + ids[:, None, :]).ravel()

    def linearize(u, old, r):
        local = np.append(u, 0.)[ids]
        ku = local @ ke.T
        tbar = local.mean(axis=1)
        flux = (r * g(tbar, np))[:, None] * ku
        summed = np.bincount(ids.ravel(), flux.ravel(), minlength=n)
        residual = mass @ (u - old) + dt * summed[:-1]
        blocks = (r * g(tbar, np))[:, None, None] * ke + (r * dg(tbar, np))[:, None, None] * ku[:, :, None] / 4
        stiff = np.bincount(pairs, blocks.ravel(), minlength=n * n).reshape(n, n)
        return residual, mass + dt * stiff[:-1, :-1], flux / r[:, None]

    def forward(r):
        states = [start]
        for _ in range(steps):
            old = states[-1]
            u = old.copy()
            for _ in range(30):
                res, jac, _ = linearize(u, old, r)
                c = solve(jac, res)
                u -= c
                if np.max(np.abs(c)) <= 1e-14 * max(1, np.max(np.abs(u))):
                    break
            else:
                raise ArithmeticError("reference Newton did not converge")
            states.append(u)
        return states

    states = forward(rho)
    cot, grad = 2 * mass @ states[-1], np.zeros(len(ids))
    for i in range(steps, 0, -1):
        _, jac, dflux = linearize(states[i], states[i - 1], rho)
        adj = solve(jac.T, cot)
        grad -= dt * np.einsum("ei,ei->e", np.append(adj, 0.)[ids], dflux)
        cot = mass.T @ adj
    d = np.random.default_rng(7).standard_normal(len(rho))
    d /= np.linalg.norm(d)
    h = 1e-6
    jp, jm = (forward(rho * (1 + s * h * d))[-1] for s in (1, -1))  # relative perturbation keeps rho > 0
    fd = (jp @ mass @ jp - jm @ mass @ jm) / (2 * h)
    fd_err = abs(fd - grad @ (rho * d)) / abs(fd)
    return states[-1] @ mass @ states[-1], grad, fd_err


def optimistix_gradient(ids, ke, mass, start, rho, dt, steps, law, tol, mode):
    g, _ = LAWS[law]
    ids, ke, mass, start = map(jnp.asarray, (ids, ke, mass, start))
    solver = optx.Newton(rtol=tol, atol=tol)
    kwargs = {} if mode == "implicit" else {"adjoint": optx.RecursiveCheckpointAdjoint()}

    def residual(u, args):
        r, old = args
        local = jnp.append(u, 0.)[ids]
        flux = (r * g(local.mean(axis=1), jnp))[:, None] * (local @ ke.T)
        return mass @ (u - old) + dt * jnp.zeros(start.size + 1).at[ids].add(flux)[:-1]

    def loss(r):
        def step(old, _):
            sol = optx.root_find(residual, solver, old, args=(r, old), max_steps=64, throw=False, **kwargs)
            return sol.value, (sol.stats["num_steps"], sol.result == optx.RESULTS.successful)
        terminal, diag = jax.lax.scan(step, start, None, length=steps)
        return terminal @ mass @ terminal, diag

    (value, (iters, ok)), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(jnp.asarray(rho))
    return float(value), np.asarray(grad), np.asarray(iters), bool(jnp.all(ok))


def main():
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(("nx", "ny", "field", "law", "steps", "tol", "adjoint", "converged", "min_iters", "max_iters",
                "J_rel_err", "E", "ref_fd_err"))
    for nx, ny in ((12, 6), (20, 10)):
        p = Problem(nx=nx, ny=ny)
        for field in (200, 201, 202):
            ids, ke, mass, t0, rho = discretization(p, field)
            for law in LAWS:
                for steps in (10, 100):
                    dt = p.t_end / steps
                    ref_j, ref, fd_err = reference(ids, ke, mass, t0, rho, dt, steps, law)
                    for tol in (1e-2, 1e-4, 1e-8):
                        for mode in ("implicit", "unrolled"):
                            j, gr, it, ok = optimistix_gradient(ids, ke, mass, t0, rho, dt, steps, law, tol, mode)
                            w.writerow((nx, ny, field, law, steps, tol, mode, ok, it.min(), it.max(),
                                        abs(j - ref_j) / ref_j, np.linalg.norm(gr - ref) / np.linalg.norm(ref),
                                        fd_err))
                            sys.stdout.flush()


if __name__ == "__main__":
    main()
