"""Held-out FD check study; criteria in PREREG.md. Run with the transient-heat-spike venv. Writes results.csv."""

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pcg, setup  # noqa: E402

jax.config.update("jax_enable_x64", True)
EPS = (1e-1, 1e-2, 1e-3, 1e-4)
RTOLS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-8)
SCHEDULES = ((1, 1.0), (1, 0.01), (10, 0.1))  # (steps, dt / t_end)
FIELDS = range(100, 105)
PROBES = range(4)


def solve_cg(a, rhs, start, rtol):
    n = 0

    def count(_):
        nonlocal n
        n += 1
    x, info = cg(a, rhs, x0=start, M=np.diag(1.0 / a.diagonal()), rtol=rtol, atol=0.0, callback=count, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"CG info={info}")
    return x, n


def run_case(nx, ny, field, steps, frac, rtol, writer, fd_rtol=None):
    p, ke_j, m_j, x, y, _, _ = setup(nx, ny)
    ke, m = np.asarray(ke_j), np.asarray(m_j)
    rng = np.random.default_rng(field)
    k0 = np.exp(0.5 * rng.standard_normal(p.n_cells))
    mode = np.sin(np.pi * x / (2 * p.Lx))
    mode /= np.sqrt(mode @ m @ mode)
    noise = rng.standard_normal(mode.size)
    t0 = mode + 0.3 * noise / np.sqrt(noise @ m @ noise)
    right = np.isclose(x, p.Lx).astype(float)
    right /= right.sum()
    dt = frac * p.t_end
    observables = {"energy": lambda u: u @ m @ u, "right_mean": lambda u: right @ u}

    def assemble(k):
        return m + dt * np.einsum("e,eij->ij", k, ke)

    def forward(k, tol=rtol):
        a, u, counts, states = assemble(k), t0, [], [t0]
        for _ in range(steps):
            u, n = solve_cg(a, m @ u, u, tol)
            counts.append(n)
            states.append(u)
        return states, counts

    a0 = assemble(k0)
    states, counts = forward(k0)
    exact = [t0]
    for _ in range(steps):
        exact.append(np.linalg.solve(a0, m @ exact[-1]))

    def adjoint(seq, dJ):  # discrete adjoint along a state sequence; exact transpose solves
        cot, g = dJ(seq[-1]), np.zeros(p.n_cells)
        for i in range(steps, 0, -1):
            lam = np.linalg.solve(a0.T, cot)
            g -= dt * np.einsum("i,eij,j->e", lam, ke, seq[i])
            cot = m.T @ lam
        return g

    grads = {"energy": lambda u: 2 * m @ u, "right_mean": lambda u: right}
    counts_t = tuple(counts)

    def unrolled_loss(k, obs):
        a = m_j + dt * jnp.einsum("e,eij->ij", k, ke_j)
        u = jnp.asarray(t0)
        for n in counts_t:
            u = pcg(a, m_j @ u, u, 1.0 / jnp.diag(a), n)
        return u @ m_j @ u if obs == "energy" else jnp.asarray(right) @ u

    for obs, f in observables.items():
        ref = adjoint(exact, grads[obs])
        cand = {"unrolled": np.asarray(jax.grad(unrolled_loss)(jnp.asarray(k0), obs)),
                "implicit": adjoint(states, grads[obs])}
        err = {name: np.linalg.norm(g - ref) / np.linalg.norm(ref) for name, g in cand.items()}
        rms = np.sqrt(np.mean(k0**2))
        for probe in PROBES:
            keys = jax.random.split(jax.random.PRNGKey(probe), 6)
            dirs = np.stack([np.asarray(jax.random.normal(kk, (p.n_cells,))) for kk in keys])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            for eps in EPS:
                h, fd, fd_counts = eps * rms, [], []
                for d in dirs:
                    sp, cp = forward(k0 + h * d, fd_rtol or rtol)
                    sm, cm = forward(k0 - h * d, fd_rtol or rtol)
                    fd.append((f(sp[-1]) - f(sm[-1])) / (2 * h))
                    fd_counts += cp + cm
                fd = np.array(fd)
                row = dict(nx=nx, ny=ny, field=field, steps=steps, dt_frac=frac, rtol=rtol, obs=obs, probe=probe,
                           eps=eps, base_iters=f"{min(counts)}-{max(counts)}",
                           fd_iters=f"{min(fd_counts)}-{max(fd_counts)}",
                           state_err=np.linalg.norm(states[-1] - exact[-1]) / np.linalg.norm(exact[-1]))
                for name, g in {**cand, "reference": ref}.items():
                    pred = dirs @ g
                    rel = np.abs(fd - pred) / np.maximum(np.maximum(np.abs(fd), np.abs(pred)), 1e-300)
                    row[f"{name}_E"] = err.get(name, 0.0)
                    row[f"{name}_med"] = float(np.median(rel))
                    row[f"{name}_cos"] = float(fd @ pred / (np.linalg.norm(fd) * np.linalg.norm(pred)))
                writer.writerow(row)


def main():
    fd_rtol = float(sys.argv[1]) if len(sys.argv) > 1 else None
    out = Path(__file__).with_name("results.csv" if fd_rtol is None else "results_fd_tight.csv")
    with out.open("w", newline="") as fh:
        writer = None
        for nx, ny in ((12, 6), (16, 8)):
            for field in FIELDS:
                for steps, frac in SCHEDULES:
                    for rtol in (RTOLS if fd_rtol is None else RTOLS[:2]):
                        class W:
                            def writerow(self, row):
                                nonlocal writer
                                if writer is None:
                                    writer = csv.DictWriter(fh, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                        run_case(nx, ny, field, steps, frac, rtol, W(), fd_rtol)
                        fh.flush()
            print("mesh done", nx, ny, flush=True)


if __name__ == "__main__":
    main()
