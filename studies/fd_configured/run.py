"""Configured-workload FD study; see PREREG.md. Run with the transient-heat-spike venv; writes results.csv."""
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
STEPS = 50
p, KE_J, M_J, X, _, _, _ = setup(8, 4)
KE, M = np.asarray(KE_J), np.asarray(M_J)
T0 = np.sin(np.pi * X / (2 * p.Lx)).astype(np.float32).astype(np.float64)
K05 = 1e-3 + (1 - 1e-3) * 0.5**3
DT = 1.0 / (K05 * (np.pi / (2 * p.Lx)) ** 2) / STEPS


def simp(rho, xp=np):
    return 1e-3 + (1 - 1e-3) * xp.clip(rho, 0.0, 1.0) ** 3


def assemble(rho):
    return M + DT * np.einsum("e,eij->ij", simp(rho), KE)


def forward(rho, rtol):
    a, u, counts, states = assemble(rho), T0, [], [T0]
    for _ in range(STEPS):
        n = 0

        def cb(_):
            nonlocal n
            n += 1
        u, info = cg(a, M @ u, x0=u, M=np.diag(1 / a.diagonal()), rtol=rtol, atol=0.0, callback=cb, maxiter=5000)
        assert info == 0
        counts.append(n)
        states.append(u)
    return states, counts


def J(u):  # Dirichlet nodes are zero, so the free-node sum is the nodal sum
    return float(u @ u)


def exact_grad(rho):
    a = assemble(rho)
    states = [T0]
    for _ in range(STEPS):
        states.append(np.linalg.solve(a, M @ states[-1]))
    cot, g = 2 * states[-1], np.zeros(p.n_cells)
    for i in range(STEPS, 0, -1):
        lam = np.linalg.solve(a.T, cot)
        g -= DT * np.einsum("i,eij,j->e", lam, KE, states[i])
        cot = M.T @ lam
    dk = 3 * (1 - 1e-3) * np.clip(rho, 0, 1) ** 2
    return g * dk


def unrolled_grad(rho, counts):
    def loss(r):
        a = M_J + DT * jnp.einsum("e,eij->ij", simp(r, jnp), KE_J)
        u = jnp.asarray(T0)
        for n in counts:
            u = pcg(a, M_J @ u, u, 1 / jnp.diag(a), n)
        return u @ u
    return np.asarray(jax.grad(loss)(jnp.asarray(rho)))


def directions():
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    d = np.stack([np.asarray(jax.random.normal(k, (p.n_cells,), dtype=jnp.float32)) for k in keys]).astype(np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main():
    dirs = directions()
    w = None
    with open(Path(__file__).with_name("results.csv"), "w", newline="") as fh:
        for seed in [0, *range(1000, 1030)]:
            rng = np.random.default_rng(seed)
            rho = np.clip(0.5 + 0.3 * rng.standard_normal(p.n_cells).astype(np.float32), 0.05, 0.95)
            rho = rho.astype(np.float32).astype(np.float64)
            ref = exact_grad(rho)
            rms = np.sqrt(np.mean(rho**2) + 1e-30)
            for rtol in (1e-3, 1e-6):
                states, counts = forward(rho, rtol)
                cand = {"unrolled": unrolled_grad(rho, tuple(counts)), "exact": ref}
                for dtype in ("f64", "f32"):
                    rnd = (lambda v: v) if dtype == "f64" else (lambda v: float(np.float32(v)))
                    meds = {c: [] for c in cand}
                    coss = {c: [] for c in cand}
                    iters = []
                    for eps in EPS:
                        h = eps * rms
                        fd = []
                        for d in dirs:
                            sp, cp = forward(rho + h * d, rtol)
                            sm, cm = forward(rho - h * d, rtol)
                            fd.append((rnd(J(sp[-1])) - rnd(J(sm[-1]))) / (2 * h))
                            iters.append(int(np.sum(cp) != np.sum(counts)) + int(np.sum(cm) != np.sum(counts)))
                        fd = np.array(fd)
                        for c, g in cand.items():
                            pr = dirs @ g
                            meds[c].append(float(np.median(np.abs(fd - pr) / np.maximum(np.maximum(abs(fd), abs(pr)), 1e-30))))
                            coss[c].append(float(fd @ pr / (np.linalg.norm(fd) * np.linalg.norm(pr) + 1e-30)))
                    row = dict(seed=seed, rtol=rtol, dtype=dtype, iters_total=sum(counts),
                               iters_min=min(counts), iters_max=max(counts),
                               fd_solves_with_other_counts=sum(iters),
                               state_err=np.linalg.norm(states[-1] - forward(rho, 1e-14)[0][-1]) / np.linalg.norm(states[-1]))
                    for c, g in cand.items():
                        row[f"{c}_E"] = np.linalg.norm(g - ref) / np.linalg.norm(ref)
                        row[f"{c}_minmed"] = min(meds[c])
                        row[f"{c}_maxcos"] = max(coss[c])
                        row[f"{c}_pass"] = min(meds[c]) <= 1e-2 and max(coss[c]) >= 0.99
                    if w is None:
                        w = csv.DictWriter(fh, fieldnames=list(row))
                        w.writeheader()
                    w.writerow(row)
                    fh.flush()
            print("seed", seed, flush=True)


if __name__ == "__main__":
    main()
